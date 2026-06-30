#!/usr/bin/env python3
"""
episode_recorder.py — ROS 2 (Humble) benchmark recorder for the VLN comparison.

What it does, per episode:
  * subscribes to the Vicon ground-truth pose (world frame) and stores the
    dense trajectory;
  * counts executed steps, collects per-step inference latency, and counts
    safety-layer interventions (all optional, enable by setting the topics);
  * on 'stop' it loads the route's reference path + goal + threshold, computes
    every metric via vln_metrics, appends a row to a CSV laid out like the
    'Trials' sheet, and dumps the raw trajectory to .npz for offline recompute.

Control interface — publish std_msgs/String on  /benchmark/command :
    "start R01 NaVILA 1"     start episode: route, model, trial
    "stop"                   normal end (agent emitted STOP)        -> success eligible
    "abort"                  manual/safety abort                    -> success = 0
    "timeout"                step/time budget exhausted             -> success = 0

Topic names are intentionally left blank in the defaults — set them as ROS
params once you wire up your stack (see the param block below). The pose topic
is the only one required.
"""
import csv
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import yaml

from std_msgs.msg import String, Float64, Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

import vln_metrics as M


def _pose_xy(msg):
    """Extract (x, y) from PoseStamped or Odometry."""
    if isinstance(msg, Odometry):
        p = msg.pose.pose.position
    else:  # PoseStamped
        p = msg.pose.position
    return (p.x, p.y)


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("episode_recorder")

        # ---- parameters (fill the topic names later) ----
        self.declare_parameter("routes_yaml", "routes.yaml")
        self.declare_parameter("output_dir", "benchmark_runs")
        self.declare_parameter("command_topic", "/benchmark/command")
        self.declare_parameter("pose_topic", "")          # REQUIRED: Vicon GT pose
        self.declare_parameter("pose_type", "PoseStamped") # or "Odometry"
        self.declare_parameter("step_topic", "")           # e.g. /navila/primitive_status
        self.declare_parameter("step_done_token", "done")  # payload that counts as a step
        self.declare_parameter("latency_topic", "")        # Float64, milliseconds
        self.declare_parameter("safety_topic", "")         # Bool, True = intervention fired
        self.declare_parameter("dtw_spacing", 0.25)        # arc-length resample (m)
        self.declare_parameter("tl_deadband", 0.003)       # jitter filter for TL (m)

        gp = lambda n: self.get_parameter(n).value
        self.routes = self._load_routes(gp("routes_yaml"))
        self.output_dir = gp("output_dir")
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "trials_log.csv")
        self.dtw_spacing = float(gp("dtw_spacing"))
        self.tl_deadband = float(gp("tl_deadband"))
        self.step_done_token = str(gp("step_done_token"))

        # ---- episode state ----
        self._reset_state()
        self.run_order = 0

        # reliable QoS for control/metrics, sensor QoS for the high-rate pose
        rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        sensor = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(String, gp("command_topic"), self._on_command, rel)

        pose_topic = gp("pose_topic")
        if not pose_topic:
            self.get_logger().warn("pose_topic is EMPTY — set it before recording.")
        else:
            pose_cls = Odometry if gp("pose_type") == "Odometry" else PoseStamped
            self.create_subscription(pose_cls, pose_topic, self._on_pose, sensor)

        if gp("step_topic"):
            self.create_subscription(String, gp("step_topic"), self._on_step, rel)
        if gp("latency_topic"):
            self.create_subscription(Float64, gp("latency_topic"), self._on_latency, rel)
        if gp("safety_topic"):
            self.create_subscription(Bool, gp("safety_topic"), self._on_safety, rel)

        self.get_logger().info(
            "Episode recorder ready. Command on '%s': "
            "'start <route> <model> <trial>' / 'stop' / 'abort' / 'timeout'."
            % gp("command_topic"))

    # ---------------------------------------------------------------- config
    def _load_routes(self, path):
        if not os.path.exists(path):
            self.get_logger().warn(f"routes_yaml '{path}' not found — "
                                   "episodes will fail to resolve goals.")
            return {}
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("routes", data)

    def _reset_state(self):
        self.active = False
        self.route_id = self.model = self.trial = None
        self.traj = []          # dense GT (x, y)
        self.latencies = []     # ms
        self.n_steps = 0
        self.n_safety = 0
        self.t_start = None
        self._prev_safety = False

    # ---------------------------------------------------------------- inputs
    def _on_pose(self, msg):
        if self.active:
            self.traj.append(_pose_xy(msg))

    def _on_step(self, msg):
        if self.active and msg.data.strip().lower() == self.step_done_token.lower():
            self.n_steps += 1

    def _on_latency(self, msg):
        if self.active:
            self.latencies.append(float(msg.data))

    def _on_safety(self, msg):
        if self.active and msg.data and not self._prev_safety:
            self.n_safety += 1
        self._prev_safety = bool(msg.data) if self.active else False

    # ---------------------------------------------------------------- control
    def _on_command(self, msg):
        parts = msg.data.split()
        if not parts:
            return
        cmd = parts[0].lower()
        if cmd == "start":
            if len(parts) < 4:
                self.get_logger().error("start needs: start <route> <model> <trial>")
                return
            self._start(parts[1], parts[2], parts[3])
        elif cmd in ("stop", "abort", "timeout"):
            self._finish(stopped=(cmd == "stop"), aborted=(cmd != "stop"), reason=cmd)
        else:
            self.get_logger().warn(f"unknown command '{cmd}'")

    def _start(self, route_id, model, trial):
        if route_id not in self.routes:
            self.get_logger().error(f"route '{route_id}' not in routes_yaml")
            return
        self._reset_state()
        self.active = True
        self.route_id, self.model, self.trial = route_id, model, trial
        self.t_start = self.get_clock().now()
        self.get_logger().info(f"[START] {route_id} | {model} | trial {trial}")

    def _finish(self, stopped, aborted, reason):
        if not self.active:
            self.get_logger().warn("no active episode")
            return
        self.active = False
        route = self.routes[self.route_id]
        reference = route["reference_path"]      # list of [x, y] in Vicon world frame
        goal = route.get("goal", reference[-1])
        d_th = float(route.get("threshold", 1.0))
        ref_len = route.get("ref_len")           # optional explicit override

        traj = np.array(self.traj, dtype=float) if self.traj else np.empty((0, 2))
        tl = M.trajectory_length(traj, min_step=self.tl_deadband)
        if ref_len is None:
            ref_len = M.trajectory_length(reference, min_step=0.0)
        succ = M.is_success(traj, goal, d_th, stopped=stopped)
        metrics = {
            "success": succ,
            "oracle_success": M.oracle_success(traj, goal, d_th),
            "ne_m": M.navigation_error(traj, goal),
            "tl_m": tl,
            "ref_len_m": float(ref_len),
            "spl": M.spl(succ, ref_len, tl),
            "ndtw": M.ndtw(traj, reference, d_th, spacing=self.dtw_spacing),
        }
        metrics["sdtw"] = succ * metrics["ndtw"]
        lat_mean = float(np.mean(self.latencies)) if self.latencies else float("nan")
        dur = (self.get_clock().now() - self.t_start).nanoseconds / 1e9

        self.run_order += 1
        row = {
            "trial_id": f"{self.route_id}-{self.model}-{self.trial}",
            "route_id": self.route_id,
            "model": self.model,
            "trial": self.trial,
            "run_order": self.run_order,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": metrics["success"],
            "oracle": metrics["oracle_success"],
            "ne_m": round(metrics["ne_m"], 3),
            "ref_len_m": round(metrics["ref_len_m"], 3),
            "tl_m": round(metrics["tl_m"], 3),
            "spl": round(metrics["spl"], 4),
            "ndtw": round(metrics["ndtw"], 4),
            "sdtw": round(metrics["sdtw"], 4),
            "steps": self.n_steps,
            "latency_ms": round(lat_mean, 1) if np.isfinite(lat_mean) else "",
            "safety_interv": self.n_safety,
            "aborted": int(aborted),
            "duration_s": round(dur, 1),
            "end_reason": reason,
        }
        self._append_csv(row)
        self._dump_traj(traj, reference, goal, d_th)
        self.get_logger().info(
            "[%s] %s %s t%s | SR=%d OSR=%d NE=%.2f TL=%.2f SPL=%.3f nDTW=%.3f "
            "steps=%d lat=%s ms safety=%d"
            % (reason.upper(), self.route_id, self.model, self.trial,
               row["success"], row["oracle"], metrics["ne_m"], metrics["tl_m"],
               metrics["spl"], metrics["ndtw"], self.n_steps,
               row["latency_ms"], self.n_safety))

    # ---------------------------------------------------------------- output
    def _append_csv(self, row):
        new = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    def _dump_traj(self, traj, reference, goal, d_th):
        fn = os.path.join(
            self.output_dir,
            f"{self.route_id}_{self.model}_t{self.trial}_"
            f"{datetime.now().strftime('%H%M%S')}.npz")
        np.savez(fn, trajectory=traj, reference=np.array(reference, float),
                 goal=np.array(goal, float), threshold=d_th)


def main():
    rclpy.init()
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()