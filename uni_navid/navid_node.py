#!/usr/bin/env python3

import threading
from collections import deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Empty
from sensor_msgs.msg import CompressedImage

from uni_navid.third_party.uni_navid_agent import UniNaVid_Agent    

from huggingface_hub import snapshot_download

UNINAVID_REPO_ID = "Jzzhang/Uni-NaVid"


class UniNaVidNode(Node):
    def __init__(self, vla_model="uninavid"):
        super().__init__("uninavid_node")

        # ---- parametri ----
        self.declare_parameter("model_path", "/models")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("goal_topic", "/goal_instruction")
        self.declare_parameter("action_topic", f"/{vla_model}/action")
        self.declare_parameter("reset_topic", f"/{vla_model}/reset")
        self.declare_parameter("status_topic", f"/{vla_model}/primitive_status")

        prm = lambda n: self.get_parameter(n).value
        model_path = prm("model_path")
        image_topic = prm("image_topic")
        goal_topic = prm("goal_topic")
        action_topic = prm("action_topic")
        reset_topic = prm("reset_topic")
        status_topic = prm("status_topic")

        # ---- stato ----
        self._agent = None
        self._model_ready = False
        self._lock = threading.Lock()
        self._last_image_msg = None
        self._goal = None
        self._queue = deque()
        self._busy = False

        # ---- I/O ----
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_image = self.create_subscription(CompressedImage, image_topic, self._image_cb, qos_sensor)
        self.sub_goal = self.create_subscription(String, goal_topic, self._goal_cb, 10)
        self.sub_reset = self.create_subscription(Empty, reset_topic, self._reset_cb, 10)
        self.sub_status = self.create_subscription(String, status_topic, self._status_cb, 10)
        self.pub_action = self.create_publisher(String, action_topic, 10)

        threading.Thread(target=self._load_model_thread, args=(model_path,), daemon=True).start()

    # ================= Model loading =================
    def _load_model_thread(self, model_path):
        try:
            model_path = self.ensure_model(model_path)
            
            self._agent = UniNaVid_Agent(model_path)
            self._agent.reset(task_type="vln")
            with self._lock:
                self._model_ready = True
            self.get_logger().info("UniNaVid ready - waiting for goal instruction")
        except Exception as e:
            self.get_logger().error(f"Error while loading model: {e}")

    # ================= Inferenza =================
    def infer_action(self, frame, goal):
        # streaming: passo SOLO il frame corrente, la history la tiene il modello
        result = self._agent.act({"instruction": goal, "observations": frame})
        return result["actions"]    # es. ["forward", "forward", "left", "stop"]

    # ================= Callbacks =================
    def _image_cb(self, msg: CompressedImage):
        with self._lock:
            self._last_image_msg = msg

    def _goal_cb(self, msg: String):
        goal = msg.data.strip()
        if not goal:
            return
        self._goal = goal
        self._reset_agent()
        self.get_logger().info(f"Nuovo goal: {goal}")
        self._step()

    def _reset_cb(self, msg: Empty):
        self._goal = None
        self._reset_agent()
        self.get_logger().info("Reset")

    def _status_cb(self, msg: String):
        # il nodo action->cmd_vel segnala qui quando ha finito la primitiva
        if msg.data.strip().lower() in ("done", "idle", "ready"):
            self._busy = False
            self._step()

    # ================= Loop step-synchronous =================
    def _step(self):
        if self._busy or self._goal is None or not self._model_ready:
            return

        # coda vuota -> nuova inferenza sul frame corrente
        if not self._queue:
            frame = self._decode_latest()
            if frame is None:
                return
            actions = self.infer_action(frame, self._goal)
            if not actions:
                return
            self._queue.extend(actions)

        action = self._queue.popleft()
        self._busy = True
        self._publish_action(action)

        if action == "stop":
            self._goal = None
            self._queue.clear()
            self.get_logger().info("STOP raggiunto")

    # ================= Helper =================
    def _publish_action(self, cmd: str):
        out = String()
        out.data = cmd
        self.pub_action.publish(out)

    def _reset_agent(self):
        with self._lock:
            if self._agent is not None:
                self._agent.reset(task_type="vln")
        self._queue.clear()
        self._busy = False

    def _decode_latest(self):
        with self._lock:
            msg = self._last_image_msg
        if msg is None:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        # imdecode -> BGR, come cv2.imread nella pipeline offline di UniNaVid
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    def ensure_model(model_path: str, repo_id: str = UNINAVID_REPO_ID) -> str:
        if os.path.isdir(model_path) and os.listdir(model_path):
            return model_path
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)
        return model_path











def main(args=None):
    rclpy.init(args=args)
    node = UniNaVidNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()