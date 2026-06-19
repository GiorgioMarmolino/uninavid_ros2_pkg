#!/usr/bin/env python3
"""
action_node.py (UniNaVid)

Converte le azioni UniNaVid in comandi di velocità (geometry_msgs/Twist).

UniNaVid pubblica primitive discrete come token singoli:
    forward | left | right | stop

Semantica fissa del modello:
    forward -> avanza 0.5 m
    left    -> ruota +30  (CCW)
    right   -> ruota -30  (CW)
    stop    -> termina il task

Il nodo si occupa solo di:
    - tradurre la primitiva in una manovra closed-loop su odometria
    - deadline/failsafe sulla primitiva
    - smoothing con rampa di accelerazione
    - pubblicare a rate fisso
    - segnalare done/aborted su status_topic

Il safety layer (depth, LiDAR) e' in un nodo a parte (safety_layer_node).
cmd_vel_topic parametrico:
    safety ON  -> /cmd_vel_raw  (lo prende il safety node)
    safety OFF -> /cmd_vel      (va diretto a twist_mux)

Subscribes:
    /uninavid/action (std_msgs/String)   token: forward|left|right|stop
    <odom_topic>     (nav_msgs/Odometry)
Publishes:
    <cmd_vel_topic>            (geometry_msgs/Twist)
    /uninavid/primitive_status (std_msgs/String)  -> done | aborted
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_LINEAR_X      = 0.4    # m/s    — velocita' in "forward"
DEFAULT_ANGULAR_Z     = 0.35   # rad/s  — velocita' rotazione sul posto
DEFAULT_FORWARD_STEP  = 0.5    # m      — distanza per "forward" (UniNaVid)
DEFAULT_TURN_STEP     = 30.0   # deg    — angolo per "left"/"right" (UniNaVid)

DEFAULT_PUBLISH_RATE  = 0.05   # s      — periodo pubblicazione (20 Hz)
DEFAULT_MAX_ACC_LIN   = 1.0    # m/s²   — max accelerazione lineare
DEFAULT_MAX_ACC_ANG   = 2.0    # rad/s² — max accelerazione angolare


class ActionNode(Node):

    def __init__(self):
        super().__init__("action_node")

        # ------------------------------------------------------------------
        # Parametri ROS 2
        # ------------------------------------------------------------------
        self.declare_parameter("action_topic",  "/uninavid/action")
        self.declare_parameter("cmd_vel_topic",  "/cmd_vel")
        self.declare_parameter("odom_topic",     "/platform/odom/filtered")
        self.declare_parameter("status_topic",   "/uninavid/primitive_status")

        self.declare_parameter("linear_x",      DEFAULT_LINEAR_X)
        self.declare_parameter("angular_z",     DEFAULT_ANGULAR_Z)
        self.declare_parameter("forward_step_m", DEFAULT_FORWARD_STEP)
        self.declare_parameter("turn_step_deg",  DEFAULT_TURN_STEP)

        self.declare_parameter("publish_rate_sec", DEFAULT_PUBLISH_RATE)
        self.declare_parameter("max_acc_linear",   DEFAULT_MAX_ACC_LIN)
        self.declare_parameter("max_acc_angular",  DEFAULT_MAX_ACC_ANG)

        def p(name):
            return self.get_parameter(name).value

        action_topic  = p("action_topic")
        cmd_vel_topic = p("cmd_vel_topic")
        odom_topic    = p("odom_topic")
        status_topic  = p("status_topic")

        self.lin = p("linear_x")
        self.ang = p("angular_z")
        self.forward_step = p("forward_step_m")            # m
        self.turn_step    = math.radians(p("turn_step_deg"))  # rad

        self.max_acc_lin = p("max_acc_linear")
        self.max_acc_ang = p("max_acc_angular")
        publish_rate     = p("publish_rate_sec")

        # ------------------------------------------------------------------
        # Stato interno
        # ------------------------------------------------------------------
        self._target_lin  = 0.0
        self._target_ang  = 0.0
        self._current_lin = 0.0
        self._current_ang = 0.0
        self._dt = publish_rate

        self._executing   = False
        self._start_pose  = None
        self._prim_kind   = None     # "forward" | "turn"
        self._prim_target = 0.0      # metri (forward) o radianti (turn)
        self._odom        = None
        self._deadline    = None

        # ------------------------------------------------------------------
        # Subscriber / Publisher / Timer
        # ------------------------------------------------------------------
        self.sub_action = self.create_subscription(String, action_topic, self._action_cb, 10)
        self.sub_odom = self.create_subscription(
            Odometry, odom_topic, self._odom_cb,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                       history=HistoryPolicy.KEEP_LAST,
                       depth=1))

        qos_cmd_vel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_cmd_vel = self.create_publisher(Twist, cmd_vel_topic, qos_cmd_vel)
        self.pub_status  = self.create_publisher(String, status_topic, 10)

        self._publish_timer = self.create_timer(publish_rate, self._publish_cb)

        self.get_logger().info(
            f"action_node (UniNaVid) avviato\n"
            f"  in        : {action_topic}\n"
            f"  out       : {cmd_vel_topic}\n"
            f"  odom      : {odom_topic}\n"
            f"  step      : forward={self.forward_step} m  turn={math.degrees(self.turn_step):.0f}\n"
            f"  vel       : lin={self.lin} m/s  ang={self.ang} rad/s"
        )

    # ------------------------------------------------------------------
    # Action callback
    # ------------------------------------------------------------------
    def _action_cb(self, msg: String):
        action = msg.data.strip().lower()
        if not action:
            return

        if action == "stop":
            self._target_lin = self._target_ang = 0.0
            self._executing = False
            self._deadline = None
            self._publish_status("done")
        elif action == "forward":
            self._start_primitive(kind="forward", magnitude=self.forward_step)   # m
        elif action == "left":
            self._start_primitive(kind="turn", magnitude=+self.turn_step)        # rad, +ve = CCW
        elif action == "right":
            self._start_primitive(kind="turn", magnitude=-self.turn_step)        # rad, -ve = CW
        else:
            self.get_logger().warn(f"Azione non gestita: '{msg.data}' -> done")
            self._publish_status("done")

    def _odom_cb(self, msg: Odometry):
        self._odom = msg

    def _publish_status(self, status: str):
        """status ∈ {'done', 'aborted'}"""
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

    def _start_primitive(self, kind: str, magnitude: float):
        if self._odom is None:
            self.get_logger().warn("Nessun odom ancora -> done immediato")
            self._publish_status("done")
            return
        x0, y0, yaw0 = self._pose_xy_yaw(self._odom)
        self._start_pose = (x0, y0, yaw0)
        self._prim_kind = kind
        self._prim_target = max(abs(magnitude) - (0.01 if kind == "forward" else math.radians(1.0)), 0.0)
        if self._prim_target <= 0.0:
            self._publish_status("done")
            return

        if kind == "forward":
            self._target_lin, self._target_ang = self.lin, 0.0
        else:
            self._target_lin = 0.0
            self._target_ang = math.copysign(self.ang, magnitude)

        vel = self.lin if kind == "forward" else self.ang
        self._deadline = self.get_clock().now() + rclpy.duration.Duration(
            seconds=(self._prim_target / vel) * 3.0 + 1.0)
        self._executing = True   # ultimo, dopo _deadline

    # ------------------------------------------------------------------
    # Publish callback (closed-loop + smoothing + pubblicazione)
    # ------------------------------------------------------------------
    def _publish_cb(self):
        if self._executing and self._odom is not None and self._deadline is not None:
            x, y, yaw = self._pose_xy_yaw(self._odom)
            x0, y0, yaw0 = self._start_pose
            if self._prim_kind == "forward":
                progress = math.hypot(x - x0, y - y0)
            else:
                dyaw = yaw - yaw0
                dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # normalizza
                progress = abs(dyaw)

            if self.get_clock().now() >= self._deadline:
                self.get_logger().warn("Primitiva in timeout -> stop forzato + aborted")
                self._target_lin = self._target_ang = 0.0
                self._executing = False
                self._deadline = None
                self._publish_status("aborted")
            elif progress >= self._prim_target:
                self._target_lin = self._target_ang = 0.0
                self._executing = False
                self._deadline = None
                self._publish_status("done")

        self._current_lin = self._ramp(self._current_lin, self._target_lin, self.max_acc_lin * self._dt)
        self._current_ang = self._ramp(self._current_ang, self._target_ang, self.max_acc_ang * self._dt)
        twist = Twist()
        twist.linear.x, twist.angular.z = self._current_lin, self._current_ang
        self.pub_cmd_vel.publish(twist)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _ramp(current: float, target: float, max_delta: float) -> float:
        """Avvicina current a target di al massimo max_delta."""
        delta = target - current
        delta = math.copysign(min(abs(delta), max_delta), delta)
        return current + delta

    @staticmethod
    def _pose_xy_yaw(odom):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return p.x, p.y, yaw


# =============================================================================
# Entry point
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutdown: invio STOP finale")
        node.pub_cmd_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()