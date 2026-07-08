#!/usr/bin/env python3
import os
import urllib.request
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String, Empty
from sensor_msgs.msg import CompressedImage

from third_party.uni_navid_agent import UniNaVid_Agent
from huggingface_hub import snapshot_download

import cv2
import time
from collections import deque
import numpy as np


UNINAVID_REPO_ID = "Jzzhang/Uni-NaVid"
EVA_VIT_G_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"

# Task -> how the raw goal is turned into the model instruction.
# Uni-NaVid distinguishes tasks ONLY by the instruction text (paper, Sec. V):
#   vln/eqa   -> raw instruction
#   objectnav -> "Search for <goal>."
#   following -> "Follow <goal>."
VALID_TASKS = ("vln", "objectnav", "eqa", "following")


class UniNaVidNode(Node):
    def __init__(self):
        super().__init__("uninavid_node")

        #------------------------------------------------------------------------
        # ============================ Paramters ================================
        #------------------------------------------------------------------------
        # inference topics
        self.declare_parameter("image_topic",       "/sensors/front_camera/color/image_raw/compressed")
        self.declare_parameter("goal_topic",        "/goal_instruction")

        # uninavid topics
        self.declare_parameter("action_topic",      "/uninavid/action")
        self.declare_parameter("reset_topic",       "/uninavid/reset")
        self.declare_parameter("status_topic",      "/uninavid/primitive_status")
        self.declare_parameter("answer_topic",      "/uninavid/answer")

        self.declare_parameter("model_path", os.path.join(os.environ["UNINAVID_MODEL_PATH"], "uni_navid_model"))        
        self.declare_parameter("task",              "vln")
        
        # debug 
        self.declare_parameter("save_debug_frames", True)
        self.declare_parameter("debug_dir",         "/tmp/uninavid_debug")
        self.declare_parameter("frame_rgb",         False)      # considering frame from compressed to decoded

        #------------------------------------------------------------------------
        p = lambda n: self.get_parameter(n).value

        self._frame_is_rgb =        p("frame_rgb")
        self._debug_dir =           p("debug_dir")
        image_topic =               p("image_topic")
        goal_topic =                p("goal_topic")
        action_topic =              p("action_topic")
        reset_topic =               p("reset_topic")
        status_topic =              p("status_topic")
        answer_topic =              p("answer_topic")
        self.model_path =           p("model_path")
        self._task =                p("task").strip().lower()
        self._save_debug =          p("save_debug_frames")

        if self._task not in VALID_TASKS:
            self.get_logger().warn(f"Unknown task '{self._task}', falling back to 'vln'")
            self._task = "vln"
        #------------------------------------------------------------------------

        # ---- state ----
        self._debug_run = None
        self._debug_idx = 0
        self._debug_last_instr = None
        self._agent = None
        self._model_ready = False
        self._lock = threading.Lock()
        self._last_image_msg = None
        self._goal = None
        self._queue = deque()
        self._busy = False

        # ---- i/o ----
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        #------------------------------------------------------------------------
        # ============================ Sub/Pub ==================================
        #------------------------------------------------------------------------
        # Subsribers
        self.sub_image = self.create_subscription(CompressedImage,  image_topic,    self._image_cb,     qos_sensor)
        self.sub_goal = self.create_subscription(String,            goal_topic,     self._goal_cb,      10)
        self.sub_reset = self.create_subscription(Empty,            reset_topic,    self._reset_cb,     10)
        self.sub_status = self.create_subscription(String,          status_topic,   self._status_cb,    10)
        # Publishers
        self.pub_action = self.create_publisher(String, action_topic, 10)
        self.pub_answer = self.create_publisher(String, answer_topic, 10)

        self.get_logger().info(f"Task: {self._task}  (answer -> {answer_topic})")

        threading.Thread(target=self._load_model_thread, daemon=True).start()

    #------------------------------------------------------------------------
    # ============================ Callbacks ================================
    #------------------------------------------------------------------------
    def _image_cb(self, msg: CompressedImage):
        with self._lock:
            self._last_image_msg = msg

    def _goal_cb(self, msg: String):
        goal = msg.data.strip()
        if not goal:
            return
        self._goal = goal
        self._busy = False
        self.get_logger().info(f"New goal: {goal}")
        self._step()                      # no memory reset


    def _reset_cb(self, msg: Empty):
        self._goal = None
        self._reset()
        self.get_logger().info("Reset")

    def _status_cb(self, msg: String):
        if msg.data.strip().lower() in ("done", "aborted", "idle", "ready"):
            self._busy = False
            self._step()            
    #------------------------------------------------------------------------endcallbacks

    #------------------------------------------------------------------------
    # ============================ Model Loading ============================
    #------------------------------------------------------------------------
    def _load_model_thread(self):
        try:
            model_path = self.ensure_model(self.model_path)
            os.chdir(os.path.join(os.environ["UNINAVID_REPO_DIR"], "UniNaVid"))
            self.get_logger().info(f"cwd = {os.getcwd()}")
            agent = UniNaVid_Agent(model_path)
            with self._lock:
                self._agent = agent
                self._model_ready = True
            self.get_logger().info(f"{self.get_name()} model loaded")
        except Exception as e:
            self.get_logger().error(f"Error while loading model: {e}")
            return

        # preflight 1: aspetta il primo frame camera
        while rclpy.ok():
            with self._lock:
                have_img = self._last_image_msg is not None
            if have_img:
                break
            self.get_logger().warn("Waiting for camera frame...", throttle_duration_sec=2.0)
            time.sleep(0.2)
        self.get_logger().info("Camera frame OK")

        # preflight 2: aspetta il goal
        while rclpy.ok():
            if self._goal is not None:
                break
            self.get_logger().warn("Waiting for goal instruction...", throttle_duration_sec=2.0)
            time.sleep(0.2)
        self.get_logger().info(f"Goal received: {self._goal}")
        self._step()
    #--------------------------------------------------endloading

    def infer_action(self, frame, goal):
        instruction = self._format_instruction(goal)
        model_frame = frame
        if not self._frame_is_rgb:
            model_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._agent.act({"instruction": instruction, "observations": model_frame})
        actions = result["actions"]
        self.get_logger().info(f"Next actions: {actions}")
        if self._save_debug:
            self._save_debug_frame(frame, instruction, actions)
        return actions

    def _format_instruction(self, goal: str) -> str:
        if self._task == "objectnav":
            return f"Search for {goal}."
        if self._task == "following":
            return f"Follow {goal}."
        return goal  # vln, eqa: raw instruction / question


    #------------------------------------------------------------------------
    # ============================ Step-synchronous loop ====================
    #------------------------------------------------------------------------
    def _step(self):
        if self._busy or self._goal is None or not self._model_ready:
            return
        frame = self._decode_latest()
        if frame is None:
            return
        actions = self.infer_action(frame, self._goal)
        if not actions:
            return
        action = actions[0]
        self._busy = True
        self.__publish(action)
        if action == "stop":
            self._on_stop()
    
    # ---- stop handling (task-dependent) ----
    def _on_stop(self):
        if self._task == "eqa":
            question = self._goal
            frame = self._decode_latest()
            if frame is not None:
                answer = self._agent.answer(question, frame)
                self._publish(answer)
                self.get_logger().info(f"EQA answer: {answer}")
            self._goal = None
        elif self._task == "following":
            # target reached / idle: keep the goal, the person may move again
            self.get_logger().info("Following: caught up / target idle - keep tracking")
        else:  # vln, objectnav
            self._goal = None
            self.get_logger().info("STOP reached - goal complete")
    #------------------------------------------------------------------------loopsyncstep


    #------------------------------------------------------------------------
    # ============================ weights / encoder setup ==================
    #------------------------------------------------------------------------
    @staticmethod
    def ensure_model(model_path: str, repo_id: str = UNINAVID_REPO_ID) -> str:
        # model_path = download folder (es. /models/uni_navid_model)
        if not (os.path.isdir(model_path) and os.listdir(model_path)):
            os.makedirs(model_path, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
            )
        if os.path.isfile(os.path.join(model_path, "config.json")):
            ckpt = model_path
        else:
            subs = [os.path.join(model_path, d) for d in sorted(os.listdir(model_path))
                    if os.path.isfile(os.path.join(model_path, d, "config.json"))]
            if not subs:
                raise FileNotFoundError(f"No config.json found in {model_path} or its subfolders")
            ckpt = subs[0]

        # ---- EVA encoder
        models_root = os.environ["UNINAVID_MODEL_PATH"]
        eva_dst = os.path.join(models_root, "eva_vit_g.pth")
        if not os.path.isfile(eva_dst):
            urllib.request.urlretrieve(EVA_VIT_G_URL, eva_dst)

        link = os.path.join(os.environ["UNINAVID_REPO_DIR"], "UniNaVid", "model_zoo", "eva_vit_g.pth")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.islink(link):
            if os.path.exists(link):
                os.remove(link)
            os.symlink(eva_dst, link)

        return ckpt
    #------------------------------------------------------------------------endencodersetup


    #------------------------------------------------------------------------
    # ============================ Helpers ==================================
    #------------------------------------------------------------------------
    def _publish(self, publisher, text: str):
        out = String()
        out.data = text
        publisher.publish(out)

    def _reset(self):
        with self._lock:
            if self._agent is not None:
                self._agent.reset()
        self._queue.clear()
        self._busy = False

    def _decode_latest(self):
        "Decode frames from CompressedImage to Image; image is given as BGR"
        with self._lock:
            msg = self._last_image_msg
        if msg is None:
            return None
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    
    # @staticmethod
    # def make_sensor_qos(reliability: str, depth: int = 1) -> QoSProfile:
    #     "This can be used in case code need to be tested both in simulation (Gazebo/Rviz2) and reality to set reliability as RELIABLE or BEST_EFFORT"
    #     rel = (ReliabilityPolicy.RELIABLE if reliability.strip().lower() == "reliable"
    #         else ReliabilityPolicy.BEST_EFFORT)
    #     return QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=depth)
    #------------------------------------------------------------------------endhelpers




    #------------------------------------------------------------------------
    # ============================ DEBUG ====================================
    #------------------------------------------------------------------------
    def _save_debug_frame(self, frame, instruction, actions):
        if instruction != self._debug_last_instr:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._debug_run = os.path.join(self._debug_dir, f"run_{ts}")
            os.makedirs(self._debug_run, exist_ok=True)
            self._debug_idx = 0
            self._debug_last_instr = instruction

        img = frame.copy()
        if self._frame_is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # imwrite saves BGR

        executed = actions[0] if actions else "-"
        chunk = " ".join(actions)
        h, w = img.shape[:2]

        cv2.rectangle(img, (0, 0), (w, 72), (0, 0, 0), -1)
        cv2.putText(img, f"step {self._debug_idx:04d}   EXEC: {executed.upper()}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"chunk: {chunk}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, instruction[:70],
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # freccetta direzione dell'azione eseguita
        cx, cy = w - 60, 40
        if executed == "forward":
            cv2.arrowedLine(img, (cx, cy + 12), (cx, cy - 12), (0, 255, 0), 3, tipLength=0.4)
        elif executed in ("left", "turn_left"):
            cv2.arrowedLine(img, (cx + 12, cy), (cx - 12, cy), (0, 255, 0), 3, tipLength=0.4)
        elif executed in ("right", "turn_right"):
            cv2.arrowedLine(img, (cx - 12, cy), (cx + 12, cy), (0, 255, 0), 3, tipLength=0.4)
        elif executed == "stop":
            cv2.circle(img, (cx, cy), 12, (0, 0, 255), -1)

        cv2.imwrite(os.path.join(self._debug_run, f"frame_{self._debug_idx:05d}.png"), img)
        self._debug_idx += 1
    #------------------------------------------------------------------------enddebug








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