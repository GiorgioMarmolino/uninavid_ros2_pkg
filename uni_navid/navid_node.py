#!/usr/bin/env python3
import os
import urllib.request

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from vla_base.vla_base_node import VLABaseNode
from third_party.uni_navid_agent import UniNaVid_Agent
from huggingface_hub import snapshot_download

import cv2
import time

UNINAVID_REPO_ID = "Jzzhang/Uni-NaVid"
EVA_VIT_G_URL = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"

# Task -> how the raw goal is turned into the model instruction.
# Uni-NaVid distinguishes tasks ONLY by the instruction text (paper, Sec. V):
#   vln/eqa   -> raw instruction
#   objectnav -> "Search for <goal>."
#   following -> "Follow <goal>."
VALID_TASKS = ("vln", "objectnav", "eqa", "following")

class UniNaVidNode(VLABaseNode):
    def __init__(self):
        super().__init__("uninavid_node", "uninavid")
        
        answer_topic = self.get_parameter("answer_topic").value
        self.pub_answer = self.create_publisher(String, answer_topic, 10)
        self.get_logger().info(f"Task: {self._task}  (answer -> {answer_topic})")

    def _declare_params(self):
        self.declare_parameter(
            "model_path",
            os.path.join(os.environ["UNINAVID_MODEL_PATH"], "uni_navid_model"),
        )        
        self.declare_parameter("task", "vln")
        self.declare_parameter("answer_topic", "/uninavid/answer")
        self.declare_parameter("save_debug_frames", True)
        self.declare_parameter("debug_dir", "/tmp/uninavid_debug")
        self.declare_parameter("frame_rgb", False)

        self._save_debug = self.get_parameter("save_debug_frames").value
        self._frame_is_rgb = self.get_parameter("frame_rgb").value
        self._debug_dir = self.get_parameter("debug_dir").value
        self._debug_run = None
        self._debug_idx = 0
        self._debug_last_instr = None

        self.model_path = self.get_parameter("model_path").value
        self._task = self.get_parameter("task").value.strip().lower()
        if self._task not in VALID_TASKS:
            self.get_logger().warn(f"Unknown task '{self._task}', falling back to 'vln'")
            self._task = "vln"

    def load_model(self):
        model_path = self.ensure_model(self.model_path)
        os.chdir(os.path.join(os.environ["UNINAVID_REPO_DIR"], "UniNaVid"))
        self.get_logger().info(f"cwd = {os.getcwd()}")
        return UniNaVid_Agent(model_path)

    # ---- inference ----
    def infer_action(self, frame, goal):
        # frames are already RGB upstream -> no BGR conversion
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

    # ---- stop handling (task-dependent) ----
    def _on_stop(self):
        if self._task == "eqa":
            question = self._goal
            frame = self._decode_latest()
            if frame is not None:
                answer = self._agent.answer(question, frame)
                self._publish_answer(answer)
                self.get_logger().info(f"EQA answer: {answer}")
            self._goal = None
        elif self._task == "following":
            # target reached / idle: keep the goal, the person may move again
            self.get_logger().info("Following: caught up / target idle - keep tracking")
        else:  # vln, objectnav
            self._goal = None
            self.get_logger().info("STOP reached - goal complete")

    def _publish_answer(self, text: str):
        out = String()
        out.data = text
        self.pub_answer.publish(out)

    # ---- weights / encoder setup ----
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

    def _save_debug_frame(self, frame, instruction, actions):
        if instruction != self._debug_last_instr:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._debug_run = os.path.join(self._debug_dir, f"run_{ts}")
            os.makedirs(self._debug_run, exist_ok=True)
            self._debug_idx = 0
            self._debug_last_instr = instruction

        img = frame.copy()
        if self._frame_is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # imwrite salva BGR

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