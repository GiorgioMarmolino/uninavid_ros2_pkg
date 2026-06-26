#!/usr/bin/env python3
import os
import urllib.request

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from vla_base.vla_base_node import VLABaseNode
from third_party.uni_navid_agent import UniNaVid_Agent
from huggingface_hub import snapshot_download

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
            os.path.join(os.environ["UNINAVID_MODEL_PATH"], "uni_navid_model", "uninavid-7b-full-224-video-fps-1-grid-2"),
        )        
        self.declare_parameter("task", "vln")
        self.declare_parameter("answer_topic", "/uninavid/answer")
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
        result = self._agent.act({"instruction": instruction, "observations": frame})
        return result["actions"]

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
        if not (os.path.isdir(model_path) and os.listdir(model_path)):
            os.makedirs(model_path, exist_ok=True)
            snapshot_download(
                repo_id=repo_id, 
                local_dir=model_path, 
                local_dir_use_symlinks=False,
                allow_patterns=["*.json", "*.bin", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
            )

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