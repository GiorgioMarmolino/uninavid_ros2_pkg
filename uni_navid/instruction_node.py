import json
import urllib.error
import urllib.request
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve            # avvia il server (in background o in un altro terminale)
# ollama pull qwen2.5:7b  # o llama3.1:8b — scaricato una volta, poi offline

# ros2 run navila_ros2_bridge instruction_node
# # se Ollama gira su un altro host/porta:
# ros2 run navila_ros2_bridge instruction_node --ros-args -p ollama_url:=http://192.168.x.x:11434
# # per bypassare l'LLM:
# ros2 run navila_ros2_bridge instruction_node --ros-args -p use_llm:=false
# # System prompt tuned for VLN-style VLA models (NaVILA / Uni-NaVid),
# trained on R2R/RxR-style data: English, imperative, egocentric phrasing.
REWRITE_SYSTEM_PROMPT = """You rewrite raw navigation goals into clean instructions \
for a Vision-Language-Action navigation model deployed on a mobile ground robot.

Rules:
- Output ONLY the rewritten instruction. No preamble, no quotes, no explanation.
- Always output English (the VLA is trained on English navigation data).
- Use simple, imperative, sequential phrasing (e.g. "Turn left, then go straight \
down the hallway and stop at the door").
- Convert absolute or ambiguous spatial references (north/south, coordinates, \
"over there") into egocentric ones relative to the robot's current heading \
(left/right/forward/behind).
- Preserve the original intent and any explicit landmarks. Do NOT invent landmarks, \
rooms, or objects that are not implied by the input.
- Fix obvious typos. Keep it concise (one or two sentences)."""


class GoalInstructionPublisher(Node):
    """Reads goals from stdin, optionally refines them with a local Ollama LLM,
    and publishes the result as a std_msgs/String on /goal_instruction."""

    def __init__(self):
        super().__init__('goal_instruction_publisher')

        self.declare_parameter('use_llm', True)
        self.declare_parameter('model', 'gemma3:12b')
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('timeout', 15.0)
        self.declare_parameter('require_confirmation', False)

        self.use_llm = self.get_parameter('use_llm').value
        self.model = self.get_parameter('model').value
        self.ollama_url = self.get_parameter('ollama_url').value.rstrip('/')
        self.timeout = self.get_parameter('timeout').value
        self.require_confirmation = self.get_parameter('require_confirmation').value

        self.publisher_ = self.create_publisher(String, '/goal_instruction', 10)

        if self.use_llm:
            self._check_ollama()

        self.get_logger().info(
            f"Node started (use_llm={self.use_llm}, model={self.model}, "
            f"ollama={self.ollama_url}). Type the goal and press ENTER."
        )

        self.input_thread = threading.Thread(target=self.read_input_loop, daemon=True)
        self.input_thread.start()

    def _check_ollama(self):
        """Warn early if the Ollama server is unreachable (fall back to raw)."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                tags = json.loads(resp.read())
            names = [m.get('name', '') for m in tags.get('models', [])]
            if self.model not in names:
                self.get_logger().warn(
                    f"Model '{self.model}' not pulled. Run: ollama pull {self.model}. "
                    f"Available: {names}"
                )
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"Ollama unreachable at {self.ollama_url} ({e}); publishing raw goals."
            )

    def refine(self, raw: str) -> str:
        """Rewrite the raw goal via Ollama. Fall back to raw on any failure."""
        if not self.use_llm:
            return raw

        payload = json.dumps({
            "model": self.model,
            "system": REWRITE_SYSTEM_PROMPT,
            "prompt": raw,
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            refined = data.get("response", "").strip()
            return refined or raw
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"LLM refine failed ({e}); using raw goal.")
            return raw

    def read_input_loop(self):
        while rclpy.ok():
            try:
                sys.stdout.write("Goal > ")
                sys.stdout.flush()
                text = sys.stdin.readline()
                if text is None or text == '':
                    break
                text = text.strip()
                if text == "":
                    continue
                self.get_logger().info("REFINING PROCESS")
                refined = self.refine(text)

                if refined != text:
                    self.get_logger().info(f"Raw:     '{text}'")
                    self.get_logger().info(f"Refined: '{refined}'")

                goal = refined
                if self.require_confirmation and refined != text:
                    sys.stdout.write("Publish refined? [Y/n/e(dit)] ")
                    sys.stdout.flush()
                    ans = sys.stdin.readline().strip().lower()
                    if ans == 'n':
                        goal = text
                    elif ans == 'e':
                        sys.stdout.write("Edited goal > ")
                        sys.stdout.flush()
                        edited = sys.stdin.readline().strip()
                        goal = edited or refined

                msg = String()
                msg.data = goal
                self.publisher_.publish(msg)
                self.get_logger().info(f"Published goal: '{goal}'")

            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"Input error: {str(e)}")
                break


def main(args=None):
    rclpy.init(args=args)
    node = GoalInstructionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()