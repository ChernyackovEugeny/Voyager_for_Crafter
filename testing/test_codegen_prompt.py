import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prompts.codegen_prompt import format_user_prompt


class CodegenPromptTests(unittest.TestCase):
    def test_default_omits_previous_attempt_block(self):
        prompt = format_user_prompt(
            state_text="Obs: tree\nInventory: empty\nStatus: ok",
            task="collect wood",
            retrieved_skills=[],
        )
        self.assertNotIn("Previous Attempt", prompt)

    def test_previous_failure_block_is_appended(self):
        prompt = format_user_prompt(
            state_text="Obs: tree",
            task="collect wood",
            retrieved_skills=[],
            previous_failure=(
                "def collect_wood(state):\n    yield 0\n",
                "health_low",
            ),
        )
        self.assertIn("Previous Attempt For This Task Failed", prompt)
        self.assertIn("health_low", prompt)
        self.assertIn("def collect_wood(state)", prompt)
        self.assertIn("different approach", prompt)


if __name__ == "__main__":
    unittest.main()
