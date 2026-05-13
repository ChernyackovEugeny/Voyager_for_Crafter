import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.bootstrap import BOOTSTRAP_SKILLS
from prompts.reflection_prompt import SYSTEM_PROMPT as REFLECTION_SYSTEM_PROMPT
from prompts.codegen_prompt import SYSTEM_PROMPT, format_user_prompt


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

    def test_system_prompt_emphasizes_defensive_placement(self):
        self.assertIn("Defensive placement is a core survival ability", SYSTEM_PROMPT)
        self.assertIn('can_place_ahead("stone", state)', SYSTEM_PROMPT)
        self.assertIn('place("stone")', SYSTEM_PROMPT)
        self.assertIn("survive_by_blocking_monsters", SYSTEM_PROMPT)
        self.assertIn("single zombie", SYSTEM_PROMPT)
        self.assertIn("has a sword", SYSTEM_PROMPT)

    def test_bootstrap_survive_mentions_blocking_monsters(self):
        descriptions = {skill.name: skill.description for skill in BOOTSTRAP_SKILLS}
        self.assertIn("place('stone')", descriptions["survive"])
        self.assertIn("temporary obstacle", descriptions["survive"])
        self.assertIn("defensive placement", descriptions["build_shelter"])

    def test_reflection_prompt_mentions_over_fleeing(self):
        self.assertIn("over-fleeing", REFLECTION_SYSTEM_PROMPT)
        self.assertIn("fight an", REFLECTION_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
