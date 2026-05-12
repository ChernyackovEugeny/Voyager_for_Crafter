import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skills.description import compose_description, extract_primitives_called


class SkillDescriptionTests(unittest.TestCase):
    def test_extract_primitives_from_nested_generator_code(self):
        code = """
def collect_wood(state):
    coords = find_nearest("tree", state)
    if coords is not None:
        state = yield from go_to(coords, state)
    state = yield do_action()
"""

        self.assertEqual(
            extract_primitives_called(code),
            ["do_action", "find_nearest", "go_to"],
        )

    def test_extract_primitives_filters_non_primitive_calls(self):
        code = """
def f(state):
    inv = state["info"]["inventory"]
    if inv.get("wood", 0) >= len([1, 2]):
        state = yield place("table")
    for _ in range(3):
        state = yield do_action()
"""

        self.assertEqual(extract_primitives_called(code), ["do_action", "place"])

    def test_extract_primitives_syntax_error_returns_empty(self):
        self.assertEqual(extract_primitives_called("def f(:"), [])

    def test_compose_description_adds_primitives(self):
        code = "def f(state):\n    state = yield do_action()\n"

        desc = compose_description("noop", "do something", code)

        self.assertEqual(desc, "do something. Uses: do_action")

    def test_compose_description_without_primitives_returns_task(self):
        code = "def f(state):\n    state = yield 0\n"

        desc = compose_description("noop", "do something", code)

        self.assertEqual(desc, "do something")


if __name__ == "__main__":
    unittest.main()
