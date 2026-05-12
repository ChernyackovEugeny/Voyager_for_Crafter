import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.memory import SpatialMemory
from skills.runner import SkillLoadError, SkillRuntime, load_skill


class TestSkillLoader(unittest.TestCase):
    def test_loads_simple_generator(self):
        src = (
            "def hello(state):\n"
            "    yield 0\n"
            "    yield 1\n"
        )
        name, func = load_skill(src)
        self.assertEqual(name, "hello")
        gen = func({})
        self.assertEqual(next(gen), 0)
        self.assertEqual(next(gen), 1)

    def test_syntax_error_raises(self):
        with self.assertRaises(SkillLoadError):
            load_skill("def broken(state)\n    yield 0\n")

    def test_import_rejected(self):
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill("import os\ndef f(state):\n    yield 0\n")
        self.assertIn("import", str(ctx.exception))

    def test_from_import_rejected(self):
        with self.assertRaises(SkillLoadError):
            load_skill("from os import system\ndef f(state):\n    yield 0\n")

    def test_dunder_access_rejected(self):
        with self.assertRaises(SkillLoadError):
            load_skill("def f(state):\n    yield state.__class__\n")

    def test_forbidden_builtin_call_rejected(self):
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill("def f(state):\n    yield eval('1 + 1')\n")
        self.assertIn("eval", str(ctx.exception))

    def test_no_function_rejected(self):
        with self.assertRaises(SkillLoadError):
            load_skill("x = 1\n")

    def test_top_level_assignment_rejected(self):
        with self.assertRaises(SkillLoadError):
            load_skill("x = 1\ndef f(state):\n    yield 0\n")

    def test_top_level_class_rejected(self):
        with self.assertRaises(SkillLoadError):
            load_skill("class Helper:\n    pass\ndef f(state):\n    yield 0\n")

    def test_multiple_functions_rejected(self):
        src = (
            "def a(state):\n    yield 0\n"
            "def b(state):\n    yield 1\n"
        )
        with self.assertRaises(SkillLoadError):
            load_skill(src)

    def test_skill_can_call_primitive(self):
        src = (
            "def use_primitive(state):\n"
            "    yield move_left()\n"
        )
        _, func = load_skill(src)
        action = next(func({}))
        self.assertIsInstance(action, int)

    def test_yield_from_action_primitive_rejected(self):
        src = (
            "def bad_move(state):\n"
            "    state = yield from move_right()\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src)
        self.assertIn("yield from move_right", str(ctx.exception))

    def test_yield_from_go_to_allowed(self):
        src = (
            "def navigate(state):\n"
            "    state = yield from go_to((1, 1), state)\n"
        )
        name, func = load_skill(src)
        self.assertEqual(name, "navigate")
        self.assertTrue(callable(func))

    def test_yield_from_explore_for_allowed(self):
        src = (
            "def explore(state):\n"
            "    coords, state = yield from explore_for('water', state)\n"
        )
        name, func = load_skill(src)
        self.assertEqual(name, "explore")
        self.assertTrue(callable(func))

    def test_memory_primitives_require_runtime(self):
        src = (
            "def remember(state):\n"
            "    save_in_memory('water', (1, 2))\n"
            "    yield 0\n"
        )
        with self.assertRaises(SkillLoadError):
            load_skill(src)

    def test_skill_can_use_runtime_memory(self):
        memory = SpatialMemory()
        src = (
            "def remember(state):\n"
            "    save_in_memory('water', (1, 2))\n"
            "    set_home((3, 4))\n"
            "    yield 0\n"
        )
        _, func = load_skill(src, SkillRuntime(memory=memory))
        self.assertEqual(next(func({})), 0)
        self.assertEqual(memory.get_memory(), {"water": (1, 2)})
        self.assertEqual(memory.get_home(), (3, 4))

    def test_runtime_memory_is_isolated(self):
        first = SpatialMemory()
        second = SpatialMemory()
        src = (
            "def remember(state):\n"
            "    save_in_memory('water', (1, 2))\n"
            "    yield 0\n"
        )
        _, func_a = load_skill(src, SkillRuntime(memory=first))
        _, func_b = load_skill(src, SkillRuntime(memory=second))
        next(func_a({}))
        second.memory_add("stone", (5, 6))
        next(func_b({}))

        self.assertEqual(first.get_memory(), {"water": (1, 2)})
        self.assertEqual(
            second.get_memory(),
            {"stone": (5, 6), "water": (1, 2)},
        )


if __name__ == "__main__":
    unittest.main()
