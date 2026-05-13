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

    def test_explore_for_rejected_as_unknown_helper(self):
        src = (
            "def explore(state):\n"
            "    coords, state = yield from explore_for('water', state)\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src)
        self.assertIn("explore_for", str(ctx.exception))

    def test_unknown_helper_rejected(self):
        src = (
            "def collect_wood(state):\n"
            "    state = yield from chop_tree(state)\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src)
        self.assertIn("chop_tree", str(ctx.exception))

    def test_primitive_wrong_arity_rejected(self):
        memory = SpatialMemory()
        src = (
            "def remember(state):\n"
            "    coords = get_memory('water')\n"
            "    yield 0\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src, SkillRuntime(memory=memory))
        self.assertIn("get_memory", str(ctx.exception))

    def test_nested_helper_function_rejected(self):
        src = (
            "def collect_drink(state):\n"
            "    def find_water(state):\n"
            "        return find_nearest('water', state)\n"
            "    coords = find_water(state)\n"
            "    yield 0\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src)
        self.assertIn("nested helper", str(ctx.exception))

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


    def test_extra_skill_becomes_callable_in_main(self):
        helper = (
            "def chop_tree(state):\n"
            "    state = yield 5\n"
            "    return state\n"
        )
        main = (
            "def collect_wood(state):\n"
            "    state = yield from chop_tree(state)\n"
        )
        _, func = load_skill(main, extra_skills=[("chop_tree", helper)])
        gen = func({})
        self.assertEqual(next(gen), 5)

    def test_extra_skill_bare_return_preserves_latest_state_for_yield_from(self):
        helper = (
            "def scout_area(state):\n"
            "    state = yield 5\n"
            "    return\n"
        )
        main = (
            "def collect_drink(state):\n"
            "    state = yield from scout_safely(state)\n"
            "    state['kept'] = True\n"
            "    return state\n"
        )
        _, func = load_skill(main, extra_skills=[("scout_safely", helper)])
        initial = {"kept": False}
        updated = {"kept": False}
        gen = func(initial)
        self.assertEqual(next(gen), 5)
        with self.assertRaises(StopIteration) as ctx:
            gen.send(updated)
        self.assertIs(ctx.exception.value, updated)
        self.assertTrue(updated["kept"])

    def test_extra_skill_is_callable_by_repository_alias(self):
        helper = (
            "def scout_area(state):\n"
            "    state = yield 5\n"
            "    return state\n"
        )
        main = (
            "def collect_drink(state):\n"
            "    state = yield from scout_safely(state)\n"
        )
        _, func = load_skill(main, extra_skills=[("scout_safely", helper)])
        gen = func({})
        self.assertEqual(next(gen), 5)

    def test_invalid_extra_skill_does_not_block_main(self):
        bad = "def broken(:\n    pass\n"
        main = (
            "def collect_wood(state):\n"
            "    state = yield 0\n"
        )
        _, func = load_skill(main, extra_skills=[("broken", bad)])
        gen = func({})
        self.assertEqual(next(gen), 0)

    def test_invalid_referenced_extra_skill_blocks_main(self):
        bad = "def scout_safely(:\n    pass\n"
        main = (
            "def collect_drink(state):\n"
            "    state = yield from scout_safely(state)\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(main, extra_skills=[("scout_safely", bad)])
        self.assertIn("required extra skill failed", str(ctx.exception))

    def test_function_must_accept_state_argument(self):
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill("def bad(current):\n    yield 0\n")
        self.assertIn("argument named 'state'", str(ctx.exception))

    def test_function_must_be_generator(self):
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill("def bad(state):\n    return 0\n")
        self.assertIn("generator", str(ctx.exception))

    def test_optional_find_nearest_must_be_guarded_before_go_to(self):
        src = (
            "def bad_water(state):\n"
            "    coords = find_nearest('water', state)\n"
            "    state = yield from go_to(coords, state)\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src)
        self.assertIn("optional value 'coords'", str(ctx.exception))

    def test_optional_get_home_guard_allows_go_to(self):
        src = (
            "def go_home(state):\n"
            "    home = get_home()\n"
            "    if home is not None:\n"
            "        state = yield from go_to(home, state)\n"
            "    yield noop()\n"
        )
        _, func = load_skill(src, SkillRuntime(memory=SpatialMemory()))
        self.assertTrue(callable(func))

    def test_optional_get_memory_value_must_be_guarded_before_indexing(self):
        src = (
            "def bad_memory(state):\n"
            "    coords = get_memory().get('water')\n"
            "    x = coords[0]\n"
            "    yield noop()\n"
        )
        with self.assertRaises(SkillLoadError) as ctx:
            load_skill(src, SkillRuntime(memory=SpatialMemory()))
        self.assertIn("optional value 'coords'", str(ctx.exception))

    def test_none_guard_else_branch_allows_optional_coord(self):
        src = (
            "def guarded_water(state):\n"
            "    coords = get_memory().get('water')\n"
            "    if coords is None:\n"
            "        yield noop()\n"
            "    else:\n"
            "        state = yield from go_to(coords, state)\n"
        )
        _, func = load_skill(src, SkillRuntime(memory=SpatialMemory()))
        self.assertTrue(callable(func))

    def test_none_guard_in_and_condition_allows_optional_coord(self):
        src = (
            "def guarded_grass(state):\n"
            "    current_pos = get_position(state)\n"
            "    grass_coords = find_nearest('grass', state)\n"
            "    if grass_coords is not None and grass_coords != current_pos:\n"
            "        state = yield from go_to(grass_coords, state)\n"
            "    yield noop()\n"
        )
        _, func = load_skill(src)
        self.assertTrue(callable(func))


if __name__ == "__main__":
    unittest.main()
