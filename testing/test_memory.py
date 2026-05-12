import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.memory import SpatialMemory
from skills.runner import SkillRuntime, load_skill


class SpatialMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SpatialMemory()

    def test_memory_returns_copy(self):
        self.memory.memory_add("water", (3, 4))

        snapshot = self.memory.get_memory()
        snapshot["water"] = (9, 9)

        self.assertEqual(self.memory.get_memory(), {"water": (3, 4)})

    def test_memory_runtime_functions_delegate_to_bound_memory(self):
        src = (
            "def use_memory(state):\n"
            "    save_in_memory('table', (8, 5))\n"
            "    set_home((1, 2))\n"
            "    yield 0\n"
            "    delete_memory('table')\n"
        )
        _, func = load_skill(src, SkillRuntime(memory=self.memory))

        gen = func({})
        self.assertEqual(next(gen), 0)
        self.assertEqual(self.memory.get_memory(), {"table": (8, 5)})
        self.assertEqual(self.memory.get_home(), (1, 2))

        with self.assertRaises(StopIteration):
            next(gen)
        self.assertEqual(self.memory.get_memory(), {})

    def test_reset_clears_locations_and_home(self):
        self.memory.memory_add("stone", (4, 4))
        self.memory.set_home((1, 1))

        self.memory.reset()

        self.assertEqual(self.memory.get_memory(), {})
        self.assertIsNone(self.memory.get_home())
        self.assertEqual(self.memory.text_memory(), "empty")
