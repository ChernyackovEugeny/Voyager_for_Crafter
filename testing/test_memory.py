import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.memory import SpatialMemory
from skills import primitives


class SpatialMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = SpatialMemory()
        primitives.set_memory(self.memory)

    def test_memory_returns_copy(self):
        self.memory.memory_add("water", (3, 4))

        snapshot = self.memory.get_memory()
        snapshot["water"] = (9, 9)

        self.assertEqual(self.memory.get_memory(), {"water": (3, 4)})

    def test_memory_wrappers_delegate_to_bound_memory(self):
        primitives.save_in_memory("table", (8, 5))
        primitives.set_home((1, 2))

        self.assertEqual(primitives.get_memory(), {"table": (8, 5)})
        self.assertEqual(primitives.get_home(), (1, 2))

        primitives.delete_memory("table")
        self.assertEqual(primitives.get_memory(), {})

    def test_reset_clears_locations_and_home(self):
        self.memory.memory_add("stone", (4, 4))
        self.memory.set_home((1, 1))

        self.memory.reset()

        self.assertEqual(self.memory.get_memory(), {})
        self.assertIsNone(self.memory.get_home())
        self.assertEqual(self.memory.text_memory(), "empty")
