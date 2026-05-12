import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crafter.constants as C
from environment.ids import NAME_TO_ID
from skills.primitives import craft, find_nearest, get_position, place


class PrimitiveTests(unittest.TestCase):
    def test_action_helpers_match_crafter_action_indices(self):
        self.assertEqual(craft("wood_pickaxe"), C.actions.index("make_wood_pickaxe"))
        self.assertEqual(craft("iron_sword"), C.actions.index("make_iron_sword"))
        self.assertEqual(place("table"), C.actions.index("place_table"))
        self.assertEqual(place("plant"), C.actions.index("place_plant"))

    def test_get_position_casts_numpy_values_to_plain_ints(self):
        state = {"info": {"player_pos": np.array([7, 11])}}

        pos = get_position(state)

        self.assertEqual(pos, (7, 11))
        self.assertIs(type(pos[0]), int)
        self.assertIs(type(pos[1]), int)

    def test_find_nearest_returns_none_for_unknown_or_not_visible_object(self):
        semantic = np.zeros((64, 64), dtype=int)
        semantic[30, 30] = NAME_TO_ID["diamond"]
        state = {
            "obs": None,
            "info": {
                "semantic": semantic,
                "player_pos": (10, 10),
                "view_size": (9, 9),
                "inventory": {},
            },
        }

        self.assertIsNone(find_nearest("not_a_real_object", state))
        self.assertIsNone(find_nearest("diamond", state))
