import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crafter.constants as C
from environment.ids import NAME_TO_ID
from skills.primitives import craft, find_nearest, get_position, go_to, place


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

    def test_go_to_faces_non_walkable_target_before_returning(self):
        semantic = np.full((64, 64), NAME_TO_ID["grass"], dtype=int)
        semantic[2, 0] = NAME_TO_ID["tree"]
        state = {
            "obs": None,
            "info": {
                "semantic": semantic,
                "player_pos": (0, 0),
                "view_size": (9, 9),
                "inventory": {},
            },
        }
        generator = go_to((2, 0), state)

        first_action = next(generator)
        self.assertEqual(first_action, C.actions.index("move_right"))

        state["info"]["player_pos"] = (1, 0)
        second_action = generator.send(state)
        self.assertEqual(second_action, C.actions.index("move_right"))

        with self.assertRaises(StopIteration):
            generator.send(state)

    def test_go_to_does_not_route_through_blocked_adjacent_cell(self):
        semantic = np.full((64, 64), NAME_TO_ID["grass"], dtype=int)
        semantic[1, 0] = NAME_TO_ID["stone"]
        semantic[2, 0] = NAME_TO_ID["tree"]
        state = {
            "obs": None,
            "info": {
                "semantic": semantic,
                "player_pos": (0, 0),
                "view_size": (9, 9),
                "inventory": {},
            },
        }
        generator = go_to((2, 0), state)

        first_action = next(generator)
        self.assertNotEqual(first_action, C.actions.index("move_right"))

    def test_go_to_returns_without_error_for_missing_target(self):
        state = {
            "obs": None,
            "info": {
                "semantic": np.full((64, 64), NAME_TO_ID["grass"], dtype=int),
                "player_pos": (0, 0),
                "view_size": (9, 9),
                "inventory": {},
            },
        }
        generator = go_to(None, state)

        with self.assertRaises(StopIteration) as ctx:
            next(generator)
        self.assertIs(ctx.exception.value, state)
