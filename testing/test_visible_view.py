import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.captioner import caption
from environment.ids import NAME_TO_ID
from environment.view import visible_semantic_window
from skills.primitives import find_nearest, go_to


class VisibleViewTests(unittest.TestCase):
    def _info(self, player_pos=(10, 10), view_size=(9, 9)):
        semantic = np.zeros((64, 64), dtype=int)
        return {
            "semantic": semantic,
            "player_pos": player_pos,
            "view_size": view_size,
            "inventory": {},
        }

    def test_find_nearest_uses_visible_window_and_returns_absolute_coords(self):
        info = self._info()
        info["semantic"][14, 10] = NAME_TO_ID["tree"]
        info["semantic"][15, 10] = NAME_TO_ID["tree"]

        self.assertEqual(find_nearest("tree", {"obs": None, "info": info}), (14, 10))

        info["semantic"][14, 10] = 0
        self.assertIsNone(find_nearest("tree", {"obs": None, "info": info}))

    def test_caption_uses_same_visible_window(self):
        info = self._info()
        info["semantic"][14, 10] = NAME_TO_ID["tree"]
        info["semantic"][20, 20] = NAME_TO_ID["diamond"]

        text = caption(None, info)

        self.assertIn("tree", text)
        self.assertNotIn("diamond", text)

    def test_visible_window_clamps_at_map_edge(self):
        info = self._info(player_pos=(0, 0))
        info["semantic"][4, 4] = NAME_TO_ID["water"]

        window, offset = visible_semantic_window(info)

        self.assertEqual(offset, (0, 0))
        self.assertEqual(window.shape, (5, 5))
        self.assertEqual(int(window[4, 4]), NAME_TO_ID["water"])

    def test_go_to_still_accepts_known_coords_outside_current_view(self):
        info = self._info(player_pos=(0, 0))
        info["semantic"][:, :] = NAME_TO_ID["grass"]

        generator = go_to((10, 0), {"obs": None, "info": info})

        self.assertIsInstance(next(generator), int)
