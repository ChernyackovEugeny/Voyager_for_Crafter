"""Helpers for converting full Crafter semantic maps into the visible view."""

from __future__ import annotations

import numpy as np


DEFAULT_VIEW_SIZE: tuple[int, int] = (9, 9)


def visible_semantic_window(info: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Return the semantic subarray visible to the player and its absolute offset.

    Crafter exposes the full semantic map in info["semantic"]. Agent perception
    should use only the current view window, centered on info["player_pos"].
    """
    semantic = info.get("semantic")
    if semantic is None:
        return np.zeros((0, 0), dtype=int), (0, 0)

    px, py = info.get("player_pos", (0, 0))
    view_w, view_h = info.get("view_size", DEFAULT_VIEW_SIZE)
    px, py = int(px), int(py)
    view_w, view_h = int(view_w), int(view_h)

    left = view_w // 2
    right = view_w - left
    top = view_h // 2
    bottom = view_h - top

    x0 = max(0, px - left)
    x1 = min(int(semantic.shape[0]), px + right)
    y0 = max(0, py - top)
    y1 = min(int(semantic.shape[1]), py + bottom)

    return semantic[x0:x1, y0:y1], (x0, y0)
