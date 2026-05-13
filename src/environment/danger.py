"""Shared danger perception helpers for curriculum and executor guards."""
from __future__ import annotations

import numpy as np

from environment.ids import NAME_TO_ID
from environment.view import visible_semantic_window


HOSTILE_NAMES: frozenset[str] = frozenset({"zombie", "skeleton", "arrow"})
HOSTILE_IDS: frozenset[int] = frozenset(
    NAME_TO_ID[name] for name in HOSTILE_NAMES if name in NAME_TO_ID
)


def visible_hostiles(info: dict) -> tuple[str, ...]:
    """Return hostile entities currently inside the visible semantic window."""
    semantic, _ = visible_semantic_window(info)
    if semantic.size == 0:
        return ()
    ids = set(int(value) for value in np.unique(semantic))
    return tuple(sorted(name for name in HOSTILE_NAMES if NAME_TO_ID.get(name) in ids))


def hostile_visible(info: dict) -> bool:
    """True when a hostile entity or projectile is visible."""
    semantic, _ = visible_semantic_window(info)
    if semantic.size == 0:
        return False
    return any(int(value) in HOSTILE_IDS for value in np.unique(semantic))
