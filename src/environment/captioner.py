
import numpy as np
import crafter.constants as C

from environment.ids import ID_TO_NAME as _ID_TO_NAME

# Background tiles not worth mentioning every step
_BACKGROUND: frozenset[str] = frozenset({"grass", "path", "player"})

# Items that go into Status, not Inventory
_STATUS_KEYS: frozenset[str] = frozenset({"health", "food", "drink", "energy"})


def caption(obs: np.ndarray, info: dict) -> str:
    """
    Convert raw Crafter observation + info dict to a compact text state.

    Args:
        obs:  Raw pixel observation from env.step() — not used directly,
              semantic data is taken from info["semantic"].
        info: Info dict returned by env.step().

    Returns:
        Multi-line string, e.g.:
            Observation: tree, stone, zombie
            Inventory: wood: 3, stone_pickaxe: 1
            Status: health: 7/9, food: 5/9, drink: 4/9, energy: 9/9
    """
    return "\n".join([
        _observation_line(info),
        _inventory_line(info),
        _status_line(info),
    ])


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _observation_line(info: dict) -> str:
    semantic: np.ndarray = info.get("semantic", np.zeros((1, 1), dtype=int))
    unique_ids = np.unique(semantic)
    visible = sorted(
        _ID_TO_NAME[uid]
        for uid in unique_ids
        if uid in _ID_TO_NAME and _ID_TO_NAME[uid] not in _BACKGROUND
    )
    body = ", ".join(visible) if visible else "nothing notable"
    return f"Observation: {body}"


def _inventory_line(info: dict) -> str:
    inventory: dict = info.get("inventory", {})
    items = [
        f"{k}: {v}"
        for k, v in inventory.items()
        if k not in _STATUS_KEYS and v > 0
    ]
    body = ", ".join(items) if items else "empty"
    return f"Inventory: {body}"


def _status_line(info: dict) -> str:
    inventory: dict = info.get("inventory", {})
    maxima = {k: v["max"] for k, v in C.items.items() if k in _STATUS_KEYS}
    parts = [
        f"{k}: {inventory.get(k, 0)}/{maxima.get(k, 9)}"
        for k in ("health", "food", "drink", "energy")
    ]
    return f"Status: {', '.join(parts)}"
