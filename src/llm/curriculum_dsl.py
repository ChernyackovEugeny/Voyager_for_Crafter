"""Completion-condition DSL for curriculum sub-tasks."""
from __future__ import annotations

import logging
import operator
from typing import Any, Iterable

from environment.achievements import ACHIEVEMENTS

logger = logging.getLogger(__name__)


INVENTORY_KEYS: frozenset[str] = frozenset(
    {
        "health",
        "food",
        "drink",
        "energy",
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
    }
)
ACHIEVEMENT_KEYS: frozenset[str] = frozenset(ACHIEVEMENTS)
ALLOWED_KEYS: frozenset[str] = frozenset(
    {f"inventory.{key}" for key in INVENTORY_KEYS}
    | {f"achievements.{key}" for key in ACHIEVEMENT_KEYS}
)
ALLOWED_OPS: frozenset[str] = frozenset({">=", "<=", "==", ">", "<"})

_OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}


def evaluate(conditions: Iterable[Any], info: dict[str, Any]) -> bool:
    """Return True when all conditions are satisfied by the current info."""
    return all(evaluate_one(condition, info) for condition in conditions)


def evaluate_one(condition: Any, info: dict[str, Any]) -> bool:
    key = getattr(condition, "key", None)
    op = getattr(condition, "op", None)
    expected = getattr(condition, "value", None)
    if key not in ALLOWED_KEYS:
        logger.warning("curriculum DSL: unknown key %r", key)
        return False
    if op not in _OPS:
        logger.warning("curriculum DSL: unknown op %r", op)
        return False

    actual = _lookup(key, info)
    try:
        return bool(_OPS[op](actual, expected))
    except TypeError:
        logger.warning("curriculum DSL: invalid comparison %r %s %r", actual, op, expected)
        return False


def _lookup(key: str, info: dict[str, Any]) -> int | float:
    namespace, item = key.split(".", 1)
    values = info.get(namespace, {}) or {}
    return values.get(item, 0) or 0
