"""Shared survival-task helpers for curriculum implementations."""
from __future__ import annotations

from typing import Any

from environment.danger import hostile_visible


def make_survive_task(task_cls):
    return task_cls(
        name="survive",
        description=(
            "Restore survival stats. If hungry, find and eat food. "
            "If thirsty, find water and drink. If health is low, rest safely. "
            "Use get_memory() and get_home() before random exploration, save "
            "water/food/base locations when found, avoid visible enemies, and "
            "stop only when no hostile is visible and health, food, and drink "
            "are recovered."
        ),
        achievement_key=None,
    )


def is_in_danger(info: dict[str, Any], cfg) -> bool:
    inventory = info.get("inventory", {})
    return (
        hostile_visible(info)
        or int(inventory.get("health", 9)) <= cfg.enter_health
        or int(inventory.get("food", 9)) <= cfg.enter_food
        or int(inventory.get("drink", 9)) <= cfg.enter_drink
    )


def is_recovered(info: dict[str, Any], cfg) -> bool:
    inventory = info.get("inventory", {})
    return (
        int(inventory.get("health", 0)) >= cfg.exit_health
        and int(inventory.get("food", 0)) >= cfg.exit_food
        and int(inventory.get("drink", 0)) >= cfg.exit_drink
    )
