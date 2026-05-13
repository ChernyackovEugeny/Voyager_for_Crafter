"""Hardcoded curriculum for the Crafter achievement tree."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from environment.achievements import ACHIEVEMENTS, TECH_TREE_ORDER
from environment.danger import hostile_visible

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    """A unit of work for the agent."""

    name: str
    description: str
    achievement_key: str | None

    @property
    def skip_key(self) -> str:
        return self.achievement_key or self.name


@dataclass(frozen=True)
class SurvivalSettings:
    enter_health: int = 7
    enter_food: int = 6
    enter_drink: int = 6
    exit_health: int = 8
    exit_food: int = 6
    exit_drink: int = 6
    max_consecutive_failures: int = 3


@dataclass(frozen=True)
class _TaskFailure:
    task: Task
    state_snapshot: dict[str, Any]


class HardcodedCurriculum:
    """Prerequisite-aware linear walk through the achievement catalog."""

    def __init__(self, survival: SurvivalSettings | None = None) -> None:
        self._failures: list[_TaskFailure] = []
        self._survival = survival or SurvivalSettings()

    def propose_task(
        self,
        info: dict[str, Any],
        *,
        skip: set[str] | None = None,
    ) -> Task | None:
        """Return the first unfinished unlocked achievement, or None if done."""
        completed = self._completed_from_info(info)
        skipped = skip or set()
        if hostile_visible(info):
            return self._survive_task()
        if "survive" not in skipped and self._is_in_danger(info):
            return self._survive_task()
        for achievement_key in TECH_TREE_ORDER:
            if achievement_key in completed:
                continue
            if achievement_key in skipped:
                continue
            achievement = ACHIEVEMENTS[achievement_key]
            if not all(prereq in completed for prereq in achievement.prerequisites):
                continue
            return Task(
                name=achievement.key.replace("_", "-"),
                description=achievement.description,
                achievement_key=achievement.key,
            )
        return None

    def is_task_complete(self, task: Task, info: dict[str, Any]) -> bool:
        """Check Level-1 task completion through Crafter achievement flags."""
        if task.name == "survive":
            return self._is_recovered(info) and not hostile_visible(info)
        if task.achievement_key is None:
            return False
        return bool(info.get("achievements", {}).get(task.achievement_key, 0))

    def record_task_failed(
        self,
        task: Task,
        state_snapshot: dict[str, Any],
    ) -> None:
        """Store a failure for future adaptive or LLM-driven curricula."""
        self._failures.append(
            _TaskFailure(task=task, state_snapshot=state_snapshot)
        )
        logger.debug("curriculum: recorded failure for %s", task.name)

    @property
    def failures(self) -> tuple[_TaskFailure, ...]:
        """Read-only view for diagnostics and future LLM-curriculum context."""
        return tuple(self._failures)

    @staticmethod
    def _survive_task() -> Task:
        return Task(
            name="survive",
            description=(
                "Restore survival stats. If hungry, find and eat food. "
                "If thirsty, find water and drink. If health is low, rest safely. "
                "Avoid visible enemies and stop only when no hostile is visible "
                "and health, food, and drink are recovered."
            ),
            achievement_key=None,
        )

    def _is_in_danger(self, info: dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return (
            hostile_visible(info)
            or int(inventory.get("health", 9)) <= self._survival.enter_health
            or int(inventory.get("food", 9)) <= self._survival.enter_food
            or int(inventory.get("drink", 9)) <= self._survival.enter_drink
        )

    def _is_recovered(self, info: dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return (
            int(inventory.get("health", 0)) >= self._survival.exit_health
            and int(inventory.get("food", 0)) >= self._survival.exit_food
            and int(inventory.get("drink", 0)) >= self._survival.exit_drink
        )

    @staticmethod
    def _completed_from_info(info: dict[str, Any]) -> set[str]:
        return {
            key
            for key, value in info.get("achievements", {}).items()
            if value
        }
