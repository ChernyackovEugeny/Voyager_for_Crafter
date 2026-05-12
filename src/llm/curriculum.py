"""Hardcoded curriculum for the Crafter achievement tree."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from environment.achievements import ACHIEVEMENTS, TECH_TREE_ORDER

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Task:
    """A unit of work for the agent."""

    name: str
    description: str
    achievement_key: str | None


@dataclass(frozen=True)
class _TaskFailure:
    task: Task
    state_snapshot: dict[str, Any]


class HardcodedCurriculum:
    """Prerequisite-aware linear walk through the achievement catalog."""

    def __init__(self) -> None:
        self._failures: list[_TaskFailure] = []

    def propose_task(self, info: dict[str, Any]) -> Task | None:
        """Return the first unfinished unlocked achievement, or None if done."""
        completed = self._completed_from_info(info)
        for achievement_key in TECH_TREE_ORDER:
            if achievement_key in completed:
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
    def _completed_from_info(info: dict[str, Any]) -> set[str]:
        return {
            key
            for key, value in info.get("achievements", {}).items()
            if value
        }
