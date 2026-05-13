"""Prompt construction for LLM-backed curriculum selection."""
from __future__ import annotations

from typing import Any, Iterable

from environment.achievements import ACHIEVEMENTS
from environment.captioner import caption
from llm.curriculum_dsl import ACHIEVEMENT_KEYS, INVENTORY_KEYS

CURRICULUM_TEMPLATE_ID = "curriculum.v1"

_SYSTEM_ACHIEVEMENTS = "\n".join(
    f"- {key}: {achievement.description} "
    f"(prerequisites: {', '.join(achievement.prerequisites) or 'none'})"
    for key, achievement in ACHIEVEMENTS.items()
)

SYSTEM_PROMPT = f"""You choose one next training task for a Crafter agent.

World summary:
- The agent survives by managing health, food, drink, and energy.
- The agent can collect resources, craft tools, place objects, fight hostiles, drink water, eat cows/plants, and sleep.
- Training tasks should be small enough for one skill attempt.

Achievement graph:
{_SYSTEM_ACHIEVEMENTS}

Completion-condition DSL for non-achievement sub-tasks:
- Allowed inventory keys: {', '.join(sorted(INVENTORY_KEYS))}
- Allowed achievement keys: {', '.join(sorted(ACHIEVEMENT_KEYS))}
- Keys must be namespaced as inventory.<key> or achievements.<key>.
- Allowed operators: >=, <=, ==, >, <
- Conditions are ANDed together.

Task rules:
- Survival foundation comes first: secure water, secure food, and build a small
  remembered home/base before optimizing for the achievement graph.
- Treat achievements as secondary until the agent has reliable drink, food, and
  a placed crafting table/home base.
- Prefer achievements listed in Available next achievements.
- If a high-level achievement keeps failing, propose a smaller non-achievement sub-task using conditions.
- Never propose a non-achievement sub-task whose conditions are already satisfied in Current state.
- If place_table is available and inventory.wood is at least 2, choose place_table instead of more wood stockpiling.
- Do not propose skipped or recently failed tasks when alternatives exist.
- Achievement tasks must set achievement_key and must not include conditions.
- Non-achievement sub-tasks must set achievement_key to null and include at least one condition.
- Keep task names short, kebab-case, and at most 32 characters.

Return raw JSON only, with this shape:
{{
  "name": "collect-wood",
  "description": "Chop a tree to obtain wood.",
  "achievement_key": "collect_wood",
  "conditions": []
}}

Example sub-task:
{{
  "name": "build-shelter",
  "description": "Collect wood if needed, place a crafting table as a home base, and remember it.",
  "achievement_key": null,
  "conditions": [{{"key": "achievements.place_table", "op": "==", "value": 1}}]
}}
"""


def format_user_prompt(
    *,
    info: dict[str, Any],
    available: Iterable[str],
    completed: Iterable[str],
    succeeded: Iterable[Any],
    failures: Iterable[Any],
    skip: set[str],
) -> str:
    available = tuple(available)
    completed = tuple(sorted(completed))
    succeeded = tuple(succeeded)
    failures = tuple(failures)

    return "\n".join(
        [
            "## Current state",
            caption(None, info),
            "",
            "## Completed achievements",
            _csv(completed),
            "",
            "## Available next achievements (unlocked, not yet completed)",
            _format_available(available),
            "",
            "## Recently succeeded tasks",
            _format_succeeded(succeeded),
            "",
            "## Recently failed tasks",
            _format_failures(failures),
            "",
            "## Skipped this episode",
            _csv(sorted(skip)),
            "",
            "## Your job",
            "Propose one next task as raw JSON only.",
        ]
    )


def _format_available(keys: Iterable[str]) -> str:
    lines = []
    for key in keys:
        achievement = ACHIEVEMENTS[key]
        prereqs = ", ".join(achievement.prerequisites) or "none"
        lines.append(f"- {key}: {achievement.description} (requires: {prereqs})")
    return "\n".join(lines) if lines else "None"


def _format_succeeded(tasks: Iterable[Any]) -> str:
    lines = [f"- {task.name}: {task.description}" for task in tasks]
    return "\n".join(lines) if lines else "None"


def _format_failures(failures: Iterable[Any]) -> str:
    lines = []
    for idx, failure in enumerate(failures, start=1):
        lines.append(
            " | ".join(
                [
                    f"{idx}. task={failure.task.name}",
                    f"reason={failure.failure_reason or 'unknown'}",
                    f"steps={failure.executor_steps if failure.executor_steps is not None else 'unknown'}",
                    f"inventory={failure.inventory_summary}",
                    f"position={failure.position}",
                    f"achievements={list(failure.achievements_at_failure)}",
                    f"error={failure.error_first_line or 'none'}",
                ]
            )
        )
    return "\n".join(lines) if lines else "None"


def _csv(values: Iterable[str]) -> str:
    values = tuple(values)
    return ", ".join(values) if values else "None"
