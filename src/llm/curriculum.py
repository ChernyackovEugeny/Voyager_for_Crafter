"""Curriculum implementations for the Crafter achievement tree."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Protocol

import openai

from config import get_settings
from environment.achievements import ACHIEVEMENTS, TECH_TREE_ORDER
from environment.danger import hostile_visible
from llm.curriculum_dsl import ALLOWED_KEYS, ALLOWED_OPS, evaluate
from llm.pricing import compute_cost
from llm.survival import is_in_danger, is_recovered, make_survive_task
from prompts.curriculum_prompt import (
    CURRICULUM_TEMPLATE_ID,
    SYSTEM_PROMPT,
    format_user_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    """A single numeric completion condition for a sub-task."""

    key: str
    op: str
    value: int | float


@dataclass(frozen=True)
class Task:
    """A unit of work for the agent."""

    name: str
    description: str
    achievement_key: str | None
    completion_conditions: tuple[Condition, ...] = ()

    @property
    def skip_key(self) -> str:
        return self.achievement_key or self.name


class Curriculum(Protocol):
    def propose_task(
        self,
        info: dict[str, Any],
        *,
        skip: set[str] | None = None,
    ) -> Task | None: ...

    def is_task_complete(self, task: Task, info: dict[str, Any]) -> bool: ...

    def record_task_failed(self, task: Task, state_snapshot: dict[str, Any]) -> None: ...

    def record_task_completed(
        self,
        task: Task,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None: ...

    def drain_pending_llm_calls(self) -> tuple[tuple[str, object], ...]: ...


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
    failure_reason: str | None
    inventory_summary: dict[str, int]
    position: tuple[int, int] | None
    achievements_at_failure: tuple[str, ...]
    executor_reason: str | None
    executor_steps: int | None
    error_first_line: str | None


@dataclass(frozen=True)
class CurriculumCall:
    """Curriculum LLM result plus billing/latency metadata."""

    task: Task | None
    raw_response: str
    model: str
    prompt_template_id: str
    prompt_hash: str
    prompt_tokens: int
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    latency_ms: int
    cost_usd: float
    prompt_text: str | None = None

    @property
    def tokens_in(self) -> int:
        return self.prompt_tokens

    @property
    def tokens_out(self) -> int:
        return self.completion_tokens


class CurriculumParseError(Exception):
    """Raised when an LLM curriculum response cannot be used safely."""


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
            return make_survive_task(Task)
        if "survive" not in skipped and is_in_danger(info, self._survival):
            return make_survive_task(Task)
        survival_task = _survival_foundation_task(info, completed, skipped)
        if survival_task is not None:
            return survival_task
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
        """Check Level-1 or DSL task completion."""
        if task.name == "survive":
            return is_recovered(info, self._survival) and not hostile_visible(info)
        if task.achievement_key is not None:
            return bool(info.get("achievements", {}).get(task.achievement_key, 0))
        if task.completion_conditions:
            return evaluate(task.completion_conditions, info)
        return False

    def record_task_failed(
        self,
        task: Task,
        state_snapshot: dict[str, Any],
    ) -> None:
        """Store a compact failure for diagnostics and LLM-curriculum context."""
        self._failures.append(_compact_failure(task, state_snapshot))
        logger.debug("curriculum: recorded failure for %s", task.name)

    def record_task_completed(
        self,
        task: Task,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        return None

    def drain_pending_llm_calls(self) -> tuple[tuple[str, object], ...]:
        return ()

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


class LLMCurriculum:
    """LLM-backed training curriculum with hardcoded safety fallback."""

    def __init__(
        self,
        *,
        survival: SurvivalSettings | None = None,
        client: openai.OpenAI | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
        max_failures_in_context: int = 5,
        max_retries: int = 1,
        fallback: Curriculum | None = None,
    ) -> None:
        cfg = get_settings().llm
        self._survival = survival or SurvivalSettings()
        self._model = model or cfg.curriculum_model
        self._temperature = (
            cfg.curriculum_temperature if temperature is None else temperature
        )
        self._max_failures_in_context = max_failures_in_context
        self._max_retries = max_retries
        self._fallback = fallback or HardcodedCurriculum(survival=self._survival)
        self._client = client or openai.OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=timeout_s or cfg.curriculum_timeout_s,
        )
        self._succeeded: list[Task] = []
        self._failed: list[_TaskFailure] = []
        self._pending_calls: list[tuple[str, CurriculumCall]] = []

    def propose_task(
        self,
        info: dict[str, Any],
        *,
        skip: set[str] | None = None,
    ) -> Task | None:
        skipped = skip or set()
        if hostile_visible(info):
            return make_survive_task(Task)
        if "survive" not in skipped and is_in_danger(info, self._survival):
            return make_survive_task(Task)

        completed = _completed_from_info(info)
        survival_task = _survival_foundation_task(info, completed, skipped)
        if survival_task is not None:
            return survival_task
        if len(completed) >= len(ACHIEVEMENTS):
            return None
        available = _available_achievements(info, skipped)
        if not available and not self._failed:
            return None

        user_prompt = format_user_prompt(
            info=info,
            available=available,
            completed=completed,
            succeeded=self._succeeded[-10:],
            failures=self._failed[-self._max_failures_in_context :],
            skip=skipped,
        )
        for attempt in range(self._max_retries + 1):
            try:
                call = self._call_api(user_prompt)
                self._pending_calls.append(("curriculum", call))
                task = self._parse_and_validate(
                    call.raw_response,
                    info=info,
                    available=set(available),
                    skip=skipped,
                )
                return task
            except Exception as exc:
                if isinstance(exc, CurriculumParseError) and attempt < self._max_retries:
                    user_prompt = f"{user_prompt}\n\nPrevious output rejected: {exc}. Return corrected raw JSON only."
                    continue
                logger.warning("LLMCurriculum: falling back to hardcoded: %s", exc)
                break

        return self._fallback.propose_task(info, skip=skip)

    def is_task_complete(self, task: Task, info: dict[str, Any]) -> bool:
        if task.name == "survive":
            return is_recovered(info, self._survival) and not hostile_visible(info)
        if task.achievement_key is not None:
            return bool(info.get("achievements", {}).get(task.achievement_key, 0))
        if task.completion_conditions:
            return evaluate(task.completion_conditions, info)
        return False

    def record_task_failed(
        self,
        task: Task,
        state_snapshot: dict[str, Any],
    ) -> None:
        failure = _compact_failure(task, state_snapshot)
        self._failed.append(failure)
        self._fallback.record_task_failed(task, state_snapshot)

    def record_task_completed(
        self,
        task: Task,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._succeeded.append(task)
        completed = getattr(self._fallback, "record_task_completed", None)
        if completed is not None:
            completed(task, state_snapshot)

    def drain_pending_llm_calls(self) -> tuple[tuple[str, object], ...]:
        calls = tuple(self._pending_calls)
        self._pending_calls = []
        return calls

    @property
    def failures(self) -> tuple[_TaskFailure, ...]:
        return tuple(self._failed)

    def _call_api(self, user_prompt: str) -> CurriculumCall:
        prompt_hash = _prompt_hash(user_prompt)
        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        raw = response.choices[0].message.content
        usage = _extract_usage(response)
        cost = compute_cost(
            self._model,
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        return CurriculumCall(
            task=None,
            raw_response=raw,
            model=self._model,
            prompt_template_id=CURRICULUM_TEMPLATE_ID,
            prompt_hash=prompt_hash,
            prompt_tokens=usage["prompt_tokens"],
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            cost_usd=cost,
            prompt_text=user_prompt,
        )

    def _parse_and_validate(
        self,
        raw: str,
        *,
        info: dict[str, Any],
        available: set[str],
        skip: set[str],
    ) -> Task:
        try:
            obj = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise CurriculumParseError(f"invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise CurriculumParseError("response must be a JSON object")

        name = obj.get("name")
        description = obj.get("description")
        achievement_key = obj.get("achievement_key")
        conditions_raw = obj.get("conditions", [])

        if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9-]{0,31}", name) is None:
            raise CurriculumParseError("name must be kebab-case and <=32 chars")
        if not isinstance(description, str) or not description.strip() or len(description) > 240:
            raise CurriculumParseError("description must be a non-empty string <=240 chars")
        if achievement_key is not None and not isinstance(achievement_key, str):
            raise CurriculumParseError("achievement_key must be string or null")
        if not isinstance(conditions_raw, list):
            raise CurriculumParseError("conditions must be an array")
        if len(conditions_raw) > 4:
            raise CurriculumParseError("conditions must contain at most 4 entries")
        if name in skip:
            raise CurriculumParseError(f"task {name} is in skip list")

        if achievement_key is not None:
            if achievement_key not in ACHIEVEMENTS:
                raise CurriculumParseError(f"unknown achievement_key: {achievement_key}")
            if achievement_key not in available:
                raise CurriculumParseError(
                    f"achievement {achievement_key} not currently unlocked"
                )
            if achievement_key in skip:
                raise CurriculumParseError(f"achievement {achievement_key} is in skip list")
            if conditions_raw:
                raise CurriculumParseError(
                    "achievement tasks must not include completion conditions"
                )
        elif not conditions_raw:
            raise CurriculumParseError(
                "non-achievement task must have completion_conditions"
            )

        conditions = tuple(_parse_condition(item) for item in conditions_raw)
        if achievement_key is None and evaluate(conditions, info):
            raise CurriculumParseError("non-achievement task is already complete")
        return Task(
            name=name,
            description=description.strip(),
            achievement_key=achievement_key,
            completion_conditions=conditions,
        )


def _parse_condition(raw: Any) -> Condition:
    if not isinstance(raw, dict):
        raise CurriculumParseError("condition must be an object")
    key = raw.get("key")
    op = raw.get("op")
    value = raw.get("value")
    if key not in ALLOWED_KEYS:
        raise CurriculumParseError(f"unknown condition key: {key}")
    if op not in ALLOWED_OPS:
        raise CurriculumParseError(f"unknown condition op: {op}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CurriculumParseError("condition value must be a finite number")
    return Condition(key=str(key), op=str(op), value=value)


def _extract_json(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def _compact_failure(task: Task, state_snapshot: dict[str, Any]) -> _TaskFailure:
    info = state_snapshot.get("info", {}) or {}
    inventory = info.get("inventory", {}) or {}
    position = info.get("player_pos")
    error = state_snapshot.get("error_traceback")
    return _TaskFailure(
        task=task,
        failure_reason=state_snapshot.get("failure_reason"),
        inventory_summary={
            key: int(value)
            for key, value in inventory.items()
            if int(value or 0) != 0
        },
        position=tuple(position) if position is not None else None,
        achievements_at_failure=tuple(sorted(_completed_from_info(info))),
        executor_reason=state_snapshot.get("executor_reason"),
        executor_steps=state_snapshot.get("executor_steps"),
        error_first_line=error.splitlines()[0][:200] if error else None,
    )


def _survival_foundation_task(
    info: dict[str, Any],
    completed: set[str],
    skipped: set[str],
) -> Task | None:
    """Prioritize water, food, and a remembered base before achievement chasing."""
    if "collect_drink" not in completed and "collect_drink" not in skipped:
        return Task(
            name="secure-water",
            description=(
                "Find water, save its coordinates in spatial memory as 'water', "
                "go to it, face it, and drink until collect_drink is unlocked."
            ),
            achievement_key="collect_drink",
        )
    if "eat_cow" not in completed and "eat_cow" not in skipped:
        return Task(
            name="secure-food",
            description=(
                "Find a cow or edible plant, save food coordinates in spatial "
                "memory, approach safely, and eat until eat_cow is unlocked."
            ),
            achievement_key="eat_cow",
        )
    if "place_table" not in completed and "build-shelter" not in skipped:
        return Task(
            name="build-shelter",
            description=(
                "Build a minimal home base before pursuing achievements: collect "
                "wood if needed, place a crafting table on safe ground, save the "
                "base with set_home(get_position(state)), and retreat from visible "
                "hostiles while doing it."
            ),
            achievement_key=None,
            completion_conditions=(
                Condition("achievements.place_table", "==", 1),
            ),
        )
    return None


def _available_achievements(info: dict[str, Any], skip: set[str]) -> tuple[str, ...]:
    completed = _completed_from_info(info)
    available: list[str] = []
    for key in TECH_TREE_ORDER:
        if key in completed or key in skip:
            continue
        achievement = ACHIEVEMENTS[key]
        if all(prereq in completed for prereq in achievement.prerequisites):
            available.append(key)
    return tuple(available)


def _completed_from_info(info: dict[str, Any]) -> set[str]:
    return {
        key
        for key, value in info.get("achievements", {}).items()
        if value
    }


def _prompt_hash(user_prompt: str) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    return hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()[:16]


def _extract_usage(response) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": None,
        }

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
    miss_tokens = getattr(usage, "prompt_cache_miss_tokens", None)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    if hit_tokens is None and miss_tokens is None:
        hit_tokens = 0
        miss_tokens = prompt_tokens
    else:
        hit_tokens = int(hit_tokens or 0)
        miss_tokens = int(miss_tokens or 0)
        if prompt_tokens == 0:
            prompt_tokens = hit_tokens + miss_tokens

    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning_tokens is not None:
        reasoning_tokens = int(reasoning_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": int(hit_tokens),
        "prompt_cache_miss_tokens": int(miss_tokens),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
