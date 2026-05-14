from __future__ import annotations

import logging

from agent.strategies import AgentStrategy, SkillSource

logger = logging.getLogger(__name__)


class InferenceStrategy(AgentStrategy):
    """Inference mode: use the frozen skill library, never generate or mutate."""

    name = "inference"

    def __init__(self, *, reuse_threshold: float) -> None:
        self._reuse_threshold = reuse_threshold

    def acquire_skill(
        self,
        *,
        task,
        obs,
        info: dict,
        candidates,
    ):
        best = self._select_reusable_skill(task, candidates)
        if best is not None:
            logger.info("[Agent] reuse: %s", best.skill.name)
            extra_skills = tuple(
                (candidate.skill.name, candidate.skill.code)
                for candidate in candidates
            )
            return SkillSource(
                code=best.skill.code,
                reused_name=best.skill.name,
                generated=False,
                extra_skills=extra_skills,
                allowed_skill_names=frozenset(name for name, _ in extra_skills),
            )
        logger.info("[Agent] inference skip: no reusable skill for %s", task.name)
        return None

    def on_skill_unavailable(self, *, task, state: dict, curriculum) -> None:
        logger.info("[Agent] task skipped: %s", task.name)

    def on_task_completed(self, *, task, source: SkillSource, skill_manager) -> None:
        logger.info("[Agent] task success: %s", task.name)

    def on_task_failed(
        self,
        *,
        task,
        source: SkillSource,
        state: dict,
        skill_manager,
        curriculum,
    ) -> None:
        logger.info("[Agent] task failed: %s", task.name)

    def retrieval_route(self, candidates, task=None) -> str:
        if self._select_reusable_skill(task, candidates) is not None:
            return "reuse"
        return "skip"

    @property
    def reuse_threshold(self) -> float:
        return self._reuse_threshold

    def _select_reusable_skill(self, task, candidates):
        for candidate in candidates:
            if candidate.similarity < self._reuse_threshold:
                continue
            if not _candidate_matches_task(task, candidate):
                logger.info(
                    "[Agent] reuse rejected: %s incompatible with task %s",
                    candidate.skill.name,
                    getattr(task, "name", None),
                )
                continue
            return candidate
        return None


def _candidate_matches_task(task, candidate) -> bool:
    if task is None:
        return True

    skill_name = candidate.skill.name
    achievement_key = getattr(task, "achievement_key", None)
    if achievement_key is not None:
        return _skill_name_matches_base(skill_name, achievement_key)

    task_name = getattr(task, "name", "")
    if task_name == "survive":
        return _skill_name_matches_base(skill_name, "survive")

    for condition in getattr(task, "completion_conditions", ()) or ():
        key = getattr(condition, "key", "")
        if isinstance(key, str) and key.startswith("achievements."):
            achievement_key = key.split(".", 1)[1]
            return _skill_name_matches_base(skill_name, achievement_key)

    return _skill_name_matches_base(skill_name, str(task_name).replace("-", "_"))


def _skill_name_matches_base(skill_name: str | None, base: str | None) -> bool:
    if not skill_name or not base:
        return False
    if skill_name == base or skill_name.startswith(f"{base}_v"):
        return True
    root = _skill_root(skill_name)
    return base.startswith(f"{root}_")


def _skill_root(skill_name: str) -> str:
    head, sep, tail = skill_name.rpartition("_v")
    if sep and tail.isdigit():
        return head
    return skill_name
