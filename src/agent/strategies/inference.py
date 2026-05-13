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
        if candidates and candidates[0].similarity >= self._reuse_threshold:
            best = candidates[0]
            logger.info("[Agent] reuse: %s", best.skill.name)
            return SkillSource(
                code=best.skill.code,
                reused_name=best.skill.name,
                generated=False,
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

    def retrieval_route(self, candidates) -> str:
        if candidates and candidates[0].similarity >= self._reuse_threshold:
            return "reuse"
        return "skip"

    @property
    def reuse_threshold(self) -> float:
        return self._reuse_threshold
