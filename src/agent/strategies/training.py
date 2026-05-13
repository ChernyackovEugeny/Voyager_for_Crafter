from __future__ import annotations

import logging

from agent.strategies import AgentStrategy, SkillSource
from environment.captioner import caption

logger = logging.getLogger(__name__)


class TrainingStrategy(AgentStrategy):
    """Training mode: reuse known skills, generate missing skills, and learn."""

    name = "train"

    def __init__(self, *, codegen, reuse_threshold: float) -> None:
        self._codegen = codegen
        self._reuse_threshold = reuse_threshold

    def acquire_skill(
        self,
        *,
        task,
        obs,
        info: dict,
        candidates,
    ):
        selected = self._select_reusable_skill(candidates)
        if selected is not None:
            logger.info("[Agent] reuse: %s", selected.skill.name)
            return SkillSource(
                code=selected.skill.code,
                reused_name=selected.skill.name,
                generated=False,
            )

        logger.info("[Agent] codegen: generating new skill for %s", task.name)
        try:
            call = self._codegen.get_code(
                state_text=caption(obs, info),
                task=task.description,
                retrieved_skills=self._retrieved_skill_dicts(candidates),
            )
        except Exception as exc:
            logger.warning("strategy: codegen failed for %s: %s", task.name, exc)
            return None
        return SkillSource(code=call.code, generated=True, llm_call=call)

    def on_skill_unavailable(self, *, task, state: dict, curriculum) -> None:
        logger.info("[Agent] task failed: %s", task.name)
        curriculum.record_task_failed(task, state)

    def on_task_completed(self, *, task, source: SkillSource, skill_manager) -> None:
        logger.info("[Agent] task success: %s", task.name)
        if source.reused_name is not None:
            logger.info("[Agent] metric: record_success(%s)", source.reused_name)
            skill_manager.record_success(source.reused_name)
            return

        self._save_new_skill(task, source.code, skill_manager)

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
        curriculum.record_task_failed(task, state)
        if source.reused_name is not None:
            logger.info("[Agent] metric: record_failure(%s)", source.reused_name)
            skill_manager.record_failure(source.reused_name)
            return

        logger.info("[Agent] discard: generated skill was not successful")
        logger.info("[Agent] failed generated skill code:\n%s", source.code)

    def retrieval_route(self, candidates) -> str:
        if not candidates:
            return "codegen"
        if candidates[0].similarity >= self._reuse_threshold:
            return "reuse"
        return "codegen"

    @property
    def reuse_threshold(self) -> float:
        return self._reuse_threshold

    def _select_reusable_skill(self, candidates):
        if not candidates:
            return None
        best = candidates[0]
        if best.similarity >= self._reuse_threshold:
            return best
        return None

    def _save_new_skill(self, task, code: str, skill_manager) -> None:
        name = self._unique_skill_name(task.name.replace("-", "_"), skill_manager)
        result = skill_manager.save(name=name, code=code, task=task.description)
        if result.outcome == "ok":
            logger.info("[Agent] save: %s | outcome=ok", name)
            logger.info("[Agent] metric: record_success(%s)", name)
            skill_manager.record_success(name)
        if result.outcome == "duplicate" and result.similar_to is not None:
            logger.info(
                "[Agent] save: %s | outcome=duplicate | similar_to=%s | sim=%.3f",
                name,
                result.similar_to,
                result.similarity or 0.0,
            )
            logger.info("[Agent] metric: record_success(%s)", result.similar_to)
            skill_manager.record_success(result.similar_to)

    @staticmethod
    def _unique_skill_name(base: str, skill_manager) -> str:
        if not skill_manager.exists(base):
            return base

        version = 2
        while skill_manager.exists(f"{base}_v{version}"):
            version += 1
        return f"{base}_v{version}"

    @staticmethod
    def _retrieved_skill_dicts(candidates) -> list[dict[str, str]]:
        return [
            {
                "name": candidate.skill.name,
                "description": candidate.skill.description,
                "code": candidate.skill.code,
            }
            for candidate in candidates
        ]
