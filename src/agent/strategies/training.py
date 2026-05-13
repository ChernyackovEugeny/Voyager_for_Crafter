from __future__ import annotations

import logging

from agent.strategies import AgentStrategy, SkillSource
from environment.captioner import caption
from llm.reflection import FailureContext

logger = logging.getLogger(__name__)


class TrainingStrategy(AgentStrategy):
    """Training mode: reuse known skills, generate missing skills, and learn."""

    name = "train"

    _REFLECTABLE_REASONS = {
        "timeout",
        "health_low",
        "danger_visible",
        "stagnation",
        "task_incomplete",
    }

    def __init__(
        self,
        *,
        codegen,
        bug_fixer=None,
        reuse_threshold: float,
        reflection=None,
        reflection_enabled: bool = False,
        max_reflections_per_skill: int = 3,
        max_fix_attempts: int = 3,
        skill_validator=None,
        min_reflection_steps: int = 3,
    ):
        self._codegen = codegen
        self._bug_fixer = bug_fixer
        self._reuse_threshold = reuse_threshold
        self._reflection = reflection
        self._reflection_enabled = reflection_enabled
        self._max_reflections_per_skill = max_reflections_per_skill
        self._max_fix_attempts = max_fix_attempts
        self._skill_validator = skill_validator
        self._min_reflection_steps = min_reflection_steps
        self._pending_llm_calls: list[tuple[str, object]] = []
        self._last_codegen_failure: dict[str, tuple[str, str]] = {}

    def acquire_skill(
        self,
        *,
        task,
        obs,
        info: dict,
        candidates,
    ):
        self._pending_llm_calls = []
        selected = self._select_reusable_skill(candidates)
        if selected is not None:
            logger.info("[Agent] reuse: %s", selected.skill.name)
            return SkillSource(
                code=selected.skill.code,
                reused_name=selected.skill.name,
                generated=False,
            )

        logger.info("[Agent] codegen: generating new skill for %s", task.name)
        state_text = caption(obs, info)
        previous_failure = self._last_codegen_failure.get(task.name)
        try:
            call = self._codegen.get_code(
                state_text=state_text,
                task=task.description,
                retrieved_skills=self._retrieved_skill_dicts(candidates),
                previous_failure=previous_failure,
            )
        except Exception as exc:
            logger.warning("strategy: codegen failed for %s: %s", task.name, exc)
            return None
        return self._compile_generated_skill(
            code=call.code,
            task=task,
            state_text=state_text,
            llm_calls=[("codegen", call)],
        )

    def on_skill_unavailable(self, *, task, state: dict, curriculum) -> None:
        logger.info("[Agent] task failed: %s", task.name)
        curriculum.record_task_failed(task, state)

    def drain_pending_llm_calls(self) -> tuple[tuple[str, object], ...]:
        calls = tuple(self._pending_llm_calls)
        self._pending_llm_calls = []
        return calls

    def on_task_completed(self, *, task, source: SkillSource, skill_manager) -> None:
        logger.info("[Agent] task success: %s", task.name)
        self._last_codegen_failure.pop(task.name, None)
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
            return self._reflect_reused_skill(
                task=task,
                source=source,
                state=state,
                skill_manager=skill_manager,
            )

        logger.info("[Agent] discard: generated skill was not successful")
        logger.info("[Agent] failed generated skill code:\n%s", source.code)
        self._last_codegen_failure[task.name] = (
            source.code,
            str(state.get("failure_reason") or "unknown"),
        )
        return None

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

    def _compile_generated_skill(
        self,
        *,
        code: str,
        task,
        state_text: str,
        llm_calls: list[tuple[str, object]],
    ):
        if self._skill_validator is None:
            call_type, call = llm_calls[-1]
            return SkillSource(
                code=code,
                generated=True,
                llm_call=call,
                llm_call_type=call_type,
                llm_calls=tuple(llm_calls),
            )

        for attempt in range(self._max_fix_attempts + 1):
            try:
                self._skill_validator(code)
                if attempt > 0:
                    logger.info(
                        "[Agent] fix_bug: succeeded after %d attempt(s)",
                        attempt,
                    )
                call_type, call = llm_calls[-1]
                return SkillSource(
                    code=code,
                    generated=True,
                    llm_call=call,
                    llm_call_type=call_type,
                    llm_calls=tuple(llm_calls),
                )
            except Exception as exc:
                if attempt >= self._max_fix_attempts:
                    logger.warning(
                        "[Agent] fix_bug: exhausted after %d attempt(s): %s",
                        self._max_fix_attempts,
                        exc,
                    )
                    self._pending_llm_calls = list(llm_calls)
                    return None

                logger.info(
                    "[Agent] fix_bug: attempt %d/%d for %s: %s",
                    attempt + 1,
                    self._max_fix_attempts,
                    task.name,
                    exc,
                )
                if self._bug_fixer is None:
                    logger.warning("[Agent] fix_bug: no bug_fixer configured")
                    self._pending_llm_calls = list(llm_calls)
                    return None

                try:
                    fixed_call = self._bug_fixer.fix(
                        skill_code=code,
                        error_traceback=str(exc),
                        state_text=state_text,
                        task=task.description,
                    )
                except Exception as fix_exc:
                    logger.warning(
                        "[Agent] fix_bug: API call failed for %s: %s",
                        task.name,
                        fix_exc,
                    )
                    self._pending_llm_calls = list(llm_calls)
                    return None

                fixed_code = fixed_call.code
                llm_calls.append(("fix_bug", fixed_call))
                if fixed_code.strip() == code.strip():
                    logger.warning(
                        "[Agent] fix_bug: returned identical code for %s",
                        task.name,
                    )
                    self._pending_llm_calls = list(llm_calls)
                    return None
                code = fixed_code

        return None

    def _reflect_reused_skill(self, *, task, source, state, skill_manager):
        if not self._reflection_enabled or self._reflection is None:
            return None

        failure_reason = str(state.get("failure_reason") or "unknown")
        if failure_reason not in self._REFLECTABLE_REASONS:
            logger.info(
                "[Agent] reflection skipped: reason=%s skill=%s",
                failure_reason,
                source.reused_name,
            )
            return None

        skill = skill_manager.get(source.reused_name)
        if skill is None:
            logger.info("[Agent] reflection skipped: skill missing")
            return None
        if skill.success_count <= 0:
            logger.info("[Agent] reflection skipped: skill has no prior success")
            return None
        if skill.reflected_count >= self._max_reflections_per_skill:
            logger.info(
                "[Agent] reflection skipped: max reflections reached for %s",
                skill.name,
            )
            return None
        if self._missing_prerequisites(task, state):
            logger.info(
                "[Agent] reflection skipped: missing prerequisites for %s",
                task.name,
            )
            return None
        steps = int(state.get("executor_steps", 0) or 0)
        if steps < self._min_reflection_steps and failure_reason != "task_incomplete":
            logger.info(
                "[Agent] reflection skipped: only %d step(s) before interrupt",
                steps,
            )
            return None

        logger.info("[Agent] reflection: improving %s", skill.name)
        try:
            call = self._reflection.improve_skill(
                FailureContext(
                    task_description=task.description,
                    failure_reason=failure_reason,
                    skill_code=source.code,
                    state_snapshot=state,
                    error_traceback=state.get("error_traceback"),
                )
            )
        except Exception as exc:
            logger.warning("[Agent] reflection failed for %s: %s", skill.name, exc)
            return None

        skill_manager.update_code(skill.name, call.code)
        logger.info("[Agent] reflection: updated %s", skill.name)
        return call

    @staticmethod
    def _missing_prerequisites(task, state: dict) -> bool:
        key = getattr(task, "achievement_key", None)
        if key is None:
            return False
        info = state.get("info", {})
        achievements = info.get("achievements", {}) or {}
        inventory = info.get("inventory", {}) or {}

        required_achievements = {
            "make_wood_pickaxe": {"collect_wood", "place_table"},
            "collect_stone": {"make_wood_pickaxe"},
            "collect_coal": {"make_wood_pickaxe"},
            "make_stone_pickaxe": {"collect_stone", "place_table"},
            "place_furnace": {"collect_stone"},
            "collect_iron": {"make_stone_pickaxe"},
            "make_iron_pickaxe": {"collect_iron", "collect_coal", "place_furnace"},
            "collect_diamond": {"make_iron_pickaxe"},
            "make_wood_sword": {"collect_wood", "place_table"},
            "make_stone_sword": {"collect_stone", "place_table"},
            "make_iron_sword": {"collect_iron", "collect_coal", "place_furnace"},
            "defeat_zombie": {"make_wood_sword"},
            "defeat_skeleton": {"make_wood_sword"},
            "eat_plant": {"place_plant"},
        }
        missing_achievements = [
            req for req in required_achievements.get(key, set())
            if not achievements.get(req)
        ]
        if missing_achievements:
            return True

        required_inventory = {
            "place_furnace": {"stone": 4},
            "place_plant": {"sapling": 1},
        }
        for item, count in required_inventory.get(key, {}).items():
            if int(inventory.get(item, 0) or 0) < count:
                return True
        return False

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
