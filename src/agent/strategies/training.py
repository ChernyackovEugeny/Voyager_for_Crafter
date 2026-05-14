from __future__ import annotations

import logging

from agent.bootstrap import BOOTSTRAP_CODE
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
        "episode_done",
        "stagnation",
        "task_incomplete",
    }
    _SURVIVAL_PARTIAL_SAVE_STEPS = 80

    def __init__(
        self,
        *,
        codegen,
        bug_fixer=None,
        reuse_threshold: float,
        reflection=None,
        reflection_enabled: bool = False,
        max_fix_attempts: int = 3,
        skill_validator=None,
        min_reflection_steps: int = 3,
        protected_reflection_skill_names: set[str] | frozenset[str] = frozenset(),
    ):
        self._codegen = codegen
        self._bug_fixer = bug_fixer
        self._reuse_threshold = reuse_threshold
        self._reflection = reflection
        self._reflection_enabled = reflection_enabled
        self._max_fix_attempts = max_fix_attempts
        self._skill_validator = skill_validator
        self._min_reflection_steps = min_reflection_steps
        self._protected_reflection_skill_names = frozenset(
            protected_reflection_skill_names
        )
        self._pending_llm_calls: list[tuple[str, object]] = []
        self._last_codegen_failure: dict[str, tuple[str, str]] = {}
        self._pending_fixed_sources: dict[str, SkillSource] = {}
        self._runtime_fix_attempts: dict[tuple[str, str], int] = {}

    def acquire_skill(
        self,
        *,
        task,
        obs,
        info: dict,
        candidates,
    ):
        self._pending_llm_calls = []
        pending_fixed = self._pending_fixed_sources.pop(task.name, None)
        if pending_fixed is not None:
            logger.info("[Agent] retry fixed generated skill for %s", task.name)
            return pending_fixed

        extra_skills = self._with_bootstrap_fallbacks(tuple(
            (candidate.skill.name, candidate.skill.code)
            for candidate in candidates
        ))
        allowed_skill_names = frozenset(
            candidate.skill.name for candidate in candidates
        )
        selected = self._select_reusable_skill(task, candidates)
        if selected is not None:
            logger.info("[Agent] reuse: %s", selected.skill.name)
            return SkillSource(
                code=selected.skill.code,
                reused_name=selected.skill.name,
                generated=False,
                extra_skills=extra_skills,
                allowed_skill_names=allowed_skill_names,
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
            allowed_skill_names=allowed_skill_names,
            extra_skills=extra_skills,
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
            if task.name == "survive":
                self._update_survival_score(skill_manager, source.reused_name, 1.0)
            return

        saved_name = self._save_new_skill(task, source.code, skill_manager)
        if task.name == "survive" and saved_name is not None:
            self._update_survival_score(skill_manager, saved_name, 1.0)

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
            score = self._survival_score(task=task, state=state)
            if score is not None:
                self._update_survival_score(
                    skill_manager,
                    source.reused_name,
                    score,
                )
            fixed = self._fix_reused_technical_error(
                task=task,
                source=source,
                state=state,
                skill_manager=skill_manager,
            )
            if fixed is not None:
                return fixed
            return self._reflect_reused_skill(
                task=task,
                source=source,
                state=state,
                skill_manager=skill_manager,
            )

        logger.info("[Agent] discard: generated skill was not successful")
        logger.debug("[Agent] failed generated skill code:\n%s", source.code)
        fixed = self._fix_generated_technical_error(
            task=task,
            source=source,
            state=state,
        )
        if fixed is not None:
            return fixed
        partial = self._survival_partial_progress(task=task, state=state)
        if partial is not None:
            logger.info(
                "[Agent] survival partial progress: %s | saving generated skill",
                partial,
            )
            saved_name = self._save_new_skill(task, source.code, skill_manager)
            score = self._survival_score(task=task, state=state)
            if saved_name is not None and score is not None:
                self._update_survival_score(skill_manager, saved_name, score)
        self._last_codegen_failure[task.name] = self._failure_memory(source, state)
        return None

    def retrieval_route(self, candidates, task=None) -> str:
        if self._select_reusable_skill(task, candidates) is not None:
            return "reuse"
        return "codegen"

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

    def _save_new_skill(self, task, code: str, skill_manager) -> str | None:
        base = getattr(task, "achievement_key", None) or task.name.replace("-", "_")
        name = self._unique_skill_name(base, skill_manager)
        result = skill_manager.save(name=name, code=code, task=task.description)
        if result.outcome == "ok":
            logger.info("[Agent] save: %s | outcome=ok", name)
            logger.info("[Agent] metric: record_success(%s)", name)
            skill_manager.record_success(name)
            return name
        if result.outcome == "duplicate" and result.similar_to is not None:
            if not _skill_name_matches_base(result.similar_to, base):
                logger.info(
                    "[Agent] save: %s | duplicate %s incompatible with task; "
                    "saving without dedup",
                    name,
                    result.similar_to,
                )
                retry = skill_manager.save(
                    name=name,
                    code=code,
                    task=task.description,
                    deduplicate=False,
                )
                if retry.outcome == "ok":
                    logger.info("[Agent] save: %s | outcome=ok", name)
                    logger.info("[Agent] metric: record_success(%s)", name)
                    skill_manager.record_success(name)
                    return name
                return None
            logger.info(
                "[Agent] save: %s | outcome=duplicate | similar_to=%s | sim=%.3f",
                name,
                result.similar_to,
                result.similarity or 0.0,
            )
            logger.info("[Agent] metric: record_success(%s)", result.similar_to)
            skill_manager.record_success(result.similar_to)
            return result.similar_to
        return None

    def _compile_generated_skill(
        self,
        *,
        code: str,
        task,
        state_text: str,
        llm_calls: list[tuple[str, object]],
        allowed_skill_names: set[str] | None = None,
        extra_skills: tuple[tuple[str, str], ...] = (),
    ):
        if self._skill_validator is None:
            call_type, call = llm_calls[-1]
            return SkillSource(
                code=code,
                generated=True,
                llm_call=call,
                llm_call_type=call_type,
                llm_calls=tuple(llm_calls),
                extra_skills=extra_skills,
                allowed_skill_names=frozenset(allowed_skill_names or ()),
            )

        for attempt in range(self._max_fix_attempts + 1):
            try:
                self._validate_skill(
                    code,
                    allowed_skill_names=allowed_skill_names or set(),
                    extra_skills=extra_skills,
                )
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
                    extra_skills=extra_skills,
                    allowed_skill_names=frozenset(allowed_skill_names or ()),
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

    def _validate_skill(
        self,
        code: str,
        *,
        allowed_skill_names: set[str] | frozenset[str],
        extra_skills: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if self._skill_validator is None:
            return
        try:
            self._skill_validator(
                code,
                allowed_skill_names=allowed_skill_names,
                extra_skills=extra_skills,
            )
        except TypeError:
            try:
                self._skill_validator(code, allowed_skill_names=allowed_skill_names)
            except TypeError:
                self._skill_validator(code)

    @staticmethod
    def _with_bootstrap_fallbacks(
        extra_skills: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        merged = dict(extra_skills)
        for name, code in BOOTSTRAP_CODE.items():
            merged.setdefault(name, code)
        return tuple(merged.items())

    def _fix_generated_technical_error(self, *, task, source, state):
        if not self._is_technical_error(state):
            return None
        fixed_call = self._call_bug_fixer(task=task, source=source, state=state)
        if fixed_call is None:
            self._last_codegen_failure[task.name] = self._failure_memory(source, state)
            return None
        try:
            self._validate_skill(
                fixed_call.code,
                allowed_skill_names=self._source_allowed_names(source),
                extra_skills=self._source_extra_skills(source),
            )
        except Exception as exc:
            logger.warning(
                "[Agent] runtime fix rejected for %s: %s",
                task.name,
                exc,
            )
            self._last_codegen_failure[task.name] = (
                fixed_call.code,
                f"runtime fix rejected: {exc}",
            )
            return ("fix_bug", fixed_call)

        self._pending_fixed_sources[task.name] = SkillSource(
            code=fixed_call.code,
            generated=True,
            extra_skills=self._source_extra_skills(source),
            allowed_skill_names=self._source_allowed_names(source),
        )
        self._last_codegen_failure.pop(task.name, None)
        logger.info("[Agent] runtime fix queued for retry: %s", task.name)
        return ("fix_bug", fixed_call)

    def _fix_reused_technical_error(self, *, task, source, state, skill_manager):
        if not self._is_technical_error(state):
            return None
        error_key = self._runtime_error_key(source.reused_name, state)
        if error_key is not None:
            attempts = self._runtime_fix_attempts.get(error_key, 0)
            if attempts >= self._max_fix_attempts:
                logger.info(
                    "[Agent] runtime fix skipped: repeated error limit reached "
                    "for %s: %s",
                    source.reused_name,
                    error_key[1],
                )
                return None
            self._runtime_fix_attempts[error_key] = attempts + 1
        fixed_call = self._call_bug_fixer(task=task, source=source, state=state)
        if fixed_call is None:
            return None
        if fixed_call.code.strip() == source.code.strip():
            logger.warning(
                "[Agent] runtime fix returned identical code for %s",
                source.reused_name,
            )
            return ("fix_bug", fixed_call)
        try:
            self._validate_skill(
                fixed_call.code,
                allowed_skill_names=self._source_allowed_names(source),
                extra_skills=self._source_extra_skills(source),
            )
        except Exception as exc:
            logger.warning(
                "[Agent] reused skill fix rejected for %s: %s",
                source.reused_name,
                exc,
            )
            return ("fix_bug", fixed_call)
        self._update_skill_code(
            skill_manager,
            source.reused_name,
            fixed_call.code,
            task=task.description,
            update_kind="fix",
        )
        logger.info("[Agent] runtime fix: updated %s", source.reused_name)
        return ("fix_bug", fixed_call)

    def _call_bug_fixer(self, *, task, source, state):
        if self._bug_fixer is None:
            logger.info("[Agent] runtime fix skipped: no bug_fixer configured")
            return None
        try:
            return self._bug_fixer.fix(
                skill_code=source.code,
                error_traceback=str(state.get("error_traceback") or ""),
                state_text=self._state_text(state),
                task=task.description,
            )
        except Exception as exc:
            logger.warning("[Agent] runtime fix failed for %s: %s", task.name, exc)
            return None

    @staticmethod
    def _is_technical_error(state: dict) -> bool:
        return str(state.get("failure_reason") or "") in {"error", "load_error"}

    @staticmethod
    def _failure_memory(source: SkillSource, state: dict) -> tuple[str, str]:
        reason = str(state.get("failure_reason") or "unknown")
        traceback = state.get("error_traceback")
        if traceback:
            first_line = str(traceback).splitlines()[0]
            reason = f"{reason}: {first_line}"
        return source.code, reason

    def _survival_partial_progress(self, *, task, state: dict) -> str | None:
        if getattr(task, "name", None) != "survive":
            return None
        reason = str(state.get("failure_reason") or "")
        if reason in {"error", "load_error"}:
            return None
        steps = int(state.get("executor_steps", 0) or 0)
        if steps <= 0:
            return None

        initial_inventory = (state.get("initial_info", {}) or {}).get(
            "inventory", {}
        ) or {}
        final_inventory = (state.get("info", {}) or {}).get("inventory", {}) or {}
        final_health = int(final_inventory.get("health", 0) or 0)
        if final_health <= 0:
            return None

        improved = []
        for key in ("health", "food", "drink", "energy"):
            before = int(initial_inventory.get(key, 0) or 0)
            after = int(final_inventory.get(key, 0) or 0)
            if after > before:
                improved.append(f"{key}+{after - before}")
        if improved:
            return ", ".join(improved)

        executor_reason = str(state.get("executor_reason") or reason)
        if (
            steps >= self._SURVIVAL_PARTIAL_SAVE_STEPS
            and executor_reason in {"timeout", "stagnation", "task_incomplete"}
        ):
            return f"alive_for_{steps}_steps"
        return None

    @staticmethod
    def _state_text(state: dict) -> str:
        info = state.get("info", {}) or {}
        try:
            return caption(state.get("obs"), info)
        except Exception:
            inventory = info.get("inventory", {}) or {}
            achievements = info.get("achievements", {}) or {}
            unlocked = sorted(key for key, value in achievements.items() if value)
            return (
                f"Observation: unavailable\n"
                f"Inventory: {inventory}\n"
                f"Achievements: {unlocked}"
            )

    def _survival_score(self, *, task, state: dict) -> float | None:
        if getattr(task, "name", None) != "survive":
            return None
        steps = int(state.get("executor_steps", 0) or 0)
        if steps <= 0:
            return 0.0
        initial_inventory = (state.get("initial_info", {}) or {}).get(
            "inventory", {}
        ) or {}
        final_inventory = (state.get("info", {}) or {}).get("inventory", {}) or {}
        final_health = int(final_inventory.get("health", 0) or 0)

        score = min(0.45, steps / 220.0 * 0.45)
        for key in ("health", "food", "drink", "energy"):
            before = int(initial_inventory.get(key, 0) or 0)
            after = int(final_inventory.get(key, 0) or 0)
            if after > before:
                score += min(0.12, (after - before) * 0.03)
            elif after < before:
                score -= min(0.08, (before - after) * 0.02)

        history = state.get("state_history") or []
        actions = [
            str(event.get("action") or "")
            for event in history
            if isinstance(event, dict)
        ]
        if "place_stone" in actions:
            score += 0.18
        if "place_table" in actions:
            score += 0.10
        if any(
            str(event.get("action") or "") == "do"
            and (
                event.get("emergency_hostile")
                or event.get("nearest_hostile_distance") is not None
            )
            for event in history
            if isinstance(event, dict)
        ):
            score += 0.12
        if final_health <= 0:
            score = min(score, 0.20)
        if str(state.get("failure_reason") or "") == "episode_done" and final_health <= 0:
            score -= 0.10
        return max(0.0, min(1.0, score))

    @staticmethod
    def _update_survival_score(skill_manager, name: str | None, score: float) -> None:
        if name is None:
            return
        update = getattr(skill_manager, "update_episodic_score", None)
        if update is None:
            return
        try:
            update(name, score)
        except Exception:
            logger.debug("survival score update skipped for %s", name, exc_info=True)

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
        is_survival_controller = _is_survival_controller_name(skill.name)
        if (
            skill.name in self._protected_reflection_skill_names
            and not is_survival_controller
        ):
            logger.info(
                "[Agent] reflection skipped: protected bootstrap skill %s",
                skill.name,
            )
            return None
        if skill.success_count <= 0 and not is_survival_controller:
            logger.info("[Agent] reflection skipped: skill has no prior success")
            return None
        if self._missing_prerequisites(task, state):
            logger.info(
                "[Agent] reflection skipped: missing prerequisites for %s",
                task.name,
            )
            return None
        steps = int(state.get("executor_steps", 0) or 0)
        if steps <= 0:
            logger.info(
                "[Agent] reflection skipped: no execution steps for %s",
                skill.name,
            )
            return None
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

        try:
            self._validate_skill(
                call.code,
                allowed_skill_names=self._source_allowed_names(source),
                extra_skills=self._source_extra_skills(source),
            )
        except Exception as exc:
            logger.warning(
                "[Agent] reflection rejected for %s: %s",
                skill.name,
                exc,
            )
            return call

        self._update_skill_code(
            skill_manager,
            skill.name,
            call.code,
            task=task.description,
            update_kind="reflection",
        )
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
    def _source_allowed_names(source) -> frozenset[str]:
        return frozenset(getattr(source, "allowed_skill_names", frozenset()))

    @staticmethod
    def _source_extra_skills(source) -> tuple[tuple[str, str], ...]:
        return tuple(getattr(source, "extra_skills", ()))

    @staticmethod
    def _update_skill_code(
        skill_manager,
        name: str,
        code: str,
        *,
        task: str,
        update_kind: str,
    ) -> None:
        try:
            skill_manager.update_code(
                name,
                code,
                task=task,
                update_kind=update_kind,
            )
        except TypeError:
            skill_manager.update_code(name, code)

    @staticmethod
    def _runtime_error_key(skill_name: str | None, state: dict) -> tuple[str, str] | None:
        if skill_name is None:
            return None
        traceback = str(state.get("error_traceback") or "")
        first_line = traceback.splitlines()[0] if traceback else "unknown runtime error"
        return skill_name, first_line[:200]

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


def _candidate_matches_task(task, candidate) -> bool:
    if task is None:
        return True

    skill_name = candidate.skill.name
    achievement_key = getattr(task, "achievement_key", None)
    if achievement_key is not None:
        return skill_name == achievement_key or skill_name.startswith(
            f"{achievement_key}_v"
        )

    task_name = getattr(task, "name", "")
    if task_name == "survive":
        return _is_survival_controller_name(skill_name)

    for condition in getattr(task, "completion_conditions", ()) or ():
        key = getattr(condition, "key", "")
        if isinstance(key, str) and key.startswith("achievements."):
            achievement_key = key.split(".", 1)[1]
            return _skill_name_matches_base(skill_name, achievement_key)

    base = str(task_name).replace("-", "_")
    return _skill_name_matches_base(skill_name, base)


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


def _is_survival_controller_name(skill_name: str | None) -> bool:
    if skill_name is None:
        return False
    return skill_name == "survive" or skill_name.startswith("survive_v")
