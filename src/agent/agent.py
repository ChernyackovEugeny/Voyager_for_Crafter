from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from agent.executor import Executor, InterruptReason
from agent.memory import SpatialMemory
from analytics.log_utils import log_episode, log_llm_call_ok
from skills.runner import SkillLoadError, SkillRuntime, load_skill

logger = logging.getLogger(__name__)


class Agent:
    """Coordinates curriculum, skill retrieval/generation, and execution."""

    def __init__(
        self,
        *,
        env,
        curriculum,
        skill_manager,
        strategy,
        executor: Executor,
        memory: SpatialMemory,
        max_iterations_per_episode: int,
        run_logger=None,
    ) -> None:
        self.env = env
        self.curriculum = curriculum
        self.skill_manager = skill_manager
        self.strategy = strategy
        self.executor = executor
        self.memory = memory
        self.max_iterations_per_episode = max_iterations_per_episode
        self.run_logger = run_logger
        self.current_task = None
        self.current_skill = None
        self._last_result = None
        self._episode_num = 1

    def run(self, *, episode_num: int = 1) -> dict[str, Any]:
        """Run one episode through task proposal, skill selection, and execution."""
        logger.info("[Agent] episode started")
        self._episode_num = episode_num
        started_at = datetime.now(timezone.utc)
        self.memory.reset()
        state, done = self._initial_state()
        iterations = 0
        episode_steps = 0
        episode_reward = 0.0
        skipped_task_keys: set[str] = set()

        while not done and iterations < self.max_iterations_per_episode:
            task = self.curriculum.propose_task(
                state["info"],
                skip=skipped_task_keys,
            )
            if task is None:
                logger.info("[Agent] episode stopped: no available tasks")
                break

            self.current_task = task
            logger.info("[Agent] task: %s | %s", task.name, task.description)
            done, state, skipped = self.step(state, task)
            if self._last_result is not None:
                episode_steps += self._last_result.steps
                episode_reward += self._last_result.total_reward
            if skipped and task.achievement_key is not None:
                skipped_task_keys.add(task.achievement_key)
            iterations += 1

        if done:
            logger.info("[Agent] episode stopped: environment done")
        elif iterations >= self.max_iterations_per_episode:
            logger.info(
                "[Agent] episode stopped: max_iterations reached | iterations=%d",
                iterations,
            )

        achievements = self._unlocked_achievements(state["info"])
        ended_at = datetime.now(timezone.utc)
        episode_id = log_episode(
            self.run_logger,
            episode_num=episode_num,
            achievements=state["info"].get("achievements", {}),
            steps=episode_steps,
            total_reward=episode_reward,
            started_at=started_at,
            ended_at=ended_at,
        )
        logger.info(
            "[Agent] summary: iterations=%d | steps=%d | reward=%.3f | achievements=%s",
            iterations,
            episode_steps,
            episode_reward,
            ", ".join(achievements) if achievements else "none",
        )
        return {
            "iterations": iterations,
            "final_state": state,
            "episode_done": done,
            "all_tasks_done": self.curriculum.propose_task(state["info"]) is None,
            "skipped_tasks": sorted(skipped_task_keys),
            "steps": episode_steps,
            "total_reward": episode_reward,
            "episode_id": episode_id,
        }

    def step(
        self,
        state: dict[str, Any],
        task,
    ) -> tuple[bool, dict[str, Any], bool]:
        """Run one task attempt and return (episode_done, final_state, skipped)."""
        candidates = self.skill_manager.retrieve(task.description)
        self._last_result = None
        self._log_retrieval(candidates)
        source = self.strategy.acquire_skill(
            task=task,
            obs=state["obs"],
            info=state["info"],
            candidates=candidates,
        )
        if source is None:
            self.strategy.on_skill_unavailable(
                task=task,
                state=state,
                curriculum=self.curriculum,
            )
            return False, state, True
        self._log_source_llm_call(source)

        try:
            function_name, skill_func = load_skill(
                source.code,
                runtime=SkillRuntime(memory=self.memory),
            )
        except SkillLoadError as exc:
            logger.warning("[Agent] load failed: %s | %s", task.name, exc)
            failure_state = dict(state)
            failure_state["failure_reason"] = "load_error"
            failure_state["error_traceback"] = str(exc)
            call = self.strategy.on_task_failed(
                task=task,
                source=source,
                state=failure_state,
                skill_manager=self.skill_manager,
                curriculum=self.curriculum,
            )
            self._log_llm_call(call, call_type="reflection")
            return False, state, False

        self.current_skill = source.reused_name or function_name
        if source.generated:
            logger.info("[Agent] codegen: generated function %s", function_name)
        logger.info("[Agent] execute: %s", self.current_skill)
        result = self.executor.run(skill_func, self.env, state)
        self._last_result = result
        self._log_execution_result(result)
        final_state = result.final_state
        task_complete = self.curriculum.is_task_complete(
            task,
            final_state["info"],
        )

        if task_complete:
            self.strategy.on_task_completed(
                task=task,
                source=source,
                skill_manager=self.skill_manager,
            )
        else:
            failure_state = dict(final_state)
            failure_state["failure_reason"] = (
                "task_incomplete"
                if result.reason == InterruptReason.COMPLETED
                else result.reason.value
            )
            failure_state["error_traceback"] = result.error
            call = self.strategy.on_task_failed(
                task=task,
                source=source,
                state=failure_state,
                skill_manager=self.skill_manager,
                curriculum=self.curriculum,
            )
            self._log_llm_call(call, call_type="reflection")

        return result.reason == InterruptReason.EPISODE_DONE, final_state, False

    def _initial_state(self) -> tuple[dict[str, Any], bool]:
        obs = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(0)
        self.executor.render_state(self.env)
        return {"obs": obs, "info": info}, bool(terminated or truncated)

    def _log_retrieval(self, candidates) -> None:
        if not candidates:
            logger.info("[Agent] retrieve: 0 candidates")
            return
        best = candidates[0]
        logger.info(
            "[Agent] retrieve: best=%s sim=%.3f threshold=%.3f -> %s",
            best.skill.name,
            best.similarity,
            self.strategy.reuse_threshold,
            self.strategy.retrieval_route(candidates),
        )

    @staticmethod
    def _log_execution_result(result) -> None:
        gained = ", ".join(result.achievements_gained)
        if not gained:
            gained = "none"
        logger.info(
            "[Executor] result: %s | steps=%d | reward=%.3f | gained=%s",
            result.reason.value,
            result.steps,
            result.total_reward,
            gained,
        )
        if result.error:
            first_line = result.error.splitlines()[0] if result.error else ""
            logger.info("[Executor] error: %s", first_line)

    def _log_source_llm_call(self, source) -> None:
        call = getattr(source, "llm_call", None)
        if call is None:
            return
        self._log_llm_call(call, call_type="codegen")

    def _log_llm_call(self, call, *, call_type: str) -> None:
        if call is None:
            return
        log_llm_call_ok(
            self.run_logger,
            call_type=call_type,
            episode_num=self._episode_num,
            model=call.model,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            prompt_cache_hit_tokens=call.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=call.prompt_cache_miss_tokens,
            reasoning_tokens=call.reasoning_tokens,
            cost_usd=call.cost_usd,
            latency_ms=call.latency_ms,
            prompt_template_id=call.prompt_template_id,
            prompt_hash=call.prompt_hash,
        )

    @staticmethod
    def _unlocked_achievements(info: dict[str, Any]) -> list[str]:
        return sorted(
            key
            for key, value in info.get("achievements", {}).items()
            if value
        )
