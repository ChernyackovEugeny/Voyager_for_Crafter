from __future__ import annotations

import logging
from typing import Any

from agent.executor import Executor, InterruptReason
from agent.memory import SpatialMemory
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

    def run(self) -> dict[str, Any]:
        """Run one episode through task proposal, skill selection, and execution."""
        logger.info("[Agent] episode started")
        self.memory.reset()
        state, done = self._initial_state()
        iterations = 0
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
        logger.info(
            "[Agent] summary: iterations=%d | achievements=%s",
            iterations,
            ", ".join(achievements) if achievements else "none",
        )
        return {
            "iterations": iterations,
            "final_state": state,
            "episode_done": done,
            "all_tasks_done": self.curriculum.propose_task(state["info"]) is None,
            "skipped_tasks": sorted(skipped_task_keys),
        }

    def step(
        self,
        state: dict[str, Any],
        task,
    ) -> tuple[bool, dict[str, Any], bool]:
        """Run one task attempt and return (episode_done, final_state, skipped)."""
        candidates = self.skill_manager.retrieve(task.description)
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

        try:
            function_name, skill_func = load_skill(
                source.code,
                runtime=SkillRuntime(memory=self.memory),
            )
        except SkillLoadError as exc:
            logger.warning("[Agent] load failed: %s | %s", task.name, exc)
            self.strategy.on_task_failed(
                task=task,
                source=source,
                state=state,
                skill_manager=self.skill_manager,
                curriculum=self.curriculum,
            )
            return False, state, False

        self.current_skill = source.reused_name or function_name
        if source.generated:
            logger.info("[Agent] codegen: generated function %s", function_name)
        logger.info("[Agent] execute: %s", self.current_skill)
        result = self.executor.run(skill_func, self.env, state)
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
            self.strategy.on_task_failed(
                task=task,
                source=source,
                state=final_state,
                skill_manager=self.skill_manager,
                curriculum=self.curriculum,
            )

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

    @staticmethod
    def _unlocked_achievements(info: dict[str, Any]) -> list[str]:
        return sorted(
            key
            for key, value in info.get("achievements", {}).items()
            if value
        )
