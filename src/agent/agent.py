from __future__ import annotations

import logging
from typing import Any

from agent.executor import Executor, InterruptReason
from agent.memory import SpatialMemory
from environment.captioner import caption
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
        codegen,
        executor: Executor,
        memory: SpatialMemory,
        reuse_threshold: float,
        max_iterations_per_episode: int,
        run_logger=None,
    ) -> None:
        self.env = env
        self.curriculum = curriculum
        self.skill_manager = skill_manager
        self.codegen = codegen
        self.executor = executor
        self.memory = memory
        self.reuse_threshold = reuse_threshold
        self.max_iterations_per_episode = max_iterations_per_episode
        self.run_logger = run_logger
        self.current_task = None
        self.current_skill = None

    def run(self) -> dict[str, Any]:
        """Run one episode through task proposal, skill selection, and execution."""
        self.memory.reset()
        state, done = self._initial_state()
        iterations = 0

        while not done and iterations < self.max_iterations_per_episode:
            task = self.curriculum.propose_task(state["info"])
            if task is None:
                break

            self.current_task = task
            done, state = self.step(state, task)
            iterations += 1

        return {
            "iterations": iterations,
            "final_state": state,
            "episode_done": done,
            "all_tasks_done": self.curriculum.propose_task(state["info"]) is None,
        }

    def step(self, state: dict[str, Any], task) -> tuple[bool, dict[str, Any]]:
        """Run one task attempt and return (episode_done, final_state)."""
        candidates = self.skill_manager.retrieve(task.description)
        selected = self._select_reusable_skill(candidates)
        reused_name: str | None = None

        if selected is None:
            code = self._generate_skill(task, state, candidates)
            if code is None:
                self.curriculum.record_task_failed(task, state)
                return False, state
        else:
            code = selected.skill.code
            reused_name = selected.skill.name

        try:
            function_name, skill_func = load_skill(
                code,
                runtime=SkillRuntime(memory=self.memory),
            )
        except SkillLoadError as exc:
            logger.warning("agent: failed to load skill for %s: %s", task.name, exc)
            self.curriculum.record_task_failed(task, state)
            if reused_name is not None:
                self.skill_manager.record_failure(reused_name)
            return False, state

        self.current_skill = reused_name or function_name
        result = self.executor.run(skill_func, self.env, state)
        final_state = result.final_state
        task_complete = self.curriculum.is_task_complete(
            task,
            final_state["info"],
        )

        self._record_outcome(
            task=task,
            code=code,
            reused_name=reused_name,
            task_complete=task_complete,
            state=final_state,
        )

        return result.reason == InterruptReason.EPISODE_DONE, final_state

    def _initial_state(self) -> tuple[dict[str, Any], bool]:
        obs = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(0)
        return {"obs": obs, "info": info}, bool(terminated or truncated)

    def _select_reusable_skill(self, candidates):
        if not candidates:
            return None
        best = candidates[0]
        if best.similarity >= self.reuse_threshold:
            return best
        return None

    def _generate_skill(self, task, state: dict[str, Any], candidates) -> str | None:
        try:
            return self.codegen.get_code(
                state_text=caption(state["obs"], state["info"]),
                task=task.description,
                retrieved_skills=self._retrieved_skill_dicts(candidates),
            )
        except Exception as exc:
            logger.warning("agent: codegen failed for %s: %s", task.name, exc)
            return None

    def _record_outcome(
        self,
        *,
        task,
        code: str,
        reused_name: str | None,
        task_complete: bool,
        state: dict[str, Any],
    ) -> None:
        if task_complete:
            if reused_name is not None:
                self.skill_manager.record_success(reused_name)
            else:
                self._save_new_skill(task, code)
            return

        self.curriculum.record_task_failed(task, state)
        if reused_name is not None:
            self.skill_manager.record_failure(reused_name)

    def _save_new_skill(self, task, code: str) -> None:
        result = self.skill_manager.save(
            name=self._unique_skill_name(task.name.replace("-", "_")),
            code=code,
            task=task.description,
        )
        if result.outcome == "duplicate" and result.similar_to is not None:
            self.skill_manager.record_success(result.similar_to)

    def _unique_skill_name(self, base: str) -> str:
        if not self.skill_manager.exists(base):
            return base

        version = 2
        while self.skill_manager.exists(f"{base}_v{version}"):
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
