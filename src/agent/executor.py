"""Step-by-step driver for skill generators against the Crafter env."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import traceback as tb_mod
from typing import Any, Callable

from environment.render_viewer import RenderViewer

logger = logging.getLogger(__name__)


class InterruptReason(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    HEALTH_LOW = "health_low"
    EPISODE_DONE = "episode_done"
    ERROR = "error"


@dataclass
class ExecutionResult:
    reason: InterruptReason
    steps: int
    total_reward: float
    final_state: dict[str, Any]
    achievements_gained: list[str] = field(default_factory=list)
    error: str | None = None


class Executor:
    """Runs one generated skill until completion or a hard interrupt."""

    def __init__(
        self,
        *,
        max_steps_per_skill: int,
        health_threshold: int,
        render: bool = False,
        render_size: int = 512,
        render_delay_s: float = 0.05,
        render_viewer=None,
    ) -> None:
        self._max_steps = max_steps_per_skill
        self._health_threshold = health_threshold
        self._render = render
        self._render_size = render_size
        self._render_delay_s = render_delay_s
        self._render_viewer = render_viewer
        self._render_warning_logged = False

    def run(
        self,
        skill: Callable[[dict[str, Any]], Any],
        env,
        initial_state: dict[str, Any],
    ) -> ExecutionResult:
        obs = initial_state.get("obs")
        info = dict(initial_state.get("info", {}))
        state = {"obs": obs, "info": info}
        achievements_before = self._unlocked(info)
        steps = 0
        total_reward = 0.0

        try:
            gen = skill(state)
            action = next(gen)
            while True:
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                steps += 1
                total_reward += float(reward)
                info = dict(info)
                state = {"obs": obs, "info": info}
                self._render_frame(env)

                if done:
                    return self._make_result(
                        InterruptReason.EPISODE_DONE,
                        steps,
                        total_reward,
                        obs,
                        info,
                        achievements_before,
                    )
                if self._health(info) <= self._health_threshold:
                    logger.info("executor: health interrupt at step %d", steps)
                    return self._make_result(
                        InterruptReason.HEALTH_LOW,
                        steps,
                        total_reward,
                        obs,
                        info,
                        achievements_before,
                    )
                if steps >= self._max_steps:
                    logger.info("executor: timeout at step %d", steps)
                    return self._make_result(
                        InterruptReason.TIMEOUT,
                        steps,
                        total_reward,
                        obs,
                        info,
                        achievements_before,
                    )

                action = gen.send(state)

        except StopIteration:
            return self._make_result(
                InterruptReason.COMPLETED,
                steps,
                total_reward,
                obs,
                info,
                achievements_before,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}\n{tb_mod.format_exc()}"
            logger.warning("executor: skill error: %s", exc)
            return ExecutionResult(
                reason=InterruptReason.ERROR,
                steps=steps,
                total_reward=total_reward,
                final_state={"obs": obs, "info": info},
                achievements_gained=self._diff(
                    achievements_before, self._unlocked(info)
                ),
                error=err,
            )

    def _make_result(
        self,
        reason: InterruptReason,
        steps: int,
        total_reward: float,
        obs: Any,
        info: dict[str, Any],
        achievements_before: set[str],
    ) -> ExecutionResult:
        return ExecutionResult(
            reason=reason,
            steps=steps,
            total_reward=total_reward,
            final_state={"obs": obs, "info": info},
            achievements_gained=self._diff(
                achievements_before, self._unlocked(info)
            ),
        )

    @staticmethod
    def _unlocked(info: dict[str, Any]) -> set[str]:
        return {key for key, value in info.get("achievements", {}).items() if value}

    @staticmethod
    def _diff(before: set[str], after: set[str]) -> list[str]:
        return sorted(after - before)

    @staticmethod
    def _health(info: dict[str, Any]) -> int:
        inventory = info.get("inventory", {})
        if "health" in inventory:
            return int(inventory["health"])
        return int(info.get("health", 99))

    def _render_frame(self, env) -> None:
        if not self._render:
            return
        render = getattr(env, "render", None)
        if render is None:
            if not self._render_warning_logged:
                logger.warning("executor: render requested but env has no render()")
                self._render_warning_logged = True
            return
        frame = render(size=self._render_size)
        self._display_frame(frame)
        if self._render_delay_s > 0:
            time.sleep(self._render_delay_s)

    def render_state(self, env) -> None:
        """Render the current environment state outside of the step loop."""
        self._render_frame(env)

    def _display_frame(self, frame) -> None:
        if frame is None:
            return
        if self._render_viewer is None:
            self._render_viewer = RenderViewer()
        self._render_viewer.show(frame)
