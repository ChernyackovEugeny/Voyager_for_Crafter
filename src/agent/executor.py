"""Step-by-step driver for skill generators against the Crafter env."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import traceback as tb_mod
from typing import Any, Callable

from environment.danger import hostile_visible
from environment.render_viewer import RenderViewer

logger = logging.getLogger(__name__)


class InterruptReason(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    HEALTH_LOW = "health_low"
    DANGER_VISIBLE = "danger_visible"
    STAGNATION = "stagnation"
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
        stagnation_window: int = 0,
        min_steps_before_stagnation_interrupt: int = 0,
        min_steps_before_health_interrupt: int = 0,
    ) -> None:
        self._max_steps = max_steps_per_skill
        self._health_threshold = health_threshold
        self._render = render
        self._render_size = render_size
        self._render_delay_s = render_delay_s
        self._render_viewer = render_viewer
        self._render_warning_logged = False
        self._stagnation_window = stagnation_window
        self._min_steps_before_stagnation_interrupt = (
            min_steps_before_stagnation_interrupt
        )
        self._min_steps_before_health_interrupt = min_steps_before_health_interrupt

    def run(
        self,
        skill: Callable[[dict[str, Any]], Any],
        env,
        initial_state: dict[str, Any],
        *,
        health_interrupt_enabled: bool = True,
        danger_interrupt_enabled: bool = False,
        survival_progress_enabled: bool = False,
    ) -> ExecutionResult:
        obs = initial_state.get("obs")
        info = dict(initial_state.get("info", {}))
        state = {"obs": obs, "info": info}
        achievements_before = self._unlocked(info)
        last_progress = self._progress_signature(
            info,
            include_survival_stats=survival_progress_enabled,
        )
        stagnant_steps = 0
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
                if danger_interrupt_enabled and hostile_visible(info):
                    logger.info("executor: danger interrupt at step %d", steps)
                    return self._make_result(
                        InterruptReason.DANGER_VISIBLE,
                        steps,
                        total_reward,
                        obs,
                        info,
                        achievements_before,
                    )
                if (
                    health_interrupt_enabled
                    and steps >= self._min_steps_before_health_interrupt
                    and self._health(info) <= self._health_threshold
                ):
                    logger.info("executor: health interrupt at step %d", steps)
                    return self._make_result(
                        InterruptReason.HEALTH_LOW,
                        steps,
                        total_reward,
                        obs,
                        info,
                        achievements_before,
                    )
                progress = self._progress_signature(
                    info,
                    include_survival_stats=survival_progress_enabled,
                )
                if progress == last_progress:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
                last_progress = progress
                if (
                    self._stagnation_window > 0
                    and steps >= self._min_steps_before_stagnation_interrupt
                    and stagnant_steps >= self._stagnation_window
                ):
                    logger.info("executor: stagnation interrupt at step %d", steps)
                    return self._make_result(
                        InterruptReason.STAGNATION,
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
            logger.warning("executor: skill error: %s", err)
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

    @classmethod
    def _progress_signature(
        cls,
        info: dict[str, Any],
        *,
        include_survival_stats: bool = False,
    ) -> tuple[Any, ...]:
        signature = (
            cls._position(info),
            cls._item_signature(info),
            tuple(sorted(cls._unlocked(info))),
        )
        if include_survival_stats:
            signature += (cls._survival_signature(info),)
        return signature

    @staticmethod
    def _position(info: dict[str, Any]) -> tuple[int, ...] | None:
        position = info.get("player_pos")
        if position is None:
            return None
        return tuple(int(value) for value in position)

    @staticmethod
    def _item_signature(info: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        stats = {"health", "food", "drink", "energy"}
        inventory = info.get("inventory", {})
        return tuple(
            sorted(
                (key, int(value))
                for key, value in inventory.items()
                if key not in stats
            )
        )

    @staticmethod
    def _survival_signature(info: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        inventory = info.get("inventory", {})
        return tuple(
            (key, int(inventory.get(key, 0) or 0))
            for key in ("health", "food", "drink", "energy")
        )

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
