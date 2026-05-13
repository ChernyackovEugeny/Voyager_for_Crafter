import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.executor import Executor, InterruptReason
from environment.ids import NAME_TO_ID


class FakeEnv:
    """Pre-scripted env stub matching environment.wrapper.CrafterEnv.step()."""

    def __init__(self, plan):
        self._plan = plan
        self._i = 0
        self.actions = []
        self.render_calls = []

    def step(self, action):
        self.actions.append(action)
        idx = min(self._i, len(self._plan) - 1)
        self._i += 1
        obs, reward, terminated, truncated, info = self._plan[idx]
        return obs, reward, terminated, truncated, info

    def render(self, size=None):
        self.render_calls.append(size)
        return np.zeros((4, 4, 3), dtype=np.uint8)


class FakeViewer:
    def __init__(self):
        self.frames = []

    def show(self, frame):
        self.frames.append(frame)


def state(
    health=9,
    achievements=None,
    obs="obs",
    player_pos=(0, 0),
    inventory=None,
):
    inv = {"health": health}
    if inventory:
        inv.update(inventory)
    return {
        "obs": obs,
        "info": {
            "inventory": inv,
            "achievements": achievements or {},
            "player_pos": player_pos,
        },
    }


def state_with_visible(name, player_pos=(10, 10)):
    current = state(player_pos=player_pos)
    semantic = np.zeros((64, 64), dtype=int)
    semantic[player_pos[0] + 1, player_pos[1]] = NAME_TO_ID[name]
    current["info"]["semantic"] = semantic
    current["info"]["view_size"] = (9, 9)
    return current


class TestExecutor(unittest.TestCase):
    def test_completed_when_generator_returns(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            for _ in range(3):
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.COMPLETED)
        self.assertEqual(result.steps, 3)

    def test_timeout(self):
        ex = Executor(max_steps_per_skill=5, health_threshold=4)
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)
        self.assertEqual(result.steps, 5)

    def test_health_low_at_threshold_four(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        env = FakeEnv([
            ("obs", 0.0, False, False, state(health=5)["info"]),
            ("obs", 0.0, False, False, state(health=4)["info"]),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.HEALTH_LOW)
        self.assertEqual(result.steps, 2)

    def test_health_five_is_allowed(self):
        ex = Executor(max_steps_per_skill=2, health_threshold=4)
        env = FakeEnv([("obs", 0.0, False, False, state(health=5)["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_episode_done(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        env = FakeEnv([("obs", 0.0, True, False, state()["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.EPISODE_DONE)

    def test_error_captured(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            yield 0
            raise ValueError("boom")

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.ERROR)
        self.assertIn("ValueError", result.error)
        self.assertIn("boom", result.error)

    def test_achievements_gained_diff(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        plan = [
            ("obs", 0.0, False, False, state(achievements={"collect_wood": 0})["info"]),
            ("obs", 1.0, True, False, state(achievements={"collect_wood": 1})["info"]),
        ]
        env = FakeEnv(plan)

        def skill(current):
            current = yield 0
            current = yield 0

        result = ex.run(skill, env, state(achievements={"collect_wood": 0}))
        self.assertEqual(result.achievements_gained, ["collect_wood"])
        self.assertEqual(result.total_reward, 1.0)

    def test_state_shape_sent_back_to_skill(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        env = FakeEnv([("new_obs", 0.0, False, False, state()["info"])])
        seen = []

        def skill(current):
            seen.append(sorted(current.keys()))
            current = yield 0
            seen.append(sorted(current.keys()))

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.COMPLETED)
        self.assertEqual(seen, [["info", "obs"], ["info", "obs"]])
        self.assertEqual(result.final_state["obs"], "new_obs")
        self.assertEqual(result.final_state["info"]["inventory"]["health"], 9)

    def test_render_disabled_by_default(self):
        ex = Executor(max_steps_per_skill=2, health_threshold=4)
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            current = yield 0

        ex.run(skill, env, state())
        self.assertEqual(env.render_calls, [])

    def test_render_enabled_draws_after_each_environment_step(self):
        viewer = FakeViewer()
        ex = Executor(
            max_steps_per_skill=10,
            health_threshold=4,
            render=True,
            render_size=320,
            render_delay_s=0,
            render_viewer=viewer,
        )
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            current = yield 0
            current = yield 0

        ex.run(skill, env, state())
        self.assertEqual(env.render_calls, [320, 320])
        self.assertEqual(len(viewer.frames), 2)

    def test_render_enabled_without_env_render_does_not_raise(self):
        class NoRenderEnv:
            def step(self, action):
                return "obs", 0.0, False, False, state()["info"]

        ex = Executor(
            max_steps_per_skill=2,
            health_threshold=4,
            render=True,
            render_delay_s=0,
        )

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, NoRenderEnv(), state())
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_stagnation_interrupts_noop_loop(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            stagnation_window=3,
        )
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.STAGNATION)
        self.assertEqual(result.steps, 3)

    def test_stagnation_disabled_with_zero_window(self):
        ex = Executor(
            max_steps_per_skill=3,
            health_threshold=4,
            stagnation_window=0,
        )
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_stagnation_warmup_delays_interrupt(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            stagnation_window=2,
            min_steps_before_stagnation_interrupt=5,
        )
        env = FakeEnv([("obs", 0.0, False, False, state()["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.STAGNATION)
        self.assertEqual(result.steps, 5)

    def test_movement_prevents_stagnation(self):
        ex = Executor(
            max_steps_per_skill=3,
            health_threshold=4,
            stagnation_window=2,
        )
        env = FakeEnv([
            ("obs", 0.0, False, False, state(player_pos=(1, 0))["info"]),
            ("obs", 0.0, False, False, state(player_pos=(2, 0))["info"]),
            ("obs", 0.0, False, False, state(player_pos=(3, 0))["info"]),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(player_pos=(0, 0)))
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_item_inventory_growth_prevents_stagnation(self):
        ex = Executor(
            max_steps_per_skill=3,
            health_threshold=4,
            stagnation_window=2,
        )
        env = FakeEnv([
            ("obs", 0.0, False, False, state(inventory={"wood": 1})["info"]),
            ("obs", 0.0, False, False, state(inventory={"wood": 2})["info"]),
            ("obs", 0.0, False, False, state(inventory={"wood": 3})["info"]),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(inventory={"wood": 0}))
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_achievement_progress_prevents_stagnation(self):
        ex = Executor(
            max_steps_per_skill=2,
            health_threshold=4,
            stagnation_window=1,
        )
        env = FakeEnv([
            (
                "obs",
                0.0,
                False,
                False,
                state(achievements={"collect_wood": 1})["info"],
            ),
            (
                "obs",
                0.0,
                False,
                False,
                state(achievements={"collect_wood": 1, "place_table": 1})["info"],
            ),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(achievements={"collect_wood": 0}))
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_stat_decay_does_not_prevent_stagnation(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            stagnation_window=2,
        )
        env = FakeEnv([
            (
                "obs",
                0.0,
                False,
                False,
                state(inventory={"food": 8, "drink": 8})["info"],
            ),
            (
                "obs",
                0.0,
                False,
                False,
                state(inventory={"food": 7, "drink": 7})["info"],
            ),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(inventory={"food": 9, "drink": 9}))
        self.assertEqual(result.reason, InterruptReason.STAGNATION)

    def test_survival_stats_can_count_as_progress_for_survive(self):
        ex = Executor(
            max_steps_per_skill=3,
            health_threshold=4,
            stagnation_window=1,
        )
        env = FakeEnv([
            ("obs", 0.0, False, False, state(inventory={"energy": 2})["info"]),
            ("obs", 0.0, False, False, state(inventory={"energy": 3})["info"]),
            ("obs", 0.0, False, False, state(inventory={"energy": 4})["info"]),
        ])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(inventory={"energy": 1}), survival_progress_enabled=True)
        self.assertEqual(result.reason, InterruptReason.TIMEOUT)

    def test_danger_interrupts_when_enabled(self):
        ex = Executor(max_steps_per_skill=50, health_threshold=4)
        current = state_with_visible("zombie")
        env = FakeEnv([("obs", 0.0, False, False, current["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(), danger_interrupt_enabled=True)
        self.assertEqual(result.reason, InterruptReason.DANGER_VISIBLE)

    def test_health_warmup_lets_skill_act_before_interrupt(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            min_steps_before_health_interrupt=3,
        )
        env = FakeEnv([("obs", 0.0, False, False, state(health=2)["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(health=2))
        self.assertEqual(result.reason, InterruptReason.HEALTH_LOW)
        self.assertEqual(result.steps, 3)

    def test_health_warmup_zero_keeps_step_one_interrupt(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            min_steps_before_health_interrupt=0,
        )
        env = FakeEnv([("obs", 0.0, False, False, state(health=2)["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state(health=2))
        self.assertEqual(result.reason, InterruptReason.HEALTH_LOW)
        self.assertEqual(result.steps, 1)

    def test_health_low_has_priority_over_stagnation(self):
        ex = Executor(
            max_steps_per_skill=50,
            health_threshold=4,
            stagnation_window=1,
        )
        env = FakeEnv([("obs", 0.0, False, False, state(health=4)["info"])])

        def skill(current):
            while True:
                current = yield 0

        result = ex.run(skill, env, state())
        self.assertEqual(result.reason, InterruptReason.HEALTH_LOW)


if __name__ == "__main__":
    unittest.main()
