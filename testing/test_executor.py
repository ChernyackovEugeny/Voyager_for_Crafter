import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.executor import Executor, InterruptReason


class FakeEnv:
    """Pre-scripted env stub matching environment.wrapper.CrafterEnv.step()."""

    def __init__(self, plan):
        self._plan = plan
        self._i = 0
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        idx = min(self._i, len(self._plan) - 1)
        self._i += 1
        obs, reward, terminated, truncated, info = self._plan[idx]
        return obs, reward, terminated, truncated, info


def state(health=9, achievements=None, obs="obs"):
    return {
        "obs": obs,
        "info": {
            "inventory": {"health": health},
            "achievements": achievements or {},
        },
    }


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


if __name__ == "__main__":
    unittest.main()
