import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.agent import Agent
from agent.executor import Executor


class FakeEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return "reset_obs"

    def step(self, action):
        self.actions.append(action)
        info = {
            "inventory": {"health": 9},
            "achievements": {},
            "semantic": None,
            "player_pos": (0, 0),
            "view_size": (9, 7),
        }
        return "step_obs", 0.0, True, False, info


class AgentTests(unittest.TestCase):
    def test_run_bootstraps_initial_info_with_noop_step(self):
        env = FakeEnv()
        executor = Executor(max_steps_per_skill=10, health_threshold=4)
        agent = Agent(env, executor)

        agent.run()

        self.assertEqual(env.actions, [0])


if __name__ == "__main__":
    unittest.main()
