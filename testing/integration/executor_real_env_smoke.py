import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent.executor import Executor, InterruptReason
from environment.wrapper import CrafterEnv
from skills.runner import load_skill


class TestExecutorRealEnvSmoke(unittest.TestCase):
    def test_trivial_skill_runs_against_crafter(self):
        env = CrafterEnv()
        try:
            obs = env.reset()
            obs, _, terminated, truncated, info = env.step(0)
            self.assertFalse(terminated or truncated)

            src = (
                "def noop_skill(state):\n"
                "    for _ in range(5):\n"
                "        state = yield 0\n"
            )
            _, func = load_skill(src)
            executor = Executor(max_steps_per_skill=10, health_threshold=4)
            result = executor.run(func, env, {"obs": obs, "info": info})

            self.assertEqual(result.reason, InterruptReason.COMPLETED)
            self.assertEqual(result.steps, 5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
