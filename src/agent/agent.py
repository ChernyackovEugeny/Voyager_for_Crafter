from agent.executor import Executor, InterruptReason
from agent.memory import SpatialMemory


class Agent:
    """Minimal episode runner. Full task/skill integration lands next."""

    def __init__(self, env, executor: Executor):
        self.env = env
        self.executor = executor
        self.current_task = None
        self.current_skill = None
        self.memory = SpatialMemory()

    def run(self):
        """Run one episode using the current stub skill selector."""
        self.memory.reset()
        state, done = self._initial_state()
        while not done:
            done, state = self.step(state)

    def step(self, state: dict) -> tuple[bool, dict]:
        """Run one selected skill and return (episode_done, final_state)."""
        skill_func = self._select_skill(state)
        result = self.executor.run(skill_func, self.env, state)
        if result.reason == InterruptReason.ERROR:
            print(f"[Agent] skill failed: {self.current_skill}")
        done = result.reason == InterruptReason.EPISODE_DONE
        return done, result.final_state

    def _initial_state(self) -> tuple[dict, bool]:
        obs = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(0)
        return {"obs": obs, "info": info}, bool(terminated or truncated)

    def _select_skill(self, state):
        """Stub: later this will retrieve or generate a task-specific skill."""
        def noop_skill(state):
            yield 0
        return noop_skill
