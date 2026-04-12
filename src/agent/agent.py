from agent.executor import Executor
from agent.memory import SpatialMemory


class Agent:
    def __init__(self, env):
        self.env = env
        self.executor = Executor(env)
        self.current_task = None
        self.current_skill = None
        self.memory = SpatialMemory()

    def run(self):
        """Прогон одного эпизода."""
        self.memory.reset()
        obs = self.env.reset()
        info = {}
        done = False
        while not done:
            done, obs, info = self.step(obs, info)

    def step(self, obs, info) -> tuple:
        """
        Выполняет один skill.
        Возвращает (done, obs, info) — финальное состояние после завершения skill.
        obs и info можно передать в reflection на уровне вызывающего кода.
        """
        state = {"obs": obs, "info": info}
        skill_func = self._select_skill(state)
        reason, obs, info = self.executor.run(skill_func, state)
        if reason == "error":
            print(f"[Agent] skill failed: {self.current_skill}")
        done = (reason == "done")
        return done, obs, info

    def _select_skill(self, state):
        """Заглушка: возвращает noop-skill. Позже — поиск в библиотеке или codegen."""
        def noop_skill(state):
            yield 0  # action 0 = noop
        return noop_skill
