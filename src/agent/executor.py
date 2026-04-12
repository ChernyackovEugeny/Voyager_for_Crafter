class Executor:
    def __init__(self, env):
        self.env = env

    def run(self, skill_func, state: dict) -> tuple:
        """
        Запускает yield-генератор skill_func(state) пошагово.
        Возвращает (reason, obs, info) — финальный obs/info после завершения skill.
        reason: 'done', 'completed', 'error'
        """
        obs = state["obs"]
        info = state["info"]
        try:
            generator = skill_func(state)
            action = next(generator)
            while True:
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    return "done", obs, info
                state = self._build_state(obs, info)
                try:
                    action = generator.send(state)
                except StopIteration:
                    return "completed", obs, info
        except Exception:
            return "error", obs, info

    def _build_state(self, obs, info) -> dict:
        return {"obs": obs, "info": info}
