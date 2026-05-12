import io
import sys

# gym (crafter dependency) prints an unmaintained notice to stderr on import.
# Suppress it by redirecting stderr for the duration of the import.
_stderr = sys.stderr
sys.stderr = io.StringIO()
import crafter
sys.stderr = _stderr
del _stderr


class CrafterEnv:
    """
    Thin wrapper over crafter.Env().
    Adapts the old gym API (done: bool) to gymnasium-style (terminated, truncated).
    Exposes action_space and observation_space for compatibility.
    """

    def __init__(self, **kwargs):
        self._env = crafter.Env(**kwargs)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        view = getattr(self._env, "_view", (9, 9))
        self.view_size = (int(view[0]), int(view[1]))

    def reset(self):
        """Returns: obs (np.ndarray)"""
        obs = self._env.reset()
        return obs

    def step(self, action):
        """
        Returns:
            obs         (np.ndarray)
            reward      (float)
            terminated  (bool) — episode ended naturally (death / time limit)
            truncated   (bool) — always False (crafter has no truncation)
            info        (dict) — achievements, inventory, health, semantic,
                                  player_pos, view_size
        """
        obs, reward, done, info = self._env.step(action)
        info["view_size"] = self.view_size
        return obs, reward, bool(done), False, info

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Convenience accessors into info (populated after first step)
    # ------------------------------------------------------------------

    @staticmethod
    def get_achievements(info: dict) -> dict:
        """Returns the 22-achievement flag dict from info."""
        return info.get("achievements", {})

    @staticmethod
    def get_inventory(info: dict) -> dict:
        return info.get("inventory", {})

    @staticmethod
    def get_health(info: dict) -> int:
        return info.get("health", 9)

    @staticmethod
    def get_player_pos(info: dict) -> tuple:
        return info.get("player_pos", (0, 0))

    @staticmethod
    def get_semantic_map(info: dict):
        return info.get("semantic", None)
