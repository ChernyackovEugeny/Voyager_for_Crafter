class SpatialMemory:
    """Episode-local memory for known object coordinates and home location."""

    def __init__(self):
        self._memory: dict[str, tuple] = {}
        self._home: tuple | None = None

    def reset(self):
        """Clear all locations at the start of a new episode."""
        self._memory = {}
        self._home = None

    def memory_add(self, obj_name: str, coordinates: tuple):
        """Save or overwrite a named object location."""
        self._memory[obj_name] = coordinates

    def memory_delete(self, obj_name: str):
        """Remove a named object location. Raises KeyError if missing."""
        del self._memory[obj_name]

    def get_memory(self) -> dict:
        """Return a copy so skills cannot mutate memory accidentally."""
        return dict(self._memory)

    def text_memory(self) -> str:
        """Return a compact text summary for future LLM prompts."""
        if not self._memory:
            return "empty"
        return ", ".join(self._memory.keys())

    def set_home(self, coordinates: tuple):
        """Save a special home/base coordinate."""
        self._home = coordinates

    def get_home(self) -> tuple | None:
        """Return saved home coordinates, or None if unset."""
        return self._home
