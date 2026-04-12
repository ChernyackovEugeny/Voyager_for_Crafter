class SpatialMemory:
    def __init__(self):
        self._memory: dict[str, tuple] = {}  # {obj_name: (x, y)}
        self._home: tuple | None = None

    def reset(self):
        # Полное очищение памяти и home.
        # Вызывается в начале каждого эпизода — позиции из прошлого эпизода
        # не валидны (мир генерируется заново).
        self._memory = {}
        self._home = None

    def memory_add(self, obj_name: str, coordinates: tuple):
        # Добавляет или перезаписывает координаты объекта.
        # Перезапись нужна если объект переместился или был обновлён.
        self._memory[obj_name] = coordinates

    def memory_delete(self, obj_name: str):
        # Удаляет объект. Бросает KeyError если объекта нет —
        # ловится на уровне выше (agent/skill).
        # Намеренно не используем discard() чтобы ошибка была явной.
        del self._memory[obj_name]

    def get_memory(self) -> dict:
        # Возвращает копию словаря чтобы skill не мог случайно мутировать память.
        return dict(self._memory)

    def text_memory(self) -> str:
        # Возвращает объекты через запятую для вставки в текстовый prompt.
        # Пустая память → "empty" чтобы LLM понял что памяти нет.
        if not self._memory:
            return "empty"
        return ", ".join(self._memory.keys())

    def set_home(self, coordinates: tuple):
        # Сохраняет домашнюю позицию (место с кроватью / базой).
        # Выделен отдельно от _memory — home это особый объект,
        # часто используемый в skills для возврата.
        self._home = coordinates

    def get_home(self) -> tuple | None:
        # Возвращает None если home не установлен.
        # Skills должны проверять на None перед использованием.
        return self._home
