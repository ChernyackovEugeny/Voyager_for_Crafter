"""
Primitive functions injected into every generated skill's execution namespace.

Design decisions:
  - SemanticView returns the full 64×64 world map.
    find_nearest crops it to the current view window before searching.
  - go_to still plans on the full map because it consumes already-known
    coordinates (from visible perception or spatial memory), not new perception.
  - go_to is a sub-generator (uses yield + return state). Skills call it as:
        state = yield from go_to(coords, state)
    The return value is captured by `yield from` via Python 3.3+ StopIteration.value.
  - Spatial memory primitives are injected by skills.runner.SkillRuntime.
    This module keeps action/navigation primitives only.
  - All action-index constants are pre-computed at module load from
    crafter.constants — no magic numbers, no per-call .index() overhead.
  - ID tables (_NAME_TO_ID, WALKABLE_IDS) are shared with captioner.py via
    environment/ids.py — single source of truth.
  - Lava is excluded from WALKABLE_IDS despite being in Player.walkable because
    stepping on lava sets health to 0.
"""

from collections import deque

import numpy as np
import crafter.constants as _C

from environment.ids import NAME_TO_ID as _NAME_TO_ID, WALKABLE_IDS as _WALKABLE_IDS
from environment.view import visible_semantic_window
from environment.danger import HOSTILE_NAMES as _HOSTILE_NAMES, hostile_visible


ACTION_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do_action",
    "sleep_action",
    "craft",
    "place",
    "find_nearest",
    "find_nearest_hostile",
    "go_to",
    "get_position",
    "is_hostile_visible",
    "move_away_from_hostile",
})

MEMORY_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "get_memory",
    "save_in_memory",
    "delete_memory",
    "set_home",
    "get_home",
})

PRIMITIVE_NAMES: frozenset[str] = ACTION_PRIMITIVE_NAMES | MEMORY_PRIMITIVE_NAMES


# ---------------------------------------------------------------------------
# Action-index constants (pre-computed at import time)
# ---------------------------------------------------------------------------

# Map item name → craft/place action integer.
_CRAFT_ACTIONS: dict[str, int] = {
    name: _C.actions.index(f"make_{name}")
    for name in ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                 "wood_sword", "stone_sword", "iron_sword")
}
_PLACE_ACTIONS: dict[str, int] = {
    name: _C.actions.index(f"place_{name}")
    for name in ("stone", "table", "furnace", "plant")
}

# Pre-computed movement action integers.
# objects.py Player._move: left=(-1,0), right=(+1,0), up=(0,-1), down=(0,+1).
_ACT_MOVE_LEFT  = _C.actions.index("move_left")
_ACT_MOVE_RIGHT = _C.actions.index("move_right")
_ACT_MOVE_UP    = _C.actions.index("move_up")
_ACT_MOVE_DOWN  = _C.actions.index("move_down")
_ACT_DO         = _C.actions.index("do")
_ACT_SLEEP      = _C.actions.index("sleep")
_ACT_NOOP       = _C.actions.index("noop")

# (dx, dy) → action integer for BFS path construction.
_DIR_TO_ACTION: dict[tuple[int, int], int] = {
    (-1, 0): _ACT_MOVE_LEFT,
    (+1, 0): _ACT_MOVE_RIGHT,
    (0, -1): _ACT_MOVE_UP,
    (0, +1): _ACT_MOVE_DOWN,
}
_DIRS = list(_DIR_TO_ACTION.keys())

_MAP_W, _MAP_H = 64, 64  # fixed world size (crafter default area)


# ---------------------------------------------------------------------------
# Simple action-return functions (return int; skill yields the result)
# ---------------------------------------------------------------------------

def noop()         -> int: return _ACT_NOOP
def move_left()    -> int: return _ACT_MOVE_LEFT
def move_right()   -> int: return _ACT_MOVE_RIGHT
def move_up()      -> int: return _ACT_MOVE_UP
def move_down()    -> int: return _ACT_MOVE_DOWN
def do_action()    -> int: return _ACT_DO
def sleep_action() -> int: return _ACT_SLEEP


def craft(item: str) -> int:
    """Return the action integer for crafting `item`. KeyError on unknown item."""
    return _CRAFT_ACTIONS[item]


def place(item: str) -> int:
    """Return the action integer for placing `item`. KeyError on unknown item."""
    return _PLACE_ACTIONS[item]


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

def get_position(state: dict) -> tuple[int, int]:
    """
    Return the player's absolute (x, y) position as a plain Python int tuple.
    player_pos from the engine is a numpy array — casting to int is required
    for use as dict keys, set elements, and BFS queue items.
    """
    pos = state["info"]["player_pos"]
    return (int(pos[0]), int(pos[1]))


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

def find_nearest(object_type: str, state: dict) -> tuple[int, int] | None:
    """
    Search the visible semantic window for the nearest tile of `object_type`.

    state["info"]["semantic"] contains the full world grid, but perception is
    limited to the current view window before searching.

    Returns absolute (x, y) of the nearest visible match by Manhattan distance,
    or None if no such tile is currently visible.
    """
    target_id = _NAME_TO_ID.get(object_type)
    if target_id is None:
        return None
    semantic, (x_offset, y_offset) = visible_semantic_window(state["info"])
    positions = np.argwhere(semantic == target_id)  # shape (N, 2): [[x, y], ...]
    if len(positions) == 0:
        return None
    px, py = get_position(state)
    absolute_positions = positions + np.array([x_offset, y_offset])
    dists = np.abs(absolute_positions - np.array([px, py])).sum(axis=1)
    nearest = absolute_positions[np.argmin(dists)]
    return (int(nearest[0]), int(nearest[1]))


def is_hostile_visible(state: dict) -> bool:
    """Return True when a hostile entity or projectile is currently visible."""
    return hostile_visible(state["info"])


def find_nearest_hostile(state: dict) -> tuple[int, int] | None:
    """Return the nearest visible hostile coordinate, preferring closest threat."""
    targets = [
        coords
        for name in _HOSTILE_NAMES
        for coords in [find_nearest(name, state)]
        if coords is not None
    ]
    if not targets:
        return None
    px, py = get_position(state)
    return min(targets, key=lambda coords: abs(coords[0] - px) + abs(coords[1] - py))


def move_away_from_hostile(state: dict) -> int:
    """
    Return a movement action that increases Manhattan distance from the nearest
    visible hostile when possible. Falls back to move_up.
    """
    threat = find_nearest_hostile(state)
    if threat is None:
        return move_up()
    px, py = get_position(state)
    dx = px - threat[0]
    dy = py - threat[1]
    if abs(dx) >= abs(dy):
        return move_right() if dx >= 0 else move_left()
    return move_down() if dy >= 0 else move_up()


def _is_adjacent(x1: int, y1: int, x2: int, y2: int) -> bool:
    return abs(x1 - x2) + abs(y1 - y2) == 1


def _bfs(
    sx: int, sy: int,
    tx: int, ty: int,
    semantic,
    target_is_walkable: bool,
) -> list[int] | None:
    """
    BFS pathfinding on the 64×64 grid.

    Returns a list of action integers (possibly empty if already at goal),
    or None if the target is unreachable.

    Termination:
      - Walkable target:     player arrives AT (tx, ty).
      - Non-walkable target: player arrives at any orthogonal neighbor of
                             (tx, ty). After the last step the player is
                             facing the target (objects.py _move always sets
                             facing = last movement direction), so do_action()
                             will correctly interact with the target.

    The BFS never routes the player onto lava (not in _WALKABLE_IDS) or into
    non-walkable tiles (stone, tree, water, objects, etc.).
    """

    def is_goal(x: int, y: int) -> bool:
        if target_is_walkable:
            return x == tx and y == ty
        return _is_adjacent(x, y, tx, ty)

    def can_enter(x: int, y: int) -> bool:
        if not (0 <= x < _MAP_W and 0 <= y < _MAP_H):
            return False
        return int(semantic[x, y]) in _WALKABLE_IDS

    if is_goal(sx, sy):
        return []  # already at goal; no movement needed

    visited: set[tuple[int, int]] = {(sx, sy)}
    queue: deque[tuple[int, int, list[int]]] = deque()
    queue.append((sx, sy, []))

    while queue:
        cx, cy, path = queue.popleft()
        for dx, dy in _DIRS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            new_path = path + [_DIR_TO_ACTION[(dx, dy)]]
            if is_goal(nx, ny):
                if not can_enter(nx, ny):
                    continue  # target tile is walkable type but currently blocked
                return new_path
            if not can_enter(nx, ny):
                continue
            visited.add((nx, ny))
            queue.append((nx, ny, new_path))

    return None  # unreachable


def _face_target_action(px: int, py: int, tx: int, ty: int) -> int | None:
    dx, dy = tx - px, ty - py
    if abs(dx) + abs(dy) != 1:
        return None
    return _DIR_TO_ACTION.get((dx, dy))


def go_to(target_coords: tuple, state: dict):
    """
    Sub-generator: BFS pathfinding from the player's current position to
    target_coords. Yields movement actions one by one; receives updated state
    via the executor's send() call after each step.

    Usage in skills:
        state = yield from go_to(coords, state)

    Termination semantics:
      - If target tile is walkable (grass/path/sand): navigate to exact coords.
      - If target tile is non-walkable (tree, stone, cow, etc.): navigate to
        an adjacent cell so that do_action() reaches the target.

    Replanning: after executing the planned path the function re-checks arrival.
    If the player was knocked back or an object moved, it replans up to
    MAX_REPLANS times before giving up and returning the current state.
    """
    try:
        if target_coords is None or len(target_coords) < 2:
            return state
    except TypeError:
        return state
    MAX_REPLANS = 10
    tx, ty = int(target_coords[0]), int(target_coords[1])

    for _ in range(MAX_REPLANS):
        px, py = get_position(state)

        # Determine if the target tile is walkable at this moment.
        semantic = state["info"]["semantic"]
        target_id = int(semantic[tx, ty]) if 0 <= tx < _MAP_W and 0 <= ty < _MAP_H else 0
        target_is_walkable = target_id in _WALKABLE_IDS

        # Check arrival before doing anything.
        if target_is_walkable and (px, py) == (tx, ty):
            return state
        if not target_is_walkable and _is_adjacent(px, py, tx, ty):
            action = _face_target_action(px, py, tx, ty)
            if action is not None:
                state = yield action
            return state

        path = _bfs(px, py, tx, ty, semantic, target_is_walkable)
        if path is None:
            return state   # unreachable; caller decides what to do
        if not path:
            return state   # already at goal (caught by _bfs returning [])

        # Execute the planned path step by step.
        for action in path:
            state = yield action
            # Early-exit: check if we already arrived (target may have shifted).
            cx, cy = get_position(state)
            new_target_id = int(state["info"]["semantic"][tx, ty]) \
                if 0 <= tx < _MAP_W and 0 <= ty < _MAP_H else 0
            new_target_walkable = new_target_id in _WALKABLE_IDS
            if new_target_walkable and (cx, cy) == (tx, ty):
                return state
            if not new_target_walkable and _is_adjacent(cx, cy, tx, ty):
                action = _face_target_action(cx, cy, tx, ty)
                if action is not None:
                    state = yield action
                return state

        # Path exhausted but not arrived → replan (outer loop).

    return state  # gave up after MAX_REPLANS
