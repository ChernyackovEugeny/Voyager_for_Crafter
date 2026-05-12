"""
Tile and object ID mappings for the Crafter semantic map.

Single source of truth — used by both captioner.py (ID → name, for rendering)
and primitives.py (name → ID, for find_nearest / go_to).

ID scheme (matches engine.py World._mat_ids and SemanticView.__init__):
  0          = None / void
  1 .. N     = materials (from crafter.constants.materials, 1-indexed)
  N+1 .. N+6 = objects in the order SemanticView receives them:
               Player, Cow, Zombie, Skeleton, Arrow, Plant
"""

import crafter.constants as _C

# id → name
ID_TO_NAME: dict[int, str] = {
    i + 1: name for i, name in enumerate(_C.materials)
}
_OBJ_BASE = len(_C.materials) + 1   # = 13
for _i, _n in enumerate(("player", "cow", "zombie", "skeleton", "arrow", "plant")):
    ID_TO_NAME[_OBJ_BASE + _i] = _n

# name → id  (inverse of ID_TO_NAME, derived — never diverges)
NAME_TO_ID: dict[str, int] = {name: id_ for id_, name in ID_TO_NAME.items()}

# Tile IDs the player can safely walk onto.
# Lava is excluded: Player.walkable includes lava in the engine, but
# stepping on lava immediately sets health to 0.
WALKABLE_IDS: frozenset[int] = frozenset(
    NAME_TO_ID[n] for n in ("grass", "path", "sand")
)
