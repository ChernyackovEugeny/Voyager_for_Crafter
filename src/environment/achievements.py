"""Static catalog of Crafter achievements and their dependency graph."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Achievement:
    """One Crafter achievement known to the curriculum."""

    key: str
    description: str
    prerequisites: tuple[str, ...]


ACHIEVEMENTS: dict[str, Achievement] = {
    "collect_wood": Achievement(
        key="collect_wood",
        description="Chop a tree to obtain wood.",
        prerequisites=(),
    ),
    "collect_drink": Achievement(
        key="collect_drink",
        description="Drink water by facing a water tile and using do_action.",
        prerequisites=(),
    ),
    "eat_cow": Achievement(
        key="eat_cow",
        description="Defeat a cow and eat it.",
        prerequisites=(),
    ),
    "collect_sapling": Achievement(
        key="collect_sapling",
        description="Pick up a sapling from a grass tile.",
        prerequisites=(),
    ),
    "place_table": Achievement(
        key="place_table",
        description="Place a crafting table on the ground (consumes 2 wood).",
        prerequisites=("collect_wood",),
    ),
    "make_wood_pickaxe": Achievement(
        key="make_wood_pickaxe",
        description="Craft a wooden pickaxe while standing next to a crafting table.",
        prerequisites=("collect_wood", "place_table"),
    ),
    "collect_stone": Achievement(
        key="collect_stone",
        description="Mine a stone block (requires at least a wooden pickaxe).",
        prerequisites=("make_wood_pickaxe",),
    ),
    "place_stone": Achievement(
        key="place_stone",
        description="Place a stone block on the ground.",
        prerequisites=("collect_stone",),
    ),
    "make_stone_pickaxe": Achievement(
        key="make_stone_pickaxe",
        description="Craft a stone pickaxe at a crafting table.",
        prerequisites=("collect_stone", "place_table"),
    ),
    "collect_coal": Achievement(
        key="collect_coal",
        description="Mine a coal block (requires at least a wooden pickaxe).",
        prerequisites=("make_wood_pickaxe",),
    ),
    "place_furnace": Achievement(
        key="place_furnace",
        description="Place a furnace on the ground (consumes stone).",
        prerequisites=("collect_stone",),
    ),
    "collect_iron": Achievement(
        key="collect_iron",
        description="Mine an iron block (requires at least a stone pickaxe).",
        prerequisites=("make_stone_pickaxe",),
    ),
    "make_iron_pickaxe": Achievement(
        key="make_iron_pickaxe",
        description="Craft an iron pickaxe at a crafting table next to a furnace.",
        prerequisites=("collect_iron", "collect_coal", "place_table", "place_furnace"),
    ),
    "collect_diamond": Achievement(
        key="collect_diamond",
        description="Mine a diamond block (requires an iron pickaxe).",
        prerequisites=("make_iron_pickaxe",),
    ),
    "place_plant": Achievement(
        key="place_plant",
        description="Plant a sapling on a grass tile.",
        prerequisites=("collect_sapling",),
    ),
    "eat_plant": Achievement(
        key="eat_plant",
        description="Eat a fully grown plant.",
        prerequisites=("place_plant",),
    ),
    "make_wood_sword": Achievement(
        key="make_wood_sword",
        description="Craft a wooden sword while standing next to a crafting table.",
        prerequisites=("collect_wood", "place_table"),
    ),
    "make_stone_sword": Achievement(
        key="make_stone_sword",
        description="Craft a stone sword at a crafting table.",
        prerequisites=("collect_stone", "place_table"),
    ),
    "make_iron_sword": Achievement(
        key="make_iron_sword",
        description="Craft an iron sword at a crafting table next to a furnace.",
        prerequisites=("collect_iron", "collect_coal", "place_table", "place_furnace"),
    ),
    "defeat_zombie": Achievement(
        key="defeat_zombie",
        description="Defeat a zombie in combat.",
        prerequisites=(),
    ),
    "defeat_skeleton": Achievement(
        key="defeat_skeleton",
        description="Defeat a skeleton in combat.",
        prerequisites=(),
    ),
    "wake_up": Achievement(
        key="wake_up",
        description="Sleep to recover energy and wake up safely.",
        prerequisites=(),
    ),
}


# Survival-first order for HardcodedCurriculum.
TECH_TREE_ORDER: tuple[str, ...] = (
    "collect_wood",
    "collect_drink",
    "eat_cow",
    "collect_sapling",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "place_stone",
    "make_stone_pickaxe",
    "collect_coal",
    "place_furnace",
    "collect_iron",
    "make_iron_pickaxe",
    "collect_diamond",
    "place_plant",
    "eat_plant",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "defeat_zombie",
    "defeat_skeleton",
    "wake_up",
)
