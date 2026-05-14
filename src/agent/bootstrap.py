from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from analytics.log_utils import log_llm_call_ok

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapSkillSpec:
    name: str
    description: str


BOOTSTRAP_SKILLS: tuple[BootstrapSkillSpec, ...] = (
    BootstrapSkillSpec(
        name="scout_safely",
        description=(
            "Scout the nearby area for useful resources without using explore_for. "
            "Move in short explicit bursts, after every step update state, check "
            "is_hostile_visible, retreat when needed, and save visible water, cow, "
            "tree, table, and safe base coordinates in spatial memory. Call "
            "movement primitives directly; do not use direction_func or a list "
            "of movement functions. Do not walk in a four-cell circle; use "
            "expanding sweeps or longer segments before turning. Guard every "
            "optional coordinate with `if coords is not None` before indexing, "
            "go_to, save_in_memory, or set_home. End with return state, not "
            "bare return."
        ),
    ),
    BootstrapSkillSpec(
        name="collect_wood",
        description=(
            "Find a visible or remembered tree, go adjacent to it, chop until "
            "collect_wood is unlocked or enough wood is available, and save tree "
            "coordinates when found. Avoid visible hostiles and end with return "
            "state."
        ),
    ),
    BootstrapSkillSpec(
        name="collect_drink",
        description=(
            "Find visible or remembered water, save it as 'water', go adjacent to "
            "it, face it, and drink until collect_drink is unlocked or drink is "
            "restored to a safe level. If water is "
            "not visible, scout with short movement loops and danger checks after "
            "every yielded action. Do not call explore_for, find_water, "
            "go_to_water, or find_and_drink_water. Do not walk in a four-cell "
            "circle. Guard remembered or visible water coordinates before use."
            " End with return state, not bare return."
        ),
    ),
    BootstrapSkillSpec(
        name="eat_cow",
        description=(
            "Find visible or remembered food, prefer cow then plant, save food "
            "coordinates, approach safely, and eat until food is restored or "
            "eat_cow is unlocked. Check hostiles after every yielded action and "
            "do not call explore_for or invented helper functions. Guard all "
            "remembered or visible food coordinates before use."
            " End with return state, not bare return."
        ),
    ),
    BootstrapSkillSpec(
        name="build_shelter",
        description=(
            "Build a minimal base: collect enough wood if needed, place a table "
            "only when the tile in front can accept it, save "
            "set_home(get_position(state)) and the table location, then use "
            "defensive placement when possible: place stone blocks to close "
            "gaps or make a small barrier against visible hostiles. Do not call "
            "explore_for. Guard any grass, tree, stone, or table coordinates "
            "before go_to, save_in_memory, or indexing."
            " End with return state, not bare return."
        ),
    ),
    BootstrapSkillSpec(
        name="make_wood_pickaxe",
        description=(
            "Craft a wooden pickaxe reliably. Ensure a crafting table exists, "
            "collect enough wood for table plus pickaxe when needed, stand next "
            "to the table, and craft wood_pickaxe. Avoid visible hostiles and "
            "return state."
        ),
    ),
    BootstrapSkillSpec(
        name="collect_stone",
        description=(
            "Mine at least one stone with a wooden pickaxe. Craft the pickaxe "
            "first if needed, use visible or remembered stone, save stone "
            "coordinates, go adjacent, mine, and avoid visible hostiles."
        ),
    ),
    BootstrapSkillSpec(
        name="place_stone",
        description=(
            "Place a stone block near the saved home or crafting table to start "
            "fortifying the base. Collect stone first if needed, return to the "
            "base, place only when can_place_ahead('stone', state) is true, and "
            "avoid visible hostiles."
        ),
    ),
    BootstrapSkillSpec(
        name="fight_isolated_zombie",
        description=(
            "Defeat a single visible zombie only when no skeleton or arrow is "
            "visible, health is at least 5, and a sword is available. Go "
            "adjacent, face the zombie, attack with do_action several times, "
            "and stop fighting if conditions become unsafe."
        ),
    ),
    BootstrapSkillSpec(
        name="block_with_stone",
        description=(
            "Use defensive placement to block a visible threat or close a gap "
            "near home. If stone is available and can_place_ahead('stone', "
            "state) is true, place stone, update state, and re-check hostiles."
        ),
    ),
    BootstrapSkillSpec(
        name="place_table_barrier",
        description=(
            "Use a crafting table as an emergency obstacle and home anchor when "
            "wood is available but stone is not. Place a table only if "
            "can_place_ahead('table', state) is true, save home, and re-check "
            "hostiles."
        ),
    ),
    BootstrapSkillSpec(
        name="wait_at_home",
        description=(
            "Return to remembered home if it exists, save current position as "
            "home otherwise, and wait with noop while repeatedly checking "
            "hostile visibility and survival stats."
        ),
    ),
    BootstrapSkillSpec(
        name="restore_drink",
        description=(
            "Restore drink to at least 8/9. Use remembered or visible water, "
            "save water coordinates, go adjacent, face it, and drink repeatedly. "
            "Do this even if collect_drink is already unlocked. Avoid visible "
            "hostiles after every yielded action and return state."
        ),
    ),
    BootstrapSkillSpec(
        name="restore_food",
        description=(
            "Restore food to at least 8/9. Use remembered or visible cows or "
            "plants, save food coordinates, approach safely, and eat repeatedly. "
            "Do this even if eat_cow is already unlocked. Avoid visible hostiles "
            "after every yielded action and return state."
        ),
    ),
    BootstrapSkillSpec(
        name="survive",
        description=(
            "Survive the first night and recover health, food, drink, and energy. "
            "Use remembered water, food, and home first; otherwise scout only in "
            "short explicit loops with a danger check after every yield. If a "
            "hostile is visible, immediately move away for several steps, or if "
            "stone is available use can_place_ahead('stone', state) and "
            "place('stone') to block a path or close a gap near home. If no "
            "stone is available but wood is available, place a table as a "
            "temporary obstacle and home anchor. Fight only when armed. Stop "
            "only when no hostile is visible and health, food, and drink are "
            "recovered. Use get_memory().get('water'), not get_memory('water'). "
            "Do not call explore_for or invented helper functions. Do not walk "
            "in a four-cell circle. Guard remembered water, food, and home "
            "coordinates before go_to or indexing."
            " End with return state, not bare return."
        ),
    ),
)


BOOTSTRAP_CODE: dict[str, str] = {
    "scout_safely": """\
def scout_area(state):
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(60):
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        for name in ("water", "cow", "tree", "stone", "coal", "table"):
            coords = find_nearest(name, state)
            if coords is not None:
                save_in_memory(name, coords)
        if get_home() is None:
            set_home(get_position(state))
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "collect_wood": """\
def collect_wood(state):
    if state["info"]["inventory"].get("wood", 0) >= 3:
        return state
    tree_coords = get_memory().get("tree")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(80):
        if state["info"]["inventory"].get("wood", 0) >= 3:
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        if tree_coords is None:
            tree_coords = find_nearest("tree", state)
            if tree_coords is not None:
                save_in_memory("tree", tree_coords)
        if tree_coords is not None:
            state = yield from go_to(tree_coords, state)
            for _ in range(5):
                if state["info"]["inventory"].get("wood", 0) >= 3:
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            tree_coords = None
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "collect_drink": """\
def collect_drink(state):
    if (
        state["info"]["achievements"].get("collect_drink", 0)
        and state["info"]["inventory"].get("drink", 9) >= 8
    ):
        return state
    water_coords = get_memory().get("water")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(90):
        if (
            state["info"]["achievements"].get("collect_drink", 0)
            and state["info"]["inventory"].get("drink", 9) >= 8
        ):
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        if water_coords is None:
            water_coords = find_nearest("water", state)
            if water_coords is not None:
                save_in_memory("water", water_coords)
        if water_coords is not None:
            state = yield from go_to(water_coords, state)
            for _ in range(6):
                if (
                    state["info"]["achievements"].get("collect_drink", 0)
                    and state["info"]["inventory"].get("drink", 9) >= 8
                ):
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            water_coords = get_memory().get("water")
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "eat_cow": """\
def find_and_eat_food(state):
    if (
        state["info"]["achievements"].get("eat_cow", 0)
        and state["info"]["inventory"].get("food", 9) >= 8
    ):
        return state
    food_coords = get_memory().get("cow")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(120):
        if (
            state["info"]["achievements"].get("eat_cow", 0)
            and state["info"]["inventory"].get("food", 9) >= 8
        ):
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        visible_cow = find_nearest("cow", state)
        if visible_cow is not None:
            food_coords = visible_cow
            save_in_memory("cow", food_coords)
        if food_coords is not None:
            state = yield from go_to(food_coords, state)
            for _ in range(8):
                if (
                    state["info"]["achievements"].get("eat_cow", 0)
                    and state["info"]["inventory"].get("food", 9) >= 8
                ):
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            food_coords = None
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "build_shelter": """\
def build_minimal_base(state):
    if state["info"]["achievements"].get("place_table", 0):
        if get_home() is None:
            set_home(get_position(state))
        return state
    if state["info"]["inventory"].get("wood", 0) < 3:
        state = yield from collect_wood(state)
    if state["info"]["inventory"].get("wood", 0) < 2:
        return state
    direction = 0
    segment_len = 3
    segment_progress = 0
    turns = 0
    for step in range(40):
        if state["info"]["achievements"].get("place_table", 0):
            break
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        if can_place_ahead("table", state):
            state = yield place("table")
            break
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 1
    table_coords = find_nearest("table", state)
    if table_coords is not None:
        save_in_memory("table", table_coords)
    set_home(get_position(state))
    return state
""",
    "make_wood_pickaxe": """\
def make_wood_pickaxe(state):
    if state["info"]["inventory"].get("wood_pickaxe", 0) >= 1:
        return state
    table_coords = get_memory().get("table")
    visible_table = find_nearest("table", state)
    if visible_table is not None:
        table_coords = visible_table
        save_in_memory("table", table_coords)
    if table_coords is None and not state["info"]["achievements"].get("place_table", 0):
        if state["info"]["inventory"].get("wood", 0) < 3:
            state = yield from collect_wood(state)
        if state["info"]["inventory"].get("wood", 0) >= 2:
            state = yield from build_shelter(state)
        visible_table = find_nearest("table", state)
        if visible_table is not None:
            table_coords = visible_table
            save_in_memory("table", table_coords)
    if state["info"]["inventory"].get("wood", 0) < 1:
        state = yield from collect_wood(state)
    if state["info"]["inventory"].get("wood", 0) < 1:
        return state
    if table_coords is None:
        table_coords = get_memory().get("table")
    if table_coords is None:
        table_coords = find_nearest("table", state)
        if table_coords is not None:
            save_in_memory("table", table_coords)
    if table_coords is not None:
        state = yield from go_to(table_coords, state)
        for _ in range(4):
            if state["info"]["inventory"].get("wood_pickaxe", 0) >= 1:
                return state
            if is_hostile_visible(state):
                state = yield move_away_from_hostile(state)
                break
            state = yield craft("wood_pickaxe")
    return state
""",
    "collect_stone": """\
def collect_stone(state):
    if state["info"]["inventory"].get("stone", 0) >= 1:
        return state
    if state["info"]["inventory"].get("wood_pickaxe", 0) < 1:
        state = yield from make_wood_pickaxe(state)
    if state["info"]["inventory"].get("wood_pickaxe", 0) < 1:
        return state
    stone_coords = get_memory().get("stone")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(90):
        if state["info"]["inventory"].get("stone", 0) >= 1:
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        visible_stone = find_nearest("stone", state)
        if visible_stone is not None:
            stone_coords = visible_stone
            save_in_memory("stone", stone_coords)
        if stone_coords is not None:
            state = yield from go_to(stone_coords, state)
            for _ in range(6):
                if state["info"]["inventory"].get("stone", 0) >= 1:
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            stone_coords = None
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "place_stone": """\
def place_stone(state):
    if state["info"]["achievements"].get("place_stone", 0):
        return state
    if state["info"]["inventory"].get("stone", 0) < 1:
        state = yield from collect_stone(state)
    if state["info"]["inventory"].get("stone", 0) < 1:
        return state
    home = get_home()
    if home is not None:
        state = yield from go_to(home, state)
    table_coords = get_memory().get("table")
    visible_table = find_nearest("table", state)
    if visible_table is not None:
        table_coords = visible_table
        save_in_memory("table", table_coords)
    if table_coords is not None:
        state = yield from go_to(table_coords, state)
    direction = 0
    segment_len = 2
    segment_progress = 0
    turns = 0
    for step in range(40):
        if state["info"]["achievements"].get("place_stone", 0):
            return state
        if state["info"]["inventory"].get("stone", 0) < 1:
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        if can_place_ahead("stone", state):
            state = yield place("stone")
            return state
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 1
    return state
""",
    "fight_isolated_zombie": """\
def fight_isolated_zombie(state):
    inv = state["info"]["inventory"]
    zombie = find_nearest("zombie", state)
    skeleton = find_nearest("skeleton", state)
    arrow = find_nearest("arrow", state)
    has_sword = (
        inv.get("wood_sword", 0) > 0
        or inv.get("stone_sword", 0) > 0
        or inv.get("iron_sword", 0) > 0
    )
    if skeleton is not None or arrow is not None:
        return state
    if not has_sword or inv.get("health", 9) < 5:
        return state
    if zombie is not None:
        state = yield from go_to(zombie, state)
    else:
        return state
    for _ in range(6):
        if find_nearest("zombie", state) is None:
            return state
        if find_nearest("skeleton", state) is not None or find_nearest("arrow", state) is not None:
            return state
        if state["info"]["inventory"].get("health", 9) < 4:
            return state
        state = yield do_action()
    return state
""",
    "block_with_stone": """\
def block_with_stone(state):
    if state["info"]["inventory"].get("stone", 0) < 1:
        return state
    for _ in range(4):
        if not is_hostile_visible(state):
            return state
        if can_place_ahead("stone", state):
            state = yield place("stone")
            return state
        state = yield move_away_from_hostile(state)
    return state
""",
    "place_table_barrier": """\
def place_table_barrier(state):
    if state["info"]["inventory"].get("wood", 0) < 2:
        return state
    for _ in range(4):
        if not is_hostile_visible(state):
            return state
        if can_place_ahead("table", state):
            state = yield place("table")
            set_home(get_position(state))
            table_coords = find_nearest("table", state)
            if table_coords is not None:
                save_in_memory("table", table_coords)
            return state
        state = yield move_away_from_hostile(state)
    return state
""",
    "wait_at_home": """\
def wait_at_home(state):
    home = get_home()
    if home is not None:
        state = yield from go_to(home, state)
    else:
        set_home(get_position(state))
    for _ in range(25):
        if is_hostile_visible(state):
            return state
        inv = state["info"]["inventory"]
        if inv.get("health", 9) >= 8 and inv.get("food", 9) >= 8 and inv.get("drink", 9) >= 8:
            return state
        state = yield noop()
    return state
""",
    "survive": """\
def survive_first_night(state):
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    calm_steps = 0
    for step in range(220):
        inv = state["info"]["inventory"]
        if is_hostile_visible(state):
            calm_steps = 0
        else:
            calm_steps += 1
        if (
            inv.get("health", 9) >= 8
            and inv.get("food", 9) >= 8
            and inv.get("drink", 9) >= 8
            and inv.get("energy", 9) >= 5
            and not is_hostile_visible(state)
            and calm_steps >= 25
        ):
            return state
        if is_hostile_visible(state):
            before_pos = get_position(state)
            before_health = inv.get("health", 9)
            state = yield from fight_isolated_zombie(state)
            if not is_hostile_visible(state):
                continue
            if (
                get_position(state) != before_pos
                or state["info"]["inventory"].get("health", 9) < before_health
            ):
                continue
            state = yield from block_with_stone(state)
            if not is_hostile_visible(state):
                continue
            state = yield from place_table_barrier(state)
            if not is_hostile_visible(state):
                continue
            state = yield move_away_from_hostile(state)
            continue
        if inv.get("drink", 9) < 7:
            water_coords = get_memory().get("water")
            visible_water = find_nearest("water", state)
            if visible_water is not None:
                water_coords = visible_water
                save_in_memory("water", water_coords)
            if water_coords is not None:
                state = yield from go_to(water_coords, state)
                for _ in range(6):
                    if state["info"]["inventory"].get("drink", 9) >= 8:
                        break
                    if is_hostile_visible(state):
                        state = yield move_away_from_hostile(state)
                        break
                    state = yield do_action()
                continue
        if inv.get("food", 9) < 7:
            food_coords = get_memory().get("cow") or get_memory().get("plant")
            visible_cow = find_nearest("cow", state)
            visible_plant = find_nearest("plant", state)
            if visible_cow is not None:
                food_coords = visible_cow
                save_in_memory("cow", food_coords)
            elif visible_plant is not None:
                food_coords = visible_plant
                save_in_memory("plant", food_coords)
            if food_coords is not None:
                state = yield from go_to(food_coords, state)
                for _ in range(6):
                    if state["info"]["inventory"].get("food", 9) >= 8:
                        break
                    if is_hostile_visible(state):
                        state = yield move_away_from_hostile(state)
                        break
                    state = yield do_action()
                continue
        if inv.get("energy", 9) < 5:
            home = get_home()
            if home is not None:
                state = yield from go_to(home, state)
            if not is_hostile_visible(state):
                state = yield sleep_action()
            continue
        home = get_home()
        if home is not None and calm_steps < 25:
            state = yield from go_to(home, state)
            if state["info"]["inventory"].get("stone", 0) > 0 and can_place_ahead("stone", state):
                state = yield place("stone")
                continue
            if not is_hostile_visible(state):
                state = yield from wait_at_home(state)
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "restore_drink": """\
def restore_drink(state):
    water_coords = get_memory().get("water")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(100):
        if state["info"]["inventory"].get("drink", 9) >= 8:
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        visible_water = find_nearest("water", state)
        if visible_water is not None:
            water_coords = visible_water
            save_in_memory("water", water_coords)
        if water_coords is not None:
            state = yield from go_to(water_coords, state)
            for _ in range(8):
                if state["info"]["inventory"].get("drink", 9) >= 8:
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            water_coords = get_memory().get("water")
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
    "restore_food": """\
def restore_food(state):
    food_coords = get_memory().get("cow")
    direction = 0
    segment_len = 4
    segment_progress = 0
    turns = 0
    for step in range(120):
        if state["info"]["inventory"].get("food", 9) >= 8:
            return state
        if is_hostile_visible(state):
            state = yield move_away_from_hostile(state)
            continue
        visible_cow = find_nearest("cow", state)
        visible_plant = find_nearest("plant", state)
        if visible_cow is not None:
            food_coords = visible_cow
            save_in_memory("cow", food_coords)
        elif visible_plant is not None:
            food_coords = visible_plant
            save_in_memory("plant", food_coords)
        if food_coords is not None:
            state = yield from go_to(food_coords, state)
            for _ in range(8):
                if state["info"]["inventory"].get("food", 9) >= 8:
                    return state
                if is_hostile_visible(state):
                    state = yield move_away_from_hostile(state)
                    break
                state = yield do_action()
            food_coords = get_memory().get("cow") or get_memory().get("plant")
            continue
        if direction == 0:
            state = yield move_right()
        elif direction == 1:
            state = yield move_down()
        elif direction == 2:
            state = yield move_left()
        else:
            state = yield move_up()
        segment_progress += 1
        if segment_progress >= segment_len:
            direction = (direction + 1) % 4
            segment_progress = 0
            turns += 1
            if turns % 2 == 0:
                segment_len += 2
    return state
""",
}


BOOTSTRAP_STATE_TEXT = (
    "Observation: unknown fresh world\n"
    "Inventory: empty\n"
    "Status: health: 9/9, food: 9/9, drink: 9/9, energy: 9/9"
)
BOOTSTRAP_ATTEMPTS = 2


def bootstrap_initial_skills(
    *,
    skill_manager,
    codegen,
    skill_validator: Callable[..., object],
    run_logger=None,
) -> int:
    """Generate and save the initial survival skill set for an empty library."""
    try:
        existing_count = skill_manager.count()
    except Exception as exc:
        logger.warning("[Bootstrap] skipped: cannot count skills: %s", exc)
        return 0
    missing_specs = [
        spec for spec in BOOTSTRAP_SKILLS if not skill_manager.exists(spec.name)
    ]
    if not missing_specs:
        logger.info(
            "[Bootstrap] skipped: all bootstrap skills already exist "
            "(library has %d skill(s))",
            existing_count,
        )
        return 0
    if existing_count > 0:
        logger.info(
            "[Bootstrap] generating %d missing bootstrap skill(s) "
            "(library has %d skill(s))",
            len(missing_specs),
            existing_count,
        )

    saved = 0
    generated_context: list[dict[str, str]] = []
    for spec in BOOTSTRAP_SKILLS:
        if spec in missing_specs:
            continue
        existing = skill_manager.get(spec.name)
        if existing is not None:
            generated_context.append({
                "name": existing.name,
                "description": existing.description,
                "code": existing.code,
            })
    for spec in missing_specs:
        deterministic_code = BOOTSTRAP_CODE.get(spec.name)
        if deterministic_code is not None:
            try:
                skill_validator(
                    deterministic_code,
                    allowed_skill_names={
                        entry["name"] for entry in generated_context
                    },
                    extra_skills=tuple(
                        (entry["name"], entry["code"])
                        for entry in generated_context
                    ),
                )
                try:
                    result = skill_manager.save(
                        name=spec.name,
                        code=deterministic_code,
                        task=spec.description,
                        deduplicate=False,
                        origin="bootstrap",
                    )
                except TypeError:
                    result = skill_manager.save(
                        name=spec.name,
                        code=deterministic_code,
                        task=spec.description,
                    )
                if result.outcome == "ok":
                    generated_context.append({
                        "name": spec.name,
                        "description": spec.description,
                        "code": deterministic_code,
                    })
                    saved += 1
                    logger.info("[Bootstrap] saved deterministic %s", spec.name)
                    continue
                logger.info(
                    "[Bootstrap] did not save deterministic %s: outcome=%s",
                    spec.name,
                    result.outcome,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "[Bootstrap] deterministic validation failed for %s: %s",
                    spec.name,
                    exc,
                )

        previous_failure: tuple[str, str] | None = None
        for attempt in range(1, BOOTSTRAP_ATTEMPTS + 1):
            if attempt == 1:
                logger.info("[Bootstrap] generating %s", spec.name)
            else:
                logger.info(
                    "[Bootstrap] regenerating %s after validation failure",
                    spec.name,
                )
            try:
                call = codegen.get_code(
                    state_text=BOOTSTRAP_STATE_TEXT,
                    task=spec.description,
                    retrieved_skills=generated_context,
                    previous_failure=previous_failure,
                )
            except Exception as exc:
                logger.warning("[Bootstrap] failed for %s: %s", spec.name, exc)
                break
            log_llm_call_ok(
                run_logger,
                call_type="codegen",
                episode_num=0,
                model=call.model,
                tokens_in=call.tokens_in,
                tokens_out=call.tokens_out,
                prompt_cache_hit_tokens=call.prompt_cache_hit_tokens,
                prompt_cache_miss_tokens=call.prompt_cache_miss_tokens,
                reasoning_tokens=call.reasoning_tokens,
                cost_usd=call.cost_usd,
                latency_ms=call.latency_ms,
                prompt_template_id=call.prompt_template_id,
                prompt_hash=call.prompt_hash,
                prompt_text=getattr(call, "prompt_text", None),
                generated_code=getattr(call, "code", None),
                raw_response=getattr(call, "raw_response", None),
            )
            try:
                skill_validator(
                    call.code,
                    allowed_skill_names={entry["name"] for entry in generated_context},
                    extra_skills=tuple(
                        (entry["name"], entry["code"])
                        for entry in generated_context
                    ),
                )
            except Exception as exc:
                previous_failure = (call.code, str(exc))
                logger.warning(
                    "[Bootstrap] validation failed for %s attempt %d/%d: %s",
                    spec.name,
                    attempt,
                    BOOTSTRAP_ATTEMPTS,
                    exc,
                )
                continue
            try:
                result = skill_manager.save(
                    name=spec.name,
                    code=call.code,
                    task=spec.description,
                    deduplicate=False,
                    origin="bootstrap",
                )
            except TypeError:
                result = skill_manager.save(
                    name=spec.name,
                    code=call.code,
                    task=spec.description,
                )
            if result.outcome == "ok":
                generated_context.append({
                    "name": spec.name,
                    "description": spec.description,
                    "code": call.code,
                })
                saved += 1
                logger.info("[Bootstrap] saved %s", spec.name)
                break
            else:
                logger.info(
                    "[Bootstrap] did not save %s: outcome=%s",
                    spec.name,
                    result.outcome,
                )
                break
        else:
            if previous_failure is not None:
                logger.warning(
                    "[Bootstrap] failed for %s: %s",
                    spec.name,
                    previous_failure[1],
                )
    return saved
