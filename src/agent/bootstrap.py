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
            "go_to, save_in_memory, or set_home."
        ),
    ),
    BootstrapSkillSpec(
        name="collect_drink",
        description=(
            "Find visible or remembered water, save it as 'water', go adjacent to "
            "it, face it, and drink until collect_drink is unlocked. If water is "
            "not visible, scout with short movement loops and danger checks after "
            "every yielded action. Do not call explore_for, find_water, "
            "go_to_water, or find_and_drink_water. Do not walk in a four-cell "
            "circle. Guard remembered or visible water coordinates before use."
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
        ),
    ),
    BootstrapSkillSpec(
        name="build_shelter",
        description=(
            "Build a minimal base: collect enough wood if needed, place a table "
            "on safe walkable ground, save set_home(get_position(state)) and the "
            "table location, then retreat from visible hostiles. Do not call "
            "explore_for. Guard any grass, tree, or table coordinates before "
            "go_to, save_in_memory, or indexing."
        ),
    ),
    BootstrapSkillSpec(
        name="survive",
        description=(
            "Survive the first night and recover health, food, drink, and energy. "
            "Use remembered water, food, and home first; otherwise scout only in "
            "short explicit loops with a danger check after every yield. If a "
            "hostile is visible, immediately move away for several steps or fight "
            "only when armed. Stop only when no hostile is visible and health, "
            "food, and drink are recovered. Use get_memory().get('water'), not "
            "get_memory('water'). Do not call explore_for or invented helper "
            "functions. Do not walk in a four-cell circle. Guard remembered "
            "water, food, and home coordinates before go_to or indexing."
        ),
    ),
)


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
                )
            except TypeError:
                result = skill_manager.save(
                    name=spec.name,
                    code=call.code,
                    task=spec.description,
                )
            if result.outcome == "ok":
                skill_manager.record_success(spec.name)
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
