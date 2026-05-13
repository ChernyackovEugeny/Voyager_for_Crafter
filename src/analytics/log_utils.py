"""Thin no-op wrappers around RunLogger.

Why this file exists:
    Business code (codegen.py, agent.py, ...) should never branch on
    `if run_log is not None`. Wrappers absorb that check so callers stay
    clean and unaware of analytics state.

Pattern (from Banking project): every wrapper takes `run_log` first; if it
is None or `_disabled`, the wrapper silently returns. RunLogger itself also
checks `_disabled` — double-safety, since the wrappers may be called before
RunLogger is fully initialised.
"""
from __future__ import annotations

from datetime import datetime

from analytics.run_logger import RunLogger


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def log_episode(
    run_log: RunLogger | None,
    *,
    episode_num: int,
    achievements: dict,
    steps: int,
    total_reward: float,
    started_at: datetime,
    ended_at: datetime,
) -> str | None:
    """Record a finished episode. Returns episode_id or None."""
    if run_log is None:
        return None
    return run_log.record_episode(
        episode_num=episode_num,
        achievements=achievements,
        steps=steps,
        total_reward=total_reward,
        started_at=started_at,
        ended_at=ended_at,
    )


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def log_llm_call_ok(
    run_log: RunLogger | None,
    *,
    call_type: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float = 0.0,
    episode_id: str | None = None,
    episode_num: int | None = None,
    model: str | None = None,
    prompt_template_id: str | None = None,
    prompt_hash: str | None = None,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
    reasoning_tokens: int | None = None,
    prompt_text: str | None = None,
    generated_code: str | None = None,
    raw_response: str | None = None,
) -> None:
    """Successful LLM call."""
    if run_log is None:
        return
    run_log.record_llm_call(
        call_type=call_type,
        episode_id=episode_id,
        episode_num=episode_num,
        model=model,
        prompt_template_id=prompt_template_id,
        prompt_hash=prompt_hash,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        error=None,
        prompt_text=prompt_text,
        generated_code=generated_code,
        raw_response=raw_response,
    )


def log_llm_call_fail(
    run_log: RunLogger | None,
    *,
    call_type: str,
    error: BaseException | str,
    episode_id: str | None = None,
    episode_num: int | None = None,
    model: str | None = None,
    prompt_template_id: str | None = None,
    prompt_hash: str | None = None,
    latency_ms: int = 0,
) -> None:
    """Failed LLM call — tokens unknown, cost zero, error recorded."""
    if run_log is None:
        return
    run_log.record_llm_call(
        call_type=call_type,
        episode_id=episode_id,
        episode_num=episode_num,
        model=model,
        prompt_template_id=prompt_template_id,
        prompt_hash=prompt_hash,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        latency_ms=latency_ms,
        error=str(error),
    )


# ---------------------------------------------------------------------------
# Task attempts
# ---------------------------------------------------------------------------

def log_task_attempt(
    run_log: RunLogger | None,
    *,
    episode_num: int,
    task_name: str,
    task_description: str | None = None,
    achievement_key: str | None = None,
    skill_name: str | None = None,
    reused_skill: str | None = None,
    generated: bool = False,
    executor_reason: str | None = None,
    failure_reason: str | None = None,
    task_complete: bool = False,
    steps: int = 0,
    total_reward: float = 0.0,
    achievements_gained: list[str] | None = None,
    inventory: dict | None = None,
    achievements: dict | None = None,
    state_text: str | None = None,
    error_traceback: str | None = None,
) -> None:
    """Record one executed task/skill attempt."""
    if run_log is None:
        return
    record = getattr(run_log, "record_task_attempt", None)
    if record is None:
        return
    record(
        episode_num=episode_num,
        task_name=task_name,
        task_description=task_description,
        achievement_key=achievement_key,
        skill_name=skill_name,
        reused_skill=reused_skill,
        generated=generated,
        executor_reason=executor_reason,
        failure_reason=failure_reason,
        task_complete=task_complete,
        steps=steps,
        total_reward=total_reward,
        achievements_gained=achievements_gained,
        inventory=inventory,
        achievements=achievements,
        state_text=state_text,
        error_traceback=error_traceback,
    )


# ---------------------------------------------------------------------------
# Session finalize
# ---------------------------------------------------------------------------

def log_session_finalize(
    run_log: RunLogger | None,
    *,
    final_achievements: dict,
) -> None:
    """Mark the session with final achievement set before the context exits."""
    if run_log is None:
        return
    run_log.finalize(final_achievements=final_achievements)
