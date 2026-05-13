from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """DeepSeek API prices in USD per 1M tokens."""

    input_cache_hit: float
    input_cache_miss: float
    output: float


_PRICING_PER_MTOK: dict[str, ModelPricing] = {
    "deepseek-v4-flash": ModelPricing(
        input_cache_hit=0.0028,
        input_cache_miss=0.14,
        output=0.28,
    ),
    # Promotional price listed by DeepSeek until 2026-05-31 15:59 UTC.
    "deepseek-v4-pro": ModelPricing(
        input_cache_hit=0.003625,
        input_cache_miss=0.435,
        output=0.87,
    ),
    "deepseek-reasoner": ModelPricing(
        input_cache_hit=0.14,
        input_cache_miss=0.55,
        output=2.19,
    ),
}

_MODEL_ALIASES: dict[str, str] = {
    "deepseek-chat": "deepseek-v4-flash",
}


def normalize_model_name(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def compute_cost(
    model: str,
    *,
    prompt_cache_hit_tokens: int,
    prompt_cache_miss_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute DeepSeek API cost using cache hit/miss token accounting."""
    normalized = normalize_model_name(model)
    pricing = _PRICING_PER_MTOK[normalized]
    return (
        prompt_cache_hit_tokens / 1_000_000 * pricing.input_cache_hit
        + prompt_cache_miss_tokens / 1_000_000 * pricing.input_cache_miss
        + completion_tokens / 1_000_000 * pricing.output
    )
