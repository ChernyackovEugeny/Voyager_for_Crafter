"""Crafter benchmark metrics."""
from __future__ import annotations

import math
from collections.abc import Iterable

from environment.achievements import ACHIEVEMENTS


def achievement_success_rates(
    achievements_per_episode: Iterable[dict[str, int | bool]],
) -> dict[str, float]:
    """Return per-achievement success rates in the range [0, 1]."""
    episodes = list(achievements_per_episode)
    if not episodes:
        return {key: 0.0 for key in ACHIEVEMENTS}

    total = len(episodes)
    return {
        key: sum(1 for episode in episodes if episode.get(key, 0)) / total
        for key in ACHIEVEMENTS
    }


def crafter_score_from_success_rates(success_rates: dict[str, float]) -> float:
    """Canonical Crafter score as a fraction.

    The reference implementation computes:
        exp(mean(log(1 + percent_i))) - 1
    where percent_i is each achievement success rate in [0, 100].

    This function accepts rates in [0, 1] and returns the score in [0, 1].
    """
    if not success_rates:
        return 0.0

    values = []
    for key in ACHIEVEMENTS:
        rate = float(success_rates.get(key, 0.0))
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"success rate for {key!r} must be in [0, 1]")
        values.append(math.log1p(100.0 * rate))

    score_percent = math.exp(sum(values) / len(values)) - 1.0
    return score_percent / 100.0


def crafter_score(
    achievements_per_episode: Iterable[dict[str, int | bool]],
) -> float:
    """Canonical Crafter score for episode achievement flags."""
    return crafter_score_from_success_rates(
        achievement_success_rates(achievements_per_episode)
    )
