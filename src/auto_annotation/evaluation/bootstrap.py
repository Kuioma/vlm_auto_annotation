"""Deterministic episode-resampling confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from auto_annotation.evaluation.models import ConfidenceInterval, EpisodeEvaluation


def quantile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def episode_bootstrap(
    episodes: Sequence[EpisodeEvaluation],
    statistic: Callable[[Sequence[EpisodeEvaluation]], float | None],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> ConfidenceInterval:
    if not episodes:
        return ConfidenceInterval(
            status="unavailable",
            confidence_level=confidence_level,
            reason="no included episodes",
        )
    generator = random.Random(seed)
    values: list[float] = []
    count = len(episodes)
    for _ in range(resamples):
        sample = [episodes[generator.randrange(count)] for _ in range(count)]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            values.append(value)
    alpha = (1 - confidence_level) * 100
    lower = quantile(values, alpha / 2)
    upper = quantile(values, 100 - alpha / 2)
    return ConfidenceInterval(
        status="computed" if values else "unavailable",
        confidence_level=confidence_level,
        lower=lower,
        upper=upper,
        valid_resamples=len(values),
        reason=None if values else "statistic was undefined in every resample",
    )


def suppressed_interval(reason: str) -> ConfidenceInterval:
    return ConfidenceInterval(status="not_applicable", reason=reason)

