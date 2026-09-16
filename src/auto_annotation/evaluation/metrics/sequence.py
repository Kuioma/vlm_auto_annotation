"""Ordered token-sequence edit distance."""

from __future__ import annotations

from typing import Any

from auto_annotation.evaluation.models import EvalSegment


def segment_tokens(segments: list[EvalSegment] | None, field: str) -> list[Any]:
    return [segment.values.get(field) for segment in (segments or [])]


def levenshtein(left: list[Any], right: list[Any]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def evaluate_sequence(
    gold_segments: list[EvalSegment],
    prediction_segments: list[EvalSegment] | None,
    field: str,
) -> dict[str, Any]:
    gold = segment_tokens(gold_segments, field)
    prediction = segment_tokens(prediction_segments, field)
    distance = levenshtein(gold, prediction)
    return {
        "gold": gold,
        "prediction": prediction,
        "distance": distance,
        "normalized_distance": distance / max(len(gold), len(prediction), 1),
        "exact": gold == prediction,
        "gold_length": len(gold),
        "prediction_length": len(prediction),
    }

