"""Boundary timing metrics."""

from __future__ import annotations


def evaluate_boundaries(
    gold: dict[str, int],
    prediction: dict[str, int | None] | None,
    thresholds_ms: list[int],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for event_id, gold_ms in gold.items():
        predicted_ms = None if prediction is None else prediction.get(event_id)
        signed = None if predicted_ms is None else predicted_ms - gold_ms
        absolute = None if signed is None else abs(signed)
        result[event_id] = {
            "gold_ms": gold_ms,
            "predicted_ms": predicted_ms,
            "signed_error_ms": signed,
            "absolute_error_ms": absolute,
            "hits": {
                str(threshold): absolute is not None and absolute <= threshold
                for threshold in thresholds_ms
            },
            "status": "missing" if predicted_ms is None else "observed",
        }
    return result

