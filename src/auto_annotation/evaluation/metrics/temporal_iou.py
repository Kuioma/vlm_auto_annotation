"""Temporal-IoU matching and threshold counts."""

from __future__ import annotations

from auto_annotation.evaluation.models import EvalSegment


def temporal_iou(gold: EvalSegment, prediction: EvalSegment) -> float:
    intersection = max(
        0,
        min(gold.end_ms, prediction.end_ms)
        - max(gold.start_ms, prediction.start_ms),
    )
    union = (
        gold.end_ms
        - gold.start_ms
        + prediction.end_ms
        - prediction.start_ms
        - intersection
    )
    return 0.0 if union <= 0 else intersection / union


def evaluate_temporal_iou(
    gold_segments: list[EvalSegment],
    prediction_segments: list[EvalSegment] | None,
    thresholds: list[float],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, int]], list[str]]:
    predicted_by_id = {
        segment.segment_id: segment for segment in (prediction_segments or [])
    }
    gold_ids = {segment.segment_id for segment in gold_segments}
    details: dict[str, dict[str, object]] = {}
    scores: dict[str, float] = {}
    for gold in gold_segments:
        prediction = predicted_by_id.get(gold.segment_id)
        score = 0.0 if prediction is None else temporal_iou(gold, prediction)
        scores[gold.segment_id] = score
        details[gold.segment_id] = {
            "gold": [gold.start_ms, gold.end_ms],
            "prediction": (
                None if prediction is None else [prediction.start_ms, prediction.end_ms]
            ),
            "tiou": score,
            "status": "missing" if prediction is None else "matched",
        }
    extras = sorted(set(predicted_by_id) - gold_ids)
    counts: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        tp = sum(
            detail["status"] == "matched" and float(detail["tiou"]) >= threshold
            for detail in details.values()
        )
        low_matched = sum(
            detail["status"] == "matched" and float(detail["tiou"]) < threshold
            for detail in details.values()
        )
        counts[str(threshold)] = {
            "tp": tp,
            "fp": low_matched + len(extras),
            "fn": len(scores) - tp,
        }
    return details, counts, extras
