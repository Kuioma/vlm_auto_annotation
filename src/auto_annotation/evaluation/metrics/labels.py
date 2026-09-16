"""Segment-ID paired structured-label accuracy."""

from __future__ import annotations

from typing import Any

from auto_annotation.evaluation.models import EvalSegment


def evaluate_labels(
    gold_segments: list[EvalSegment],
    prediction_segments: list[EvalSegment] | None,
    fields: list[str],
    exact_match_groups: list[list[str]],
) -> dict[str, Any]:
    predictions = {
        segment.segment_id: segment for segment in (prediction_segments or [])
    }
    gold_ids = {segment.segment_id for segment in gold_segments}
    field_counts = {field: {"correct": 0, "total": len(gold_segments)} for field in fields}
    group_counts = {
        "+".join(group): {"correct": 0, "total": len(gold_segments)}
        for group in exact_match_groups
    }
    missing_pairs = 0
    pairs: dict[str, Any] = {}
    for gold in gold_segments:
        prediction = predictions.get(gold.segment_id)
        if prediction is None:
            missing_pairs += 1
        field_results: dict[str, bool] = {}
        for field in fields:
            correct = prediction is not None and prediction.values.get(field) == gold.values.get(field)
            field_results[field] = correct
            field_counts[field]["correct"] += int(correct)
        group_results: dict[str, bool] = {}
        for group in exact_match_groups:
            key = "+".join(group)
            correct = prediction is not None and all(
                prediction.values.get(field) == gold.values.get(field)
                for field in group
            )
            group_results[key] = correct
            group_counts[key]["correct"] += int(correct)
        pairs[gold.segment_id] = {
            "matched": prediction is not None,
            "fields": field_results,
            "groups": group_results,
        }
    for counts in [*field_counts.values(), *group_counts.values()]:
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else 0.0
    return {
        "fields": field_counts,
        "groups": group_counts,
        "missing_pairs": missing_pairs,
        "extra_predictions": len(set(predictions) - gold_ids),
        "pairs": pairs,
    }

