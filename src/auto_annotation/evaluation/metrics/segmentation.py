"""Structural split/merge metrics based on substantial temporal overlap."""

from __future__ import annotations

from typing import Any

from auto_annotation.evaluation.models import EvalSegment


def substantial_overlap(gold: EvalSegment, prediction: EvalSegment) -> float:
    intersection = max(
        0,
        min(gold.end_ms, prediction.end_ms)
        - max(gold.start_ms, prediction.start_ms),
    )
    shorter = min(
        gold.end_ms - gold.start_ms,
        prediction.end_ms - prediction.start_ms,
    )
    return 0.0 if shorter <= 0 else intersection / shorter


def evaluate_segmentation(
    gold_segments: list[EvalSegment],
    prediction_segments: list[EvalSegment] | None,
    threshold: float,
) -> dict[str, Any]:
    predictions = prediction_segments or []
    gold_degree = {segment.segment_id: 0 for segment in gold_segments}
    prediction_degree = {segment.segment_id: 0 for segment in predictions}
    edges: list[dict[str, Any]] = []
    for gold in gold_segments:
        for prediction in predictions:
            score = substantial_overlap(gold, prediction)
            if score >= threshold:
                gold_degree[gold.segment_id] += 1
                prediction_degree[prediction.segment_id] += 1
                edges.append(
                    {
                        "gold_segment_id": gold.segment_id,
                        "prediction_segment_id": prediction.segment_id,
                        "overlap_score": score,
                    }
                )
    return {
        "missed_segment_count": sum(degree == 0 for degree in gold_degree.values()),
        "extra_segment_count": sum(degree == 0 for degree in prediction_degree.values()),
        "over_split_count": sum(max(0, degree - 1) for degree in gold_degree.values()),
        "incorrect_merge_count": sum(
            max(0, degree - 1) for degree in prediction_degree.values()
        ),
        "gold_degree": gold_degree,
        "prediction_degree": prediction_degree,
        "edges": edges,
    }

