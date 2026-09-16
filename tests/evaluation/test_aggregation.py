from pathlib import Path

import pytest

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.aggregation import (
    aggregate_boundary,
    aggregate_temporal_iou,
)
from auto_annotation.evaluation.models import EpisodeEvaluation
from auto_annotation.evaluation.runner import run_evaluation

from .conftest import prediction, write_fixture


def test_missing_prediction_counts_as_hit_miss_and_tiou_zero(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), predictions=[prediction(1)])
    summary = run_evaluation(load_evaluation_config(path))
    boundary = summary.metrics["boundary_error"]
    tiou = summary.metrics["temporal_iou"]
    assert boundary.denominator == {"episodes": 2, "gold_boundaries": 4, "observed_boundaries": 2}
    assert boundary.missing_count == 2
    assert boundary.aggregation["micro"]["hit_rate"]["1000"] == 0.5
    assert tiou.aggregation["micro"]["mean_tiou"] < tiou.aggregation["macro_episode"]["mean_tiou"] + 1e-12
    assert summary.prediction_coverage["rate"] == 0.5


def test_disabled_and_deferred_statuses_are_explicit(tmp_path: Path):
    summary = run_evaluation(load_evaluation_config(write_fixture(tmp_path, indices=(1,))))
    assert summary.metrics["sequence_edit_distance"].status == "skipped"
    assert summary.metrics["semantic_quality"].status == "not_implemented"
    assert summary.metrics["gold_reliability"].status == "not_implemented"


def boundary_detail(absolute_error):
    return {
        "gold_ms": 1000,
        "predicted_ms": (
            None if absolute_error is None else 1000 + absolute_error
        ),
        "signed_error_ms": absolute_error,
        "absolute_error_ms": absolute_error,
        "hits": {"100": absolute_error is not None and absolute_error <= 100,
                 "500": absolute_error is not None and absolute_error <= 500,
                 "1000": absolute_error is not None and absolute_error <= 1000},
        "status": "missing" if absolute_error is None else "observed",
    }


def test_macro_and_micro_diverge_with_unequal_observed_boundaries_and_segments(
    tmp_path: Path,
):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    thresholds = {str(value): {"tp": 0, "fp": 0, "fn": 0} for value in [0.3, 0.5, 0.7]}
    first_counts = {
        key: {"tp": 1, "fp": 0, "fn": 0} for key in thresholds
    }
    second_counts = {
        key: {"tp": 0, "fp": 0, "fn": 3} for key in thresholds
    }
    episodes = [
        EpisodeEvaluation(
            episode_index=1,
            video_id="one",
            status="evaluated",
            provenance_status="verified",
            boundaries={
                "transfer": boundary_detail(0),
                "place": boundary_detail(100),
            },
            segments={
                "only": {
                    "gold": [0, 1000],
                    "prediction": [0, 1000],
                    "tiou": 1.0,
                    "status": "matched",
                }
            },
            temporal_counts=first_counts,
            mean_tiou=1.0,
        ),
        EpisodeEvaluation(
            episode_index=2,
            video_id="two",
            status="missing_prediction",
            provenance_status="missing",
            boundaries={
                "transfer": boundary_detail(1000),
                "place": boundary_detail(None),
            },
            segments={
                segment_id: {
                    "gold": [index * 1000, (index + 1) * 1000],
                    "prediction": None,
                    "tiou": 0.0,
                    "status": "missing",
                }
                for index, segment_id in enumerate(["a", "b", "c"])
            },
            temporal_counts=second_counts,
            mean_tiou=0.0,
        ),
    ]
    boundary = aggregate_boundary(config, episodes)
    tiou = aggregate_temporal_iou(config, episodes)
    assert boundary.aggregation["macro_episode"]["mae_ms"] == 525.0
    assert boundary.aggregation["micro"]["mae_ms"] == pytest.approx(
        1100 / 3
    )
    assert tiou.aggregation["macro_episode"]["mean_tiou"] == 0.5
    assert tiou.aggregation["micro"]["mean_tiou"] == 0.25
