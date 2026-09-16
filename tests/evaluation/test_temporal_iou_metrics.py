import pytest

from auto_annotation.evaluation.metrics.temporal_iou import evaluate_temporal_iou, temporal_iou
from auto_annotation.evaluation.models import EvalSegment


def make(segment_id, start, end):
    return EvalSegment(segment_id=segment_id, start_ms=start, end_ms=end, values={}, sentence=segment_id)


def test_half_open_tiou():
    assert temporal_iou(make("x", 0, 1000), make("x", 500, 1500)) == pytest.approx(1 / 3)
    assert temporal_iou(make("x", 0, 1000), make("x", 1000, 2000)) == 0.0


def test_missing_extra_and_low_match_counts():
    details, counts, extras = evaluate_temporal_iou(
        [make("a", 0, 1000), make("b", 1000, 2000)],
        [make("a", 900, 1900), make("extra", 2000, 2500)],
        [0.5],
    )
    assert details["b"]["tiou"] == 0.0
    assert extras == ["extra"]
    assert counts["0.5"] == {"tp": 0, "fp": 2, "fn": 2}
