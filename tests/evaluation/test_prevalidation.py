import json
from pathlib import Path

import pytest

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.errors import EvaluationError
from auto_annotation.evaluation.inputs import load_and_prevalidate_inputs
from auto_annotation.evaluation.runner import run_evaluation

from .conftest import annotation, prediction, write_fixture


def test_official_incomplete_is_fatal_before_output(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), annotation_indices=(1,))
    config = load_evaluation_config(path)
    with pytest.raises(EvaluationError, match="official evaluation requires"):
        run_evaluation(config)
    assert not config.output.root.exists()


def test_partial_allows_incomplete(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), annotation_indices=(1,), mode="partial")
    summary = run_evaluation(load_evaluation_config(path))
    assert summary.report_label == "PARTIAL"
    assert summary.eligibility.incomplete == 1
    assert summary.metrics["boundary_error"].confidence_intervals["macro_episode.mae_ms"].status == "not_applicable"


def test_duration_mismatch_is_fatal(tmp_path: Path):
    item = prediction(1)
    item["duration_ms"] = 2999
    path = write_fixture(tmp_path, indices=(1,), predictions=[item])
    with pytest.raises(EvaluationError, match="duration mismatch"):
        load_and_prevalidate_inputs(load_evaluation_config(path))


def test_duration_mismatch_remains_fatal_when_prediction_schema_is_invalid(
    tmp_path: Path,
):
    item = prediction(1)
    item["duration_ms"] = 2999
    item["task_summary"] = ""
    path = write_fixture(tmp_path, indices=(1,), predictions=[item])
    config = load_evaluation_config(path)
    with pytest.raises(EvaluationError, match="duration mismatch"):
        run_evaluation(config)
    assert not config.output.root.exists()


def test_event_segment_mismatch_makes_prediction_invalid(tmp_path: Path):
    item = prediction(1)
    item["events"][0]["timestamp_ms"] += 1
    path = write_fixture(tmp_path, indices=(1,), predictions=[item])
    loaded = load_and_prevalidate_inputs(load_evaluation_config(path))
    assert "episode_000001_multiview" in loaded.invalid_predictions


def test_prediction_event_falls_back_to_segment_start(tmp_path: Path):
    item = prediction(1)
    item.pop("events")
    path = write_fixture(tmp_path, indices=(1,), predictions=[item])
    loaded = load_and_prevalidate_inputs(load_evaluation_config(path))
    assert loaded.predictions["episode_000001_multiview"].events == []


def test_explicit_exclusion_is_allowed_in_official_mode(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), predictions=[prediction(1)])
    excluded = annotation(2, exclusion_reason="unusable camera view")
    (tmp_path / "annotations" / "arbitrary-2.json").write_text(
        json.dumps(excluded), encoding="utf-8"
    )
    summary = run_evaluation(load_evaluation_config(path))
    assert summary.eligibility.excluded == 1
    assert summary.eligibility.incomplete == 0
    assert summary.eligibility.included == 1


def test_exclusion_reasons_are_counted_stably_in_summary(tmp_path: Path):
    path = write_fixture(
        tmp_path,
        indices=(1, 2, 3, 4),
        predictions=[prediction(1)],
    )
    reasons = {2: "camera occluded", 3: "wrong task", 4: "camera occluded"}
    for index, reason in reasons.items():
        (tmp_path / "annotations" / f"arbitrary-{index}.json").write_text(
            json.dumps(annotation(index, exclusion_reason=reason)),
            encoding="utf-8",
        )
    summary = run_evaluation(load_evaluation_config(path))
    assert summary.eligibility.excluded == 3
    assert summary.eligibility.exclusion_reasons == {
        "camera occluded": 2,
        "wrong task": 1,
    }
    assert list(summary.eligibility.exclusion_reasons) == [
        "camera occluded",
        "wrong task",
    ]
