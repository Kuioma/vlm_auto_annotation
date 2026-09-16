import pytest
from pydantic import ValidationError

from auto_annotation.evaluation.models import (
    EligibilitySummary,
    EvalSegment,
    EvaluationSummary,
    MetricResult,
)


def test_segment_uses_positive_half_open_interval():
    with pytest.raises(ValidationError, match="greater than start_ms"):
        EvalSegment(segment_id="x", start_ms=10, end_ms=10, values={}, sentence="x")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metric_result_rejects_nested_non_finite_numbers(value):
    with pytest.raises(ValidationError, match="non-finite number"):
        MetricResult(
            status="computed",
            version="metric-v1",
            aggregation={"nested": [{"value": value}]},
        )


def summary_payload(value):
    return {
        "result_schema_version": "result-v1",
        "evaluator_version": "evaluator-v1",
        "evaluation_id": f"eval-{'a' * 64}",
        "profile_id": "profile-v1",
        "evaluation_mode": "official",
        "report_label": "OFFICIAL",
        "config_sha256": "sha256:config",
        "gold_manifest_sha256": "sha256:gold",
        "included_annotations_sha256": "sha256:annotations",
        "prediction_sha256": "sha256:predictions",
        "metric_versions": {},
        "eligibility": EligibilitySummary(
            selected=1,
            complete=1,
            included=1,
            excluded=0,
            incomplete=0,
            predicted=1,
            missing_predictions=0,
            invalid_predictions=0,
            extra_predictions=0,
        ),
        "prediction_coverage": {"rate": value},
        "prediction_schema_validity": {"rate": 1.0},
        "metrics": {},
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_evaluation_summary_rejects_nested_non_finite_numbers(value):
    with pytest.raises(ValidationError, match="non-finite number"):
        EvaluationSummary.model_validate(summary_payload(value))


def test_result_models_allow_finite_numbers_and_nulls():
    result = MetricResult(
        status="computed",
        version="metric-v1",
        aggregation={"values": [0.0, -1.5, None]},
    )
    assert result.aggregation == {"values": [0.0, -1.5, None]}
