from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from auto_annotation.evaluation.config import (
    EvaluationConfig,
    evaluation_config_hash,
    load_evaluation_config,
)

from .conftest import config_payload


def test_config_resolves_relative_paths_and_hash_is_stable(tmp_path: Path):
    payload = config_payload(tmp_path)
    payload["output"]["root"] = "reports"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    first = load_evaluation_config(path)
    second = load_evaluation_config(path)
    assert first.inputs.gold_set_path == tmp_path / "gold_set.json"
    assert first.output.root == tmp_path / "reports"
    assert first.source_path == path
    assert "source_path" not in first.model_dump()
    assert evaluation_config_hash(first) == evaluation_config_hash(second)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unknown": True}), "Extra inputs"),
        (lambda value: value["metrics"]["boundary_error"].update({"thresholds_ms": [500, 100]}), "strictly increasing"),
        (lambda value: value["metrics"]["boundary_error"].update({"thresholds_ms": [500.0]}), "valid integer"),
        (lambda value: value["metrics"]["temporal_iou"].update({"thresholds": [0.0]}), "tIoU thresholds"),
        (lambda value: value["decision"].update({"mode": "threshold"}), "report_only"),
    ],
)
def test_config_rejects_unknown_or_invalid_values(tmp_path: Path, mutation, message):
    payload = config_payload(tmp_path)
    mutation(payload)
    with pytest.raises(ValidationError, match=message):
        EvaluationConfig.model_validate(payload)
