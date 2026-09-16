import json
from pathlib import Path

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.runner import run_evaluation

from .conftest import write_fixture


def test_runner_publishes_traceable_report_only_result(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path))
    summary = run_evaluation(config)
    assert summary.evaluation_id.startswith("eval-")
    assert len(summary.evaluation_id) == 69
    assert summary.decision == "report_only"
    assert not hasattr(summary, "composite_score")
    output = config.output.root / summary.evaluation_id
    serialized = json.loads((output / "summary.json").read_text())
    assert serialized["evaluation_id"] == summary.evaluation_id
    assert serialized["eligibility"]["included"] == 2

