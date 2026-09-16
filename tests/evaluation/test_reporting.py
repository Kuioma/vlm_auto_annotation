import json
from pathlib import Path

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.runner import run_evaluation

from .conftest import prediction, write_fixture


def test_outputs_are_deterministic_and_review_queue_prioritizes_missing(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), predictions=[prediction(1)])
    config = load_evaluation_config(path)
    summary = run_evaluation(config)
    output = config.output.root / summary.evaluation_id
    first = {item.name: item.read_bytes() for item in output.iterdir()}
    assert run_evaluation(config) == summary
    assert {item.name: item.read_bytes() for item in output.iterdir()} == first
    queue = [json.loads(line) for line in (output / "review_queue.jsonl").read_text().splitlines()]
    assert queue[0]["prediction_status"] == "missing_prediction"
    assert set(first) == {"summary.json", "episodes.jsonl", "failures.jsonl", "review_queue.jsonl", "report.md"}


def test_partial_markdown_is_prominently_marked(tmp_path: Path):
    path = write_fixture(tmp_path, indices=(1, 2), annotation_indices=(1,), mode="partial")
    config = load_evaluation_config(path)
    summary = run_evaluation(config)
    report = (config.output.root / summary.evaluation_id / "report.md").read_text()
    assert "PARTIAL EVALUATION" in report
    assert "1/2 complete" in report

