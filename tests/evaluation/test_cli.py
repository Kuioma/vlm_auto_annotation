import json
from pathlib import Path

from auto_annotation.cli import main

from .conftest import write_fixture


def test_cli_evaluate_prints_summary_json(tmp_path: Path, capsys):
    path = write_fixture(tmp_path, indices=(1,))
    assert main(["evaluate", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["decision"] == "report_only"
    assert captured.err == ""


def test_cli_fatal_error_is_nonzero_and_clear(tmp_path: Path, capsys):
    path = write_fixture(tmp_path, indices=(1, 2), annotation_indices=(1,))
    assert main(["evaluate", "--config", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evaluation failed:" in captured.err
    assert not (tmp_path / "evaluation-output").exists()

