from pathlib import Path
import os

import pytest

from auto_annotation.evaluation.config import load_evaluation_config
from auto_annotation.evaluation.errors import EvaluationError
from auto_annotation.evaluation.runner import run_evaluation
from auto_annotation.evaluation.store import publish_evaluation

from .conftest import write_fixture


def test_existing_corrupt_result_is_not_overwritten(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    summary = run_evaluation(config)
    summary_path = config.output.root / summary.evaluation_id / "summary.json"
    summary_path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(EvaluationError, match="differs"):
        run_evaluation(config)
    assert summary_path.read_text() == "corrupt"


def test_output_root_symlink_is_rejected_before_writes(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    real = tmp_path / "real-output"
    real.mkdir()
    config.output.root.symlink_to(real, target_is_directory=True)
    with pytest.raises(EvaluationError, match="symlink ancestor"):
        run_evaluation(config)
    assert not list(real.iterdir())


def test_output_root_cannot_be_inside_annotations(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"root": config.inputs.annotations_dir / "reports"})}
    )
    with pytest.raises(EvaluationError, match="inside an evaluator input directory"):
        run_evaluation(config)


def test_normalized_output_root_cannot_hide_inside_annotations(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    disguised = config.inputs.annotations_dir / ".." / "annotations" / "reports"
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"root": disguised})}
    )
    with pytest.raises(EvaluationError, match="inside an evaluator input directory"):
        run_evaluation(config)


def test_publish_failure_removes_staging_directory(tmp_path: Path, monkeypatch):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))

    def fail_rename(source, destination):
        raise OSError("injected rename failure")

    monkeypatch.setattr(
        "auto_annotation.evaluation.store._rename_directory_no_replace",
        fail_rename,
    )
    with pytest.raises(EvaluationError, match="cannot publish"):
        run_evaluation(config)
    assert config.output.root.exists()
    assert list(config.output.root.iterdir()) == []


def test_hardlink_target_alias_is_rejected(tmp_path: Path):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    summary = run_evaluation(config)
    target = config.output.root / summary.evaluation_id
    import shutil

    shutil.rmtree(target)
    os.link(config.inputs.prediction_path, target)
    with pytest.raises(EvaluationError, match="aliases an evaluator input"):
        run_evaluation(config)


def test_publish_race_does_not_replace_empty_target(
    tmp_path: Path, monkeypatch
):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    evaluation_id = f"eval-{'a' * 64}"
    target = config.output.root / evaluation_id
    files = {"summary.json": b"{}\n"}
    from auto_annotation.evaluation import store

    real_rename = store._rename_directory_no_replace
    target_inode = None

    def create_racing_target(source, destination):
        nonlocal target_inode
        destination.mkdir()
        target_inode = destination.stat().st_ino
        real_rename(source, destination)

    monkeypatch.setattr(store, "_rename_directory_no_replace", create_racing_target)
    with pytest.raises(EvaluationError, match="file set differs"):
        publish_evaluation(config, evaluation_id, files)
    assert target.stat().st_ino == target_inode
    assert list(target.iterdir()) == []
    assert not any(path.name.startswith(f".{evaluation_id}.") for path in config.output.root.iterdir())


def test_publish_race_reuses_byte_identical_completed_target(
    tmp_path: Path, monkeypatch
):
    config = load_evaluation_config(write_fixture(tmp_path, indices=(1,)))
    evaluation_id = f"eval-{'b' * 64}"
    target = config.output.root / evaluation_id
    files = {"summary.json": b"{}\n", "episodes.jsonl": b""}
    from auto_annotation.evaluation import store

    real_rename = store._rename_directory_no_replace
    target_inode = None

    def complete_racing_publish(source, destination):
        nonlocal target_inode
        destination.mkdir()
        for name, content in files.items():
            (destination / name).write_bytes(content)
        target_inode = destination.stat().st_ino
        real_rename(source, destination)

    monkeypatch.setattr(store, "_rename_directory_no_replace", complete_racing_publish)
    assert publish_evaluation(config, evaluation_id, files) == target
    assert target.stat().st_ino == target_inode
    assert {path.name: path.read_bytes() for path in target.iterdir()} == files
    assert not any(path.name.startswith(f".{evaluation_id}.") for path in config.output.root.iterdir())
