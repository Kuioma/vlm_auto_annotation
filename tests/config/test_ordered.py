from pathlib import Path

import pytest
import yaml

from auto_annotation.ordered_processor import parse_args


def write_config(tmp_path: Path, **updates: object) -> Path:
    payload = {
        "video": "videos/head.mp4",
        "right_wrist_video": "videos/wrist.mp4",
        "output_dir": "output",
        "schema_path": "schema.yaml",
        "ontology_path": "ontology.yaml",
        "task_path": "task.yaml",
        **updates,
    }
    path = tmp_path / "ordered.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_yaml_matches_cli_and_resolves_paths(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    actual = vars(parse_args(["--config", str(config)]))
    expected = vars(parse_args([
        "--video", str(tmp_path / "videos/head.mp4"),
        "--right-wrist-video", str(tmp_path / "videos/wrist.mp4"),
        "--output-dir", str(tmp_path / "output"),
        "--schema-path", str(tmp_path / "schema.yaml"),
        "--ontology-path", str(tmp_path / "ontology.yaml"),
        "--task-path", str(tmp_path / "task.yaml"),
    ]))
    actual.pop("config")
    expected.pop("config")
    assert actual == expected


def test_api_yaml_and_cli_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_API_ENDPOINT", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    config = write_config(tmp_path, backend={
        "kind": "openai_compatible",
        "base_url_env": "TEST_API_ENDPOINT",
        "api_key_env": "TEST_API_KEY",
        "model_id": "qwen3-vl-plus",
        "media_staging_root": "staging",
        "timeout_s": 180,
    }, sampling={"coarse_fps": 1, "refine_fps": 5, "refine_window_ms": 1000})
    args = parse_args([
        "--config", str(config), "--fps", "2", "--prepare-only",
        "--output-dir", "cli-output", "--model-id", "override-model",
    ])
    assert args.backend_kind == "openai_compatible"
    assert args.base_url_env == "TEST_API_ENDPOINT"
    assert args.api_key_env == "TEST_API_KEY"
    assert args.media_staging_root == tmp_path / "staging"
    assert args.model_id == "override-model"
    assert args.output_dir == Path("cli-output")
    assert args.fps == 2
    assert args.refine_fps == 5
    assert args.refine_window_ms == 1000
    assert args.prepare_only is True
    assert args.timeout_s == 180


@pytest.mark.parametrize("updates", [
    {"typo": True},
    {"backend": {"typo": True}},
    {"backend": {"kind": "unknown"}},
    {"sampling": {"coarse_fps": 0}},
    {"sampling": {"coarse_fps": float("inf")}},
    {"sampling": {"refine_fps": 1}},
    {"sampling": {"refine_window_ms": 1.5}},
    {"columns": 0},
])
def test_invalid_yaml_rejected(tmp_path: Path, updates: dict[str, object]) -> None:
    config = write_config(tmp_path, **updates)
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
    assert not (tmp_path / "output").exists()


def test_malformed_yaml_reports_cli_error(tmp_path: Path) -> None:
    config = tmp_path / "broken.yaml"
    config.write_text("video: [", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])


def test_cli_still_requires_inputs() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_input_mode_yaml_and_override(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    assert parse_args(["--config", str(config)]).input_mode == "direct"
    config = write_config(tmp_path, input_mode="wa2")
    assert parse_args(["--config", str(config)]).input_mode == "wa2"
    assert parse_args(["--config", str(config), "--input-mode", "direct"]).input_mode == "direct"
    config = write_config(tmp_path, input_mode="auto")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
