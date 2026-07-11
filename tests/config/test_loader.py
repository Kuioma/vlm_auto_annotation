from pathlib import Path

import pytest

from auto_annotation.config.loader import config_hash, load_run_config
from auto_annotation.config.models import RunConfig


def test_load_run_config_resolves_paths_and_has_stable_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
schema_path: schema.yaml
ontology_path: ontology.yaml
manifest_path: inputs.jsonl
artifact_root: artifacts
output_path: output.jsonl
backend:
  kind: mock
  fixture_path: mock.json
sampling:
  coarse_fps: 2.0
  refine_fps: 6.0
  refine_window_ms: 4000
timeline:
  merge_gap_ms: 250
  min_segment_ms: 300
""".strip(),
        encoding="utf-8",
    )

    first = load_run_config(config_path)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.mkdir()
    second = load_run_config(alias_parent / ".." / config_path.name)

    assert first.schema_path == tmp_path / "schema.yaml"
    assert first.backend.fixture_path == tmp_path / "mock.json"
    assert first.source_path == config_path.resolve()
    assert second.source_path == config_path.resolve()
    assert "source_path" not in first.model_dump()
    assert config_hash(first) == config_hash(second)
    assert config_hash(first) == config_hash(
        RunConfig.model_validate(first.model_dump())
    )
    assert config_hash(first, {"schema_hash": "a"}) != config_hash(
        first, {"schema_hash": "b"}
    )


def test_manually_constructed_run_config_has_no_source_path() -> None:
    config = RunConfig.model_validate(
        {
            "schema_path": "schema.yaml",
            "ontology_path": "ontology.yaml",
            "manifest_path": "inputs.jsonl",
            "artifact_root": "artifacts",
            "output_path": "output.jsonl",
            "backend": {"kind": "mock", "fixture_path": "mock.json"},
        }
    )

    assert config.source_path is None
    with pytest.raises(AttributeError, match="no setter"):
        config.source_path = Path("other.yaml")


def test_invalid_sampling_rate_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
schema_path: schema.yaml
ontology_path: ontology.yaml
manifest_path: inputs.jsonl
artifact_root: artifacts
output_path: output.jsonl
backend:
  kind: mock
  fixture_path: mock.json
sampling:
  coarse_fps: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="greater than 0"):
        load_run_config(config_path)
