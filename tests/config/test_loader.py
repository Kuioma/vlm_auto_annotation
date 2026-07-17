from pathlib import Path

import pytest

from auto_annotation.config.loader import config_hash, load_run_config
from auto_annotation.config.models import RunConfig, VllmBackendConfig


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


def test_load_vllm_config_resolves_staging_root_and_round_trips(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        """
schema_path: schema.yaml
ontology_path: ontology.yaml
manifest_path: inputs.jsonl
artifact_root: artifacts
output_path: output.jsonl
backend:
  kind: vllm
  base_url: http://127.0.0.1:8000/v1
  model_id: Qwen/Qwen3.6-27B
  model_revision: 6a9e13bd
  vllm_version: 0.24.0
  timeout_s: 180
  max_completion_tokens: 4096
  boundary_max_completion_tokens: 512
  temperature: 0
  seed: 0
  enable_thinking: false
  media_staging_root: staging
  api_key_env: VLLM_API_KEY
""".strip(),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert isinstance(config.backend, VllmBackendConfig)
    assert config.backend.media_staging_root == tmp_path / "staging"
    assert str(config.backend.base_url) == "http://127.0.0.1:8000/v1"
    assert config.backend.api_key_env == "VLLM_API_KEY"
    round_tripped = RunConfig.model_validate(config.model_dump())
    assert round_tripped.model_dump() == config.model_dump()
    assert round_tripped.source_path is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_s", 0),
        ("max_completion_tokens", 0),
        ("boundary_max_completion_tokens", 0),
        ("temperature", -0.1),
        ("api_key_env", "not-valid-name"),
    ],
)
def test_invalid_vllm_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values = {
        "schema_path": "schema.yaml",
        "ontology_path": "ontology.yaml",
        "manifest_path": "inputs.jsonl",
        "artifact_root": "artifacts",
        "output_path": "output.jsonl",
        "backend": {
            "kind": "vllm",
            "base_url": "http://127.0.0.1:8000/v1",
            "model_id": "Qwen/Qwen3.6-27B",
            "model_revision": "revision",
            "vllm_version": "0.24.0",
            "media_staging_root": str(tmp_path / "staging"),
            field: value,
        },
    }

    with pytest.raises(ValueError):
        RunConfig.model_validate(values)
