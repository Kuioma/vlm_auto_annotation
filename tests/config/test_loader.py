import json
from pathlib import Path

import pytest

from auto_annotation.config.loader import config_hash, load_run_config
from auto_annotation.config.models import (
    OpenAICompatibleBackendConfig,
    RunConfig,
    VllmBackendConfig,
)


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


def test_load_dashscope_example_resolves_staging_and_revision_fallback() -> None:
    example = (
        Path(__file__).parents[2]
        / "examples"
        / "dashscope_openai_compatible"
        / "run.yaml"
    )
    config = load_run_config(example)
    assert isinstance(config.backend, OpenAICompatibleBackendConfig)
    assert config.backend.media_staging_root == example.parent / "staging"
    assert config.backend.effective_model_revision == config.backend.model_id
    assert config.backend.base_url is None
    assert config.backend.base_url_env == "DASHSCOPE_BASE_URL"


def test_dashscope_base_url_env_resolves_without_entering_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://workspace-env.example.test/compatible-mode/v1",
    )
    config = OpenAICompatibleBackendConfig(
        kind="openai_compatible",
        base_url_env="DASHSCOPE_BASE_URL",
        model_id="qwen3-vl-plus",
        media_staging_root=tmp_path,
        api_key_env="DASHSCOPE_API_KEY",
    )

    assert str(config.effective_base_url) == (
        "https://workspace-env.example.test/compatible-mode/v1"
    )
    dumped = config.model_dump(mode="json")
    assert dumped["base_url"] is None
    assert dumped["base_url_env"] == "DASHSCOPE_BASE_URL"
    assert "workspace-env" not in json.dumps(dumped)


@pytest.mark.parametrize(
    "backend_update",
    [
        {},
        {
            "base_url": "https://workspace-id.example.test/v1",
            "base_url_env": "DASHSCOPE_BASE_URL",
        },
    ],
)
def test_dashscope_requires_exactly_one_base_url_source(
    tmp_path: Path,
    backend_update: dict[str, str],
) -> None:
    backend = {
        "kind": "openai_compatible",
        "model_id": "qwen3-vl-plus",
        "media_staging_root": str(tmp_path / "staging"),
        "api_key_env": "DASHSCOPE_API_KEY",
        **backend_update,
    }
    with pytest.raises(ValueError, match="exactly one"):
        OpenAICompatibleBackendConfig.model_validate(backend)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "missing base URL environment variable"),
        ("", "missing base URL environment variable"),
        ("not a URL", "invalid base URL environment variable"),
    ],
)
def test_dashscope_rejects_missing_or_invalid_base_url_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    message: str,
) -> None:
    if value is None:
        monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("DASHSCOPE_BASE_URL", value)
    config = OpenAICompatibleBackendConfig(
        kind="openai_compatible",
        base_url_env="DASHSCOPE_BASE_URL",
        model_id="qwen3-vl-plus",
        media_staging_root=tmp_path,
        api_key_env="DASHSCOPE_API_KEY",
    )

    with pytest.raises(ValueError, match=message):
        _ = config.effective_base_url


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout_s", 0),
        ("temperature", 2),
        ("api_key_env", "bad-name"),
        ("base_url_env", "bad-name"),
    ],
)
def test_invalid_dashscope_config_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    backend = {
        "kind": "openai_compatible",
        "base_url": "https://workspace-id.example.test/v1",
        "model_id": "qwen3-vl-plus",
        "media_staging_root": "staging",
        "api_key_env": "DASHSCOPE_API_KEY",
        field: value,
    }
    if field == "base_url_env":
        backend.pop("base_url")
    values = {
        "schema_path": "schema.yaml",
        "ontology_path": "ontology.yaml",
        "manifest_path": "inputs.jsonl",
        "artifact_root": "artifacts",
        "output_path": "output.jsonl",
        "backend": backend,
    }
    with pytest.raises(ValueError):
        RunConfig.model_validate(values)
