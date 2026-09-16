import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from auto_annotation.cli import main
from auto_annotation.inference.base import GenerationRequest, GenerationResponse
from auto_annotation.inference.mock import ScriptedMockBackend
from auto_annotation.pipeline import runner as runner_module


FIXTURES = Path(__file__).parents[1] / "fixtures"
requires_media_tools = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")),
    reason="ffmpeg and ffprobe are required",
)


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1:r=30",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def _config(
    tmp_path: Path,
    video: Path,
    *,
    use_base_url_env: bool = False,
) -> Path:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "video-1",
                "video_uri": str(video),
                "dataset_id": "demo",
                "schema_version": "subtask-v1",
                "ontology_id": "kitchen-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    config = tmp_path / "run.yaml"
    base_url_source = (
        "base_url_env: DASHSCOPE_BASE_URL"
        if use_base_url_env
        else (
            "base_url: "
            "https://workspace-id.example.test/compatible-mode/v1"
        )
    )
    config.write_text(
        f"""
schema_path: {FIXTURES / 'schema/subtask-v1.yaml'}
ontology_path: {FIXTURES / 'ontology/kitchen-v1.yaml'}
manifest_path: {manifest}
artifact_root: {tmp_path / 'artifacts'}
output_path: {tmp_path / 'output.jsonl'}
backend:
  kind: openai_compatible
  {base_url_source}
  model_id: qwen3-vl-plus
  media_staging_root: {staging}
  api_key_env: DASHSCOPE_API_KEY
  timeout_s: 10
  max_completion_tokens: 4096
  boundary_max_completion_tokens: 512
  temperature: 0
""".strip(),
        encoding="utf-8",
    )
    return config


class FakeOpenAIBackend:
    def __init__(
        self,
        config: object,
        *,
        materializer: object,
        fail_first: bool = False,
    ) -> None:
        self.model_id = "qwen3-vl-plus"
        self.model_revision = "qwen3-vl-plus"
        self._delegate = ScriptedMockBackend(
            json.loads((FIXTURES / "mock/annotation.json").read_text())
        )
        self.fail_first = fail_first

    async def __aenter__(self) -> "FakeOpenAIBackend":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.fail_first:
            raise RuntimeError("synthetic completion failure")
        return await self._delegate.generate(request)


@requires_media_tools
def test_fake_openai_run_writes_identity_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://workspace-env.example.test/compatible-mode/v1",
    )
    config = _config(
        tmp_path,
        _video(tmp_path),
        use_base_url_env=True,
    )
    monkeypatch.setattr(runner_module, "OpenAICompatibleBackend", FakeOpenAIBackend)

    assert main(["run", "--config", str(config)]) == 0
    output = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    provenance = output["provenance"]
    assert provenance["backend_kind"] == "openai_compatible"
    assert provenance["backend_profile"] == "dashscope_qwen_vision"
    assert provenance["model_revision"] == "qwen3-vl-plus"
    assert provenance["base_url"].startswith("https://workspace-env.")
    assert "dashscope" in provenance["prompt_versions"]
    run_id = f"run-{provenance['config_hash'].removeprefix('sha256:')}"
    video_root = tmp_path / "artifacts" / run_id / "video-1"
    assert (video_root / "coarse-sample-plan.json").is_file()
    assert (video_root / "coarse.json").is_file()
    assert (video_root / "final.json").is_file()


@requires_media_tools
def test_base_url_env_changes_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        _video(tmp_path),
        use_base_url_env=True,
    )
    monkeypatch.setattr(
        runner_module,
        "OpenAICompatibleBackend",
        FakeOpenAIBackend,
    )
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://workspace-one.example.test/compatible-mode/v1",
    )
    assert main(["run", "--config", str(config)]) == 0
    first = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))

    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://workspace-two.example.test/compatible-mode/v1",
    )
    assert main(["run", "--config", str(config)]) == 0
    second = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))

    assert first["provenance"]["config_hash"] != second["provenance"]["config_hash"]
    assert first["provenance"]["base_url"].startswith("https://workspace-one.")
    assert second["provenance"]["base_url"].startswith("https://workspace-two.")


@requires_media_tools
def test_missing_base_url_env_has_no_persistent_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        tmp_path,
        _video(tmp_path),
        use_base_url_env=True,
    )
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.setattr(
        runner_module,
        "OpenAICompatibleBackend",
        FakeOpenAIBackend,
    )

    with pytest.raises(
        ValueError,
        match="missing base URL environment variable",
    ):
        main(["run", "--config", str(config)])
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "output.jsonl").exists()


@requires_media_tools
def test_fake_openai_first_failure_has_no_persistent_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, _video(tmp_path))

    class FailingBackend(FakeOpenAIBackend):
        def __init__(self, config: object, *, materializer: object) -> None:
            super().__init__(config, materializer=materializer, fail_first=True)

    monkeypatch.setattr(runner_module, "OpenAICompatibleBackend", FailingBackend)
    with pytest.raises(RuntimeError, match="synthetic completion failure"):
        main(["run", "--config", str(config)])
    assert not (tmp_path / "output.jsonl").exists()
    assert not (tmp_path / "artifacts").exists()
    assert list((tmp_path / "staging").iterdir()) == []
