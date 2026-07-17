import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from auto_annotation.cli import main
from auto_annotation.config.models import VllmBackendConfig
from auto_annotation.inference.base import GenerationRequest, GenerationResponse
from auto_annotation.inference.mock import ScriptedMockBackend
from auto_annotation.pipeline import runner as runner_module


FIXTURES = Path(__file__).parents[1] / "fixtures"
requires_media_tools = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")),
    reason="ffmpeg and ffprobe are required",
)


def _make_video(tmp_path: Path) -> Path:
    video = tmp_path / "video.mp4"
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
            str(video),
        ],
        check=True,
    )
    return video


def _write_inputs(tmp_path: Path, video: Path) -> tuple[Path, Path, Path]:
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
    artifact_root = tmp_path / "artifacts"
    config = tmp_path / "run.yaml"
    config.write_text(
        f"""
schema_path: {FIXTURES / 'schema/subtask-v1.yaml'}
ontology_path: {FIXTURES / 'ontology/kitchen-v1.yaml'}
manifest_path: {manifest}
artifact_root: {artifact_root}
output_path: {tmp_path / 'output.jsonl'}
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
  media_staging_root: {staging}
""".strip(),
        encoding="utf-8",
    )
    return config, artifact_root, staging


class FakeVllmBackend:
    def __init__(
        self,
        config: VllmBackendConfig,
        script: dict[str, list[dict[str, Any]]],
        *,
        health_error: Exception | None = None,
    ) -> None:
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self._delegate = ScriptedMockBackend(script)
        self._health_error = health_error
        self.health_checked = False

    @property
    def calls(self) -> list[GenerationRequest]:
        return self._delegate.calls

    async def __aenter__(self) -> "FakeVllmBackend":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    async def healthcheck(self) -> None:
        self.health_checked = True
        if self._health_error is not None:
            raise self._health_error

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        assert self.health_checked
        return await self._delegate.generate(request)


def _install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    health_error: Exception | None = None,
) -> list[FakeVllmBackend]:
    script = json.loads(
        (FIXTURES / "mock/annotation.json").read_text(encoding="utf-8")
    )
    instances: list[FakeVllmBackend] = []

    def factory(
        config: VllmBackendConfig,
        *,
        materializer: object,
    ) -> FakeVllmBackend:
        backend = FakeVllmBackend(
            config,
            script,
            health_error=health_error,
        )
        instances.append(backend)
        return backend

    monkeypatch.setattr(runner_module, "VllmBackend", factory)
    return instances


@requires_media_tools
def test_cli_runs_vllm_backend_and_tracks_vllm_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, artifact_root, _ = _write_inputs(tmp_path, _make_video(tmp_path))
    instances = _install_fake_backend(monkeypatch)

    assert main(["run", "--config", str(config)]) == 0

    output_path = tmp_path / "output.jsonl"
    document = json.loads(output_path.read_text(encoding="utf-8"))
    baseline_digest = document["provenance"]["config_hash"]
    assert document["provenance"]["model_id"] == "Qwen/Qwen3.6-27B"
    assert document["provenance"]["model_revision"] == "6a9e13bd"
    assert instances[-1].health_checked
    assert [call.sampling_fps for call in instances[-1].calls] == [
        2.0,
        6.0,
        6.0,
    ]

    monkeypatch.setattr(
        runner_module,
        "VLLM_ADAPTER_VERSION",
        f"{runner_module.VLLM_ADAPTER_VERSION}-changed",
    )
    assert main(["run", "--config", str(config)]) == 0
    adapter_digest = json.loads(output_path.read_text(encoding="utf-8"))[
        "provenance"
    ]["config_hash"]
    assert adapter_digest != baseline_digest

    monkeypatch.setattr(
        runner_module,
        "MEDIA_MATERIALIZER_VERSION",
        f"{runner_module.MEDIA_MATERIALIZER_VERSION}-changed",
    )
    assert main(["run", "--config", str(config)]) == 0
    materializer_digest = json.loads(output_path.read_text(encoding="utf-8"))[
        "provenance"
    ]["config_hash"]
    assert materializer_digest not in {baseline_digest, adapter_digest}


@requires_media_tools
def test_vllm_healthcheck_fails_before_output_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, artifact_root, staging = _write_inputs(
        tmp_path, _make_video(tmp_path)
    )
    _install_fake_backend(
        monkeypatch,
        health_error=RuntimeError("endpoint unavailable"),
    )

    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        main(["run", "--config", str(config)])

    assert not artifact_root.exists()
    assert not (tmp_path / "output.jsonl").exists()
    assert list(staging.iterdir()) == []
