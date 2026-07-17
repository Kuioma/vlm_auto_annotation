import asyncio
import json
import subprocess
from pathlib import Path
from uuid import UUID

import httpx
import pytest

from auto_annotation.config.models import VllmBackendConfig
from auto_annotation.inference.base import GenerationRequest
from auto_annotation.inference.vllm import (
    VllmBackend,
    VllmBackendError,
    VllmResponseError,
    VllmTransportError,
)
from auto_annotation.media.materialize import LocalVideoMaterializer


def _config(tmp_path: Path, **overrides: object) -> VllmBackendConfig:
    values = {
        "kind": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "model_id": "Qwen/Qwen3.6-27B",
        "model_revision": "revision",
        "vllm_version": "0.24.0",
        "timeout_s": 180,
        "max_completion_tokens": 4096,
        "boundary_max_completion_tokens": 512,
        "temperature": 0,
        "seed": 0,
        "enable_thinking": False,
        "media_staging_root": tmp_path / "staging",
    }
    values.update(overrides)
    return VllmBackendConfig.model_validate(values)


def _request(source: Path, **overrides: object) -> GenerationRequest:
    values = {
        "request_id": "coarse:video-1",
        "stage": "coarse",
        "media_uri": source.as_uri(),
        "media_start_ms": 1000,
        "media_end_ms": 5000,
        "sample_timestamps_ms": (1000, 1500, 2000),
        "sampling_fps": 2.0,
        "prompt": "annotate",
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"status": {"const": "ok"}},
            "required": ["status"],
        },
    }
    values.update(overrides)
    return GenerationRequest.model_validate(values)


def _materializer(tmp_path: Path) -> LocalVideoMaterializer:
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"materialized video")
        return subprocess.CompletedProcess(command, 0, "", "")

    return LocalVideoMaterializer(staging, runner)


def test_vllm_backend_healthcheck_and_generate_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        requests.append((request.method, request.url.path, payload))
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "Qwen/Qwen3.6-27B"}]},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"status": "ok"})}}
                ]
            },
        )

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            await backend.healthcheck()
            response = await backend.generate(_request(source))
            assert response.content == {"status": "ok"}

    asyncio.run(exercise())

    assert requests[0] == ("GET", "/v1/models", None)
    method, path, payload = requests[1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert payload is not None
    assert payload["model"] == "Qwen/Qwen3.6-27B"
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 0
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["media_io_kwargs"] == {"video": {"fps": 2.0}}
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "auto_annotation_response",
            "schema": _request(source).response_schema,
        },
    }
    message_content = payload["messages"][0]["content"]
    assert message_content[0]["type"] == "video_url"
    assert message_content[0]["video_url"]["url"].startswith("file://")
    assert UUID(message_content[0]["uuid"]).version == 4
    assert message_content[1] == {"type": "text", "text": "annotate"}
    assert list((tmp_path / "staging").iterdir()) == []


def test_vllm_backend_generate_with_numbered_frame_sheet(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    frame_sheet = tmp_path / "contact-sheet.jpg"
    frame_sheet.write_bytes(b"image")
    seen_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"status": "ok"})}}
                ]
            },
        )

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            response = await backend.generate_with_image(
                _request(source),
                frame_sheet.as_uri(),
            )
            assert response.content == {"status": "ok"}
            retry_response = await backend.generate_with_image(
                _request(source),
                frame_sheet.as_uri(),
            )
            assert retry_response.content == {"status": "ok"}

    asyncio.run(exercise())

    media_uuids = []
    for payload in seen_payloads:
        assert "media_io_kwargs" not in payload
        message_content = payload["messages"][0]["content"]
        assert message_content[0]["type"] == "image_url"
        assert message_content[0]["image_url"] == {
            "url": frame_sheet.as_uri()
        }
        media_uuid = message_content[0]["uuid"]
        assert UUID(media_uuid).version == 4
        media_uuids.append(media_uuid)
        assert message_content[1] == {"type": "text", "text": "annotate"}
    assert len(set(media_uuids)) == 2


def test_vllm_backend_uses_api_key_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_VLLM_API_KEY", "secret-value")
    seen_authorization: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_authorization.append(request.headers.get("Authorization"))
        return httpx.Response(
            200,
            json={"data": [{"id": "Qwen/Qwen3.6-27B"}]},
        )

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path, api_key_env="TEST_VLLM_API_KEY"),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            await backend.healthcheck()

    asyncio.run(exercise())
    assert seen_authorization == ["Bearer secret-value"]


def test_vllm_backend_uses_smaller_boundary_token_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    seen_max_tokens: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_max_tokens.append(payload["max_completion_tokens"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"status": "ok"})}}
                ]
            },
        )

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            await backend.generate(_request(source, stage="boundary-start"))

    asyncio.run(exercise())
    assert seen_max_tokens == [512]


def test_vllm_backend_rejects_missing_api_key(tmp_path: Path) -> None:
    with pytest.raises(VllmBackendError, match="missing API key"):
        VllmBackend(
            _config(tmp_path, api_key_env="MISSING_VLLM_API_KEY"),
            materializer=_materializer(tmp_path),
        )


def test_vllm_backend_rejects_unserved_model(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "other-model"}]})

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            with pytest.raises(VllmResponseError, match="is not served"):
                await backend.healthcheck()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (httpx.Response(503, text="unavailable"), VllmTransportError, "503"),
        (httpx.Response(200, text="not-json"), VllmResponseError, "invalid JSON"),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not-json"}}]},
            ),
            VllmResponseError,
            "not valid JSON",
        ),
    ],
)
def test_vllm_backend_classifies_response_failures_and_cleans_media(
    tmp_path: Path,
    response: httpx.Response,
    error: type[Exception],
    message: str,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")

    def handler(request: httpx.Request) -> httpx.Response:
        return response

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(handler),
        ) as backend:
            with pytest.raises(error, match=message):
                await backend.generate(_request(source))

    asyncio.run(exercise())
    assert list((tmp_path / "staging").iterdir()) == []


def test_vllm_backend_rejects_nonlocal_media_uri(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    request = _request(source, media_uri="https://example.com/video.mp4")

    async def exercise() -> None:
        async with VllmBackend(
            _config(tmp_path),
            materializer=_materializer(tmp_path),
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500)
            ),
        ) as backend:
            with pytest.raises(VllmBackendError, match="local file URI"):
                await backend.generate(request)

    asyncio.run(exercise())
