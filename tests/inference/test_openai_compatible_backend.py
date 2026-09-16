import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
import pytest

from auto_annotation.config.models import OpenAICompatibleBackendConfig
from auto_annotation.inference.base import GenerationRequest
from auto_annotation.inference.openai_compatible import (
    OpenAICompatibleBackend,
    OpenAICompatibleBackendError,
    OpenAICompatibleHTTPError,
)


_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gARTGF2YzU4LjEzNC4xMDAA/9sAQwAIBAQEBAQFBQUFBQUGBgYGBgYGBgYGBgYGBwcHCAgIBwcHBgYHBwgICAgJCQkICAgICQkKCgoMDAsLDg4OEREU/8QASwABAQAAAAAAAAAAAAAAAAAAAAgBAQAAAAAAAAAAAAAAAAAAAAAQAQAAAAAAAAAAAAAAAAAAAAARAQAAAAAAAAAAAAAAAAAAAAD/wAARCAACAAIDASIAAhEAAxEA/9oADAMBAAIRAxEAPwCfwAf/2Q=="
)


def _config(tmp_path: Path) -> OpenAICompatibleBackendConfig:
    return OpenAICompatibleBackendConfig(
        kind="openai_compatible",
        base_url="https://workspace-id.example.test/compatible-mode/v1",
        model_id="qwen3-vl-plus",
        media_staging_root=tmp_path,
        api_key_env="DASHSCOPE_TEST_KEY",
        timeout_s=7,
        max_completion_tokens=99,
        boundary_max_completion_tokens=11,
        temperature=0.2,
    )


def _request(stage: str = "coarse") -> GenerationRequest:
    return GenerationRequest(
        request_id="request-1",
        stage=stage,
        media_uri="file:///tmp/video.mp4",
        media_start_ms=0,
        media_end_ms=1000,
        sample_timestamps_ms=(0,),
        sampling_fps=1.0,
        prompt="Keep this prompt private; return JSON.",
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )


def _backend(tmp_path: Path, handler: httpx.MockTransport) -> OpenAICompatibleBackend:
    os.environ["DASHSCOPE_TEST_KEY"] = "synthetic-secret"
    return OpenAICompatibleBackend(
        _config(tmp_path),
        transport=handler,
    )


def test_exact_json_mode_payload_and_private_two_image_order(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    # Use two independently valid images with different payload bytes while
    # retaining the tiny valid JPEG structure.
    image_two = _JPEG[:-2] + b"\x01" + _JPEG[-2:]
    async def run_two() -> None:
        backend = _backend(tmp_path, httpx.MockTransport(handler))
        try:
            await backend._post(_request(), (_JPEG, image_two))
        finally:
            await backend.aclose()

    asyncio.run(run_two())
    assert not hasattr(OpenAICompatibleBackend, "generate_with_images")
    assert len(requests) == 1
    assert requests[0].url.path == "/compatible-mode/v1/chat/completions"
    payload = json.loads(requests[0].content)
    assert set(payload) == {
        "model",
        "messages",
        "temperature",
        "max_completion_tokens",
        "stream",
        "response_format",
    }
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["stream"] is False
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "JSON Schema" in content[0]["text"]
    assert [part["type"] for part in content[1:]] == [
        "image_url",
        "image_url",
    ]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["image_url"]["url"] != content[2]["image_url"]["url"]
    assert requests[0].headers["authorization"] == "Bearer synthetic-secret"
    assert requests[0].headers["content-type"] == "application/json"


def test_base_url_env_selects_request_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL",
        "https://workspace-env.example.test/compatible-mode/v1",
    )
    config = OpenAICompatibleBackendConfig(
        kind="openai_compatible",
        base_url_env="DASHSCOPE_BASE_URL",
        model_id="qwen3-vl-plus",
        media_staging_root=tmp_path,
        api_key_env="DASHSCOPE_TEST_KEY",
    )
    monkeypatch.setenv("DASHSCOPE_TEST_KEY", "synthetic-secret")

    async def run() -> None:
        backend = OpenAICompatibleBackend(
            config,
            transport=httpx.MockTransport(handler),
        )
        try:
            await backend.generate_with_image(_request(), _JPEG)
        finally:
            await backend.aclose()

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url == (
        "https://workspace-env.example.test/"
        "compatible-mode/v1/chat/completions"
    )


def test_missing_base_url_env_fails_before_http_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_TEST_KEY", "synthetic-secret")
    config = OpenAICompatibleBackendConfig(
        kind="openai_compatible",
        base_url_env="DASHSCOPE_BASE_URL",
        model_id="qwen3-vl-plus",
        media_staging_root=tmp_path,
        api_key_env="DASHSCOPE_TEST_KEY",
    )

    with pytest.raises(
        ValueError,
        match="missing base URL environment variable: DASHSCOPE_BASE_URL",
    ):
        OpenAICompatibleBackend(
            config,
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )


@pytest.mark.parametrize(
    "bad_image",
    [b"", b"not-jpeg", b"\xff\xd8not-a-real-jpeg\xff\xd9", "not-bytes"],
)
def test_invalid_jpeg_fails_before_http(
    tmp_path: Path,
    bad_image: object,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    async def run() -> None:
        backend = _backend(tmp_path, httpx.MockTransport(handler))
        try:
            with pytest.raises(OpenAICompatibleBackendError):
                await backend._post(_request(), (bad_image,))  # type: ignore[arg-type]
        finally:
            await backend.aclose()

    asyncio.run(run())
    assert calls == 0


def test_jpeg_sof_component_length_mismatch_fails_before_http(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    malformed = bytes.fromhex(
        "ffd8ffc00008080001000101ffda00060000000001ffd9"
    )

    async def run() -> None:
        backend = _backend(tmp_path, httpx.MockTransport(handler))
        try:
            with pytest.raises(OpenAICompatibleBackendError):
                await backend._post(_request(), (malformed,))
        finally:
            await backend.aclose()

    asyncio.run(run())
    assert calls == 0


def test_data_uri_limit_fails_before_http(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    oversized = _JPEG[:-2] + (b"\x00" * (10 * 1024 * 1024)) + _JPEG[-2:]

    async def run() -> None:
        backend = _backend(tmp_path, httpx.MockTransport(handler))
        try:
            with pytest.raises(OpenAICompatibleBackendError, match="10 MiB"):
                await backend._post(_request(), (oversized,))
        finally:
            await backend.aclose()

    asyncio.run(run())
    assert calls == 0


def test_provider_error_fields_are_not_reflected_and_no_retry(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            503,
            json={
                "error": {
                    "code": "synthetic-secret",
                    "type": "Bearer synthetic-secret",
                    "message": (
                        "Keep this prompt private data:image/jpeg;base64,secret"
                    ),
                }
            },
        )

    async def run() -> None:
        backend = _backend(tmp_path, httpx.MockTransport(handler))
        try:
            with pytest.raises(OpenAICompatibleHTTPError) as caught:
                await backend.generate_with_image(_request(), _JPEG)
        finally:
            await backend.aclose()
        assert str(caught.value) == "HTTP 503"
        assert "synthetic-secret" not in str(caught.value)
        assert "prompt" not in str(caught.value)
        assert "data:image" not in str(caught.value)

    asyncio.run(run())
    assert calls == 1
