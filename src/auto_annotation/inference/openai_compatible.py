"""DashScope Qwen vision backend over OpenAI-compatible Chat Completions."""

import asyncio
import base64
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from auto_annotation.config.models import OpenAICompatibleBackendConfig
from auto_annotation.inference.base import (
    GenerationRequest,
    GenerationResponse,
    validate_response,
)
from auto_annotation.media.materialize import (
    FRAME_SHEET_MATERIALIZER_VERSION,
    LocalFrameSheetMaterializer,
)
from auto_annotation.prompts.dashscope import (
    DASHSCOPE_PROMPT_VERSION,
    adapt_request,
)


OPENAI_COMPATIBLE_PROFILE = "dashscope_qwen_vision"
OPENAI_COMPATIBLE_ADAPTER_VERSION = (
    "dashscope-openai-compatible-json-mode-v2"
)
DASHSCOPE_PROFILE = OPENAI_COMPATIBLE_PROFILE
DASHSCOPE_ADAPTER_VERSION = OPENAI_COMPATIBLE_ADAPTER_VERSION
DASHSCOPE_PROFILE_NAME = OPENAI_COMPATIBLE_PROFILE
MAX_IMAGE_DATA_URI_BYTES = 10 * 1024 * 1024


class OpenAICompatibleBackendError(RuntimeError):
    """Base error that deliberately never includes the bearer secret."""


class OpenAICompatibleTransportError(OpenAICompatibleBackendError):
    pass


class OpenAICompatibleHTTPError(OpenAICompatibleBackendError):
    pass


class OpenAICompatibleResponseError(OpenAICompatibleBackendError):
    pass


def _local_file_path(uri: str | Path) -> Path:
    if isinstance(uri, Path):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise OpenAICompatibleBackendError(
            "OpenAI-compatible backend requires a local file URI"
        )
    return Path(unquote(parsed.path))


def _is_valid_jpeg(data: bytes) -> bool:
    """Perform a dependency-free structural JPEG decode check.

    This walks the marker stream, validates segment lengths/SOF dimensions,
    handles stuffed entropy bytes and restart markers, and requires a complete
    SOS scan ending at EOI.  It intentionally rejects marker-wrapped arbitrary
    bytes before any HTTP request is attempted.
    """

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return False
    position = 2
    saw_sof = False
    saw_sos = False
    while position < len(data):
        if data[position] != 0xFF:
            return False
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            return False
        marker = data[position]
        position += 1
        if marker == 0xD9:  # EOI
            return saw_sof and saw_sos and position == len(data)
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            return False
        length = int.from_bytes(data[position : position + 2], "big")
        if length < 2 or position + length > len(data):
            return False
        segment_start = position + 2
        segment_end = position + length
        if marker == 0xDA:  # SOS; parse entropy-coded bytes until next marker
            if length < 6:
                return False
            saw_sos = True
            position = segment_end
            while position < len(data):
                if data[position] != 0xFF:
                    position += 1
                    continue
                marker_start = position
                while position < len(data) and data[position] == 0xFF:
                    position += 1
                if position >= len(data):
                    return False
                scan_marker = data[position]
                if scan_marker == 0x00 or 0xD0 <= scan_marker <= 0xD7:
                    position += 1
                    continue
                if scan_marker == 0xD9:
                    position += 1
                    return saw_sof and position == len(data)
                # A non-restart marker terminates this scan and is parsed by
                # the outer marker loop.  Restore its leading FF byte.
                position = marker_start
                break
            continue
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if length < 8:
                return False
            height = int.from_bytes(data[segment_start + 1 : segment_start + 3], "big")
            width = int.from_bytes(data[segment_start + 3 : segment_start + 5], "big")
            components = data[segment_start + 5]
            if width <= 0 or height <= 0 or components <= 0:
                return False
            # Each component contributes three bytes (id, sampling, table)
            # after the precision/size/component-count header.  Reject a
            # syntactically padded SOF before invoking the decoder.
            if length != 8 + 3 * components:
                return False
            saw_sof = True
        position = segment_end
    return False


def _decode_jpeg(data: bytes) -> bool:
    """Decode one JPEG through ffmpeg without producing an output file."""

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "image2pipe",
                "-i",
                "pipe:0",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            input=data,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _jpeg_bytes(path: Path | bytes) -> bytes:
    if isinstance(path, bytes):
        data = path
    else:
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise OpenAICompatibleBackendError(
                f"image input does not exist: {path}"
            ) from error
        if path.is_symlink() or not resolved.is_file():
            raise OpenAICompatibleBackendError(
                f"image input is not a file: {path}"
            )
        try:
            data = resolved.read_bytes()
        except OSError as error:
            raise OpenAICompatibleBackendError("cannot read image input") from error
    if not _is_valid_jpeg(data) or not _decode_jpeg(data):
        raise OpenAICompatibleBackendError(
            "image input is not a valid JPEG"
        )
    return data


def jpeg_data_uri(path: str | Path | bytes) -> str:
    data = _jpeg_bytes(path if isinstance(path, bytes) else _local_file_path(path))
    encoded = base64.b64encode(data).decode("ascii")
    uri = f"data:image/jpeg;base64,{encoded}"
    if len(uri.encode("utf-8")) > MAX_IMAGE_DATA_URI_BYTES:
        raise OpenAICompatibleBackendError(
            "image data URI exceeds the 10 MiB limit"
        )
    return uri


def _serialize_jpeg_sequence(
    images: Sequence[str | Path | bytes],
) -> tuple[str, ...]:
    if isinstance(images, (str, bytes, Path)):
        raise OpenAICompatibleBackendError(
            "images must be a non-empty ordered sequence"
        )
    if not images:
        raise OpenAICompatibleBackendError(
            "images must be a non-empty ordered sequence"
        )
    return tuple(jpeg_data_uri(image) for image in images)


def _safe_error_details(
    response: httpx.Response,
    *,
    secret: str | None = None,
) -> str:
    # Provider-controlled code/type/message fields can contain credentials,
    # prompts, or media.  Report only the locally observed HTTP status; never
    # reflect any provider body field, even after truncation or replacement.
    del secret
    return f"HTTP {response.status_code}"


class OpenAICompatibleBackend:
    model_id: str
    model_revision: str

    def __init__(
        self,
        config: OpenAICompatibleBackendConfig,
        *,
        materializer: LocalFrameSheetMaterializer | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.model_id = config.model_id
        self.model_revision = config.effective_model_revision
        api_key = os.environ.get(config.api_key_env)
        if not api_key or not api_key.strip():
            raise OpenAICompatibleBackendError(
                f"missing API key environment variable: {config.api_key_env}"
            )
        self._api_key = api_key
        self._materializer = materializer or LocalFrameSheetMaterializer(
            config.media_staging_root
        )
        self._client = httpx.AsyncClient(
            base_url=f"{str(config.effective_base_url).rstrip('/')}/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout_s,
            transport=transport,
            trust_env=False,
        )

    async def __aenter__(self) -> "OpenAICompatibleBackend":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(
        self,
        request: GenerationRequest,
        image_uris: Sequence[str | Path | bytes],
    ) -> dict[str, Any]:
        encoded_images = _serialize_jpeg_sequence(image_uris)
        adapted = adapt_request(request)
        max_tokens = (
            self.config.boundary_max_completion_tokens
            if request.stage.startswith("boundary-")
            else self.config.max_completion_tokens
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": adapted.prompt}
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": image_uri},
            }
            for image_uri in encoded_images
        )
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

    async def _post(
        self,
        request: GenerationRequest,
        image_uris: Sequence[str | Path | bytes],
    ) -> GenerationResponse:
        payload = self._payload(request, image_uris)
        try:
            response = await self._client.post("chat/completions", json=payload)
        except httpx.TimeoutException as error:
            raise OpenAICompatibleTransportError(
                "DashScope completion request timed out"
            ) from error
        except httpx.RequestError as error:
            raise OpenAICompatibleTransportError(
                f"DashScope completion request failed: {type(error).__name__}"
            ) from error
        if not response.is_success:
            raise OpenAICompatibleHTTPError(
                _safe_error_details(response, secret=self._api_key)
            )
        try:
            raw = response.json()
        except ValueError as error:
            raise OpenAICompatibleResponseError(
                "DashScope completion returned invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise OpenAICompatibleResponseError(
                "DashScope completion response must be a JSON object"
            )
        if isinstance(raw.get("error"), dict):
            raise OpenAICompatibleResponseError(
                "DashScope completion returned an error object"
            )
        try:
            content_value = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise OpenAICompatibleResponseError(
                "DashScope completion response has no assistant content"
            ) from error
        if not isinstance(content_value, str) or not content_value.strip():
            raise OpenAICompatibleResponseError(
                "DashScope completion content is empty or not a string"
            )
        try:
            content = json.loads(content_value)
        except json.JSONDecodeError as error:
            raise OpenAICompatibleResponseError(
                "DashScope completion content is not valid JSON"
            ) from error
        if not isinstance(content, dict):
            raise OpenAICompatibleResponseError(
                "DashScope completion content must be a JSON object"
            )
        try:
            validate_response(request, content)
        except Exception as error:
            raise OpenAICompatibleResponseError(
                "DashScope completion content failed local schema validation"
            ) from error
        return GenerationResponse(content=content, raw=raw)

    async def _generate_with_images(
        self,
        request: GenerationRequest,
        images: Sequence[str | Path | bytes],
    ) -> GenerationResponse:
        return await self._post(request, images)

    async def generate_with_image(
        self,
        request: GenerationRequest,
        image: str | Path | bytes,
    ) -> GenerationResponse:
        """Public single-image production entry point."""

        return await self._post(request, (image,))

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        source = _local_file_path(request.media_uri)
        materialized = await asyncio.to_thread(
            self._materializer.materialize,
            source,
            request.sample_timestamps_ms,
            label_timestamps_ms=tuple(
                timestamp - request.media_start_ms
                for timestamp in request.sample_timestamps_ms
            ),
        )
        try:
            return await self.generate_with_image(request, materialized.path)
        finally:
            await asyncio.to_thread(materialized.cleanup)
