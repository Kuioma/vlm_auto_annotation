import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx

from auto_annotation.config.models import VllmBackendConfig
from auto_annotation.inference.base import (
    GenerationRequest,
    GenerationResponse,
    validate_response,
)
from auto_annotation.media.materialize import LocalVideoMaterializer


VLLM_ADAPTER_VERSION = "vllm-adapter-v3"


class VllmBackendError(RuntimeError):
    pass


class VllmTransportError(VllmBackendError):
    pass


class VllmResponseError(VllmBackendError):
    pass


def _local_file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise VllmBackendError(f"vLLM backend requires a local file URI: {uri}")
    return Path(unquote(parsed.path))


class VllmBackend:
    def __init__(
        self,
        config: VllmBackendConfig,
        *,
        materializer: LocalVideoMaterializer | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.model_id = config.model_id
        self.model_revision = config.model_revision
        self._materializer = materializer or LocalVideoMaterializer(
            config.media_staging_root
        )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key_env is not None:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise VllmBackendError(
                    f"missing API key environment variable: {config.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        base_url = f"{str(config.base_url).rstrip('/')}/"
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=config.timeout_s,
            transport=transport,
            trust_env=False,
        )

    async def __aenter__(self) -> "VllmBackend":
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

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method,
                path,
                json=payload,
            )
        except httpx.TimeoutException as error:
            raise VllmTransportError(
                f"vLLM request timed out: {method} {path}"
            ) from error
        except httpx.RequestError as error:
            raise VllmTransportError(
                f"vLLM request failed: {method} {path}: {error}"
            ) from error
        if not response.is_success:
            body = response.text[:2000]
            raise VllmTransportError(
                f"vLLM returned HTTP {response.status_code} for "
                f"{method} {path}: {body}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise VllmResponseError(
                f"vLLM returned invalid JSON for {method} {path}"
            ) from error
        if not isinstance(payload, dict):
            raise VllmResponseError(
                f"vLLM returned a non-object response for {method} {path}"
            )
        return payload

    async def healthcheck(self) -> None:
        payload = await self._request_json("GET", "models")
        data = payload.get("data")
        if not isinstance(data, list):
            raise VllmResponseError("vLLM models response has no data list")
        model_ids = {
            item.get("id")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if self.model_id not in model_ids:
            raise VllmResponseError(
                f"configured model {self.model_id!r} is not served; "
                f"available models: {sorted(model_ids)}"
            )

    async def _generate_completion(
        self,
        request: GenerationRequest,
        media_content: dict[str, Any],
        *,
        media_io_kwargs: dict[str, Any] | None = None,
    ) -> GenerationResponse:
        max_completion_tokens = (
            self.config.boundary_max_completion_tokens
            if request.stage.startswith("boundary-")
            else self.config.max_completion_tokens
        )
        # vLLM 0.24 can populate its API-process multimodal cache before
        # validating structured-output parameters. If validation then fails,
        # retrying the same media hash can leave the API and EngineCore caches
        # out of sync. An attempt-scoped UUID prevents reuse of such a stale
        # cache entry; this workload does not benefit from repeated media.
        media_content = {
            **media_content,
            "uuid": str(uuid4()),
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        media_content,
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_completion_tokens": max_completion_tokens,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "auto_annotation_response",
                    "schema": request.response_schema,
                },
            },
        }
        if media_io_kwargs is not None:
            payload["media_io_kwargs"] = media_io_kwargs
        raw = await self._request_json(
            "POST",
            "chat/completions",
            payload,
        )
        try:
            content_value = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise VllmResponseError(
                "vLLM completion response has no assistant content"
            ) from error
        if not isinstance(content_value, str) or not content_value.strip():
            raise VllmResponseError(
                "vLLM completion content is empty or not a string"
            )
        try:
            content = json.loads(content_value)
        except json.JSONDecodeError as error:
            raise VllmResponseError(
                "vLLM completion content is not valid JSON"
            ) from error
        if not isinstance(content, dict):
            raise VllmResponseError(
                "vLLM completion content must be a JSON object"
            )
        validate_response(request, content)
        return GenerationResponse(content=content, raw=raw)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        source = _local_file_path(request.media_uri)
        materialized = await asyncio.to_thread(
            self._materializer.materialize,
            source,
            request.media_start_ms,
            request.media_end_ms,
        )
        try:
            return await self._generate_completion(
                request,
                {
                    "type": "video_url",
                    "video_url": {"url": materialized.uri},
                },
                media_io_kwargs={
                    "video": {"fps": request.sampling_fps}
                },
            )
        finally:
            await asyncio.to_thread(materialized.cleanup)

    async def generate_with_image(
        self,
        request: GenerationRequest,
        image_uri: str,
    ) -> GenerationResponse:
        image_path = _local_file_path(image_uri).resolve(strict=True)
        if not image_path.is_file():
            raise VllmBackendError(f"image input is not a file: {image_path}")
        return await self._generate_completion(
            request,
            {
                "type": "image_url",
                "image_url": {"url": image_path.as_uri()},
            },
        )
