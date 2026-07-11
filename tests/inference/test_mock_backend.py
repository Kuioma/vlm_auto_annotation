import asyncio

import pytest
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from auto_annotation.inference.base import GenerationRequest
from auto_annotation.inference.mock import ScriptedMockBackend


def _request(**overrides: object) -> GenerationRequest:
    values = {
        "request_id": "coarse:video-1",
        "stage": "coarse",
        "media_uri": "file:///tmp/video.mp4",
        "media_start_ms": 0,
        "media_end_ms": 10_000,
        "sample_timestamps_ms": (0, 500, 1000),
        "prompt": "annotate",
        "response_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_summary": {"type": "string"},
                "segments": {"type": "array"},
            },
            "required": ["task_summary", "segments"],
        },
    }
    values.update(overrides)
    return GenerationRequest.model_validate(values)


def test_mock_backend_records_calls_and_consumes_script() -> None:
    backend = ScriptedMockBackend(
        {"coarse:video-1": [{"task_summary": "拿起杯子", "segments": []}]}
    )
    request = _request()

    response = asyncio.run(backend.generate(request))

    assert response.content["task_summary"] == "拿起杯子"
    assert backend.calls == [request]
    with pytest.raises(RuntimeError, match="exhausted"):
        asyncio.run(backend.generate(request))


def test_mock_backend_validates_response_schema() -> None:
    backend = ScriptedMockBackend(
        {"coarse:video-1": [{"task_summary": "拿起杯子"}]}
    )
    request = _request(
        response_schema={
            "type": "object",
            "properties": {"segments": {"type": "array"}},
            "required": ["segments"],
        }
    )

    with pytest.raises(JsonSchemaValidationError):
        asyncio.run(backend.generate(request))


@pytest.mark.parametrize(
    "sample_timestamps_ms",
    [(999,), (2000,), (1000, 1500, 2000)],
    ids=["before-start", "at-end", "mixed-with-at-end"],
)
def test_generation_request_rejects_samples_outside_half_open_media_range(
    sample_timestamps_ms: tuple[int, ...],
) -> None:
    with pytest.raises(PydanticValidationError, match="outside media range"):
        _request(
            media_start_ms=1000,
            media_end_ms=2000,
            sample_timestamps_ms=sample_timestamps_ms,
        )


def test_generation_request_accepts_samples_at_start_and_before_end() -> None:
    request = _request(
        media_start_ms=1000,
        media_end_ms=2000,
        sample_timestamps_ms=(1000, 1999),
    )

    assert request.sample_timestamps_ms == (1000, 1999)
