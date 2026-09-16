from auto_annotation.inference.base import GenerationRequest
from auto_annotation.prompts.dashscope import (
    DASHSCOPE_PROMPT_VERSION,
    adapt_request,
)


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="coarse:test",
        stage="coarse",
        media_uri="file:///tmp/video.mp4",
        media_start_ms=0,
        media_end_ms=1000,
        sample_timestamps_ms=(0,),
        sampling_fps=1.0,
        prompt="请返回 JSON object。",
        response_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )


def test_dashscope_prompt_is_deterministic_idempotent_and_schema_complete() -> None:
    request = _request()
    first = adapt_request(request)
    second = adapt_request(request)
    repeated = adapt_request(first)

    assert DASHSCOPE_PROMPT_VERSION in first.prompt or "DashScope" in first.prompt
    assert first.prompt == second.prompt == repeated.prompt
    assert "JSON object" in first.prompt
    assert "JSON Schema" in first.prompt
    assert '"additionalProperties":false' in first.prompt
    assert "Ontology" in first.prompt
    assert request.prompt == "请返回 JSON object。"

