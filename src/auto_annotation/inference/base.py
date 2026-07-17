from typing import Any, Protocol

from jsonschema.validators import validator_for
from pydantic import BaseModel, Field, model_validator


class GenerationRequest(BaseModel):
    request_id: str
    stage: str
    media_uri: str
    media_start_ms: int = Field(ge=0)
    media_end_ms: int = Field(gt=0)
    sample_timestamps_ms: tuple[int, ...]
    sampling_fps: float = Field(gt=0, allow_inf_nan=False)
    prompt: str
    response_schema: dict[str, Any]

    @model_validator(mode="after")
    def validate_media_range(self) -> "GenerationRequest":
        if self.media_end_ms <= self.media_start_ms:
            raise ValueError("media_end_ms must be greater than media_start_ms")
        if any(
            timestamp < self.media_start_ms or timestamp >= self.media_end_ms
            for timestamp in self.sample_timestamps_ms
        ):
            raise ValueError("sample timestamp outside media range")
        return self


class GenerationResponse(BaseModel):
    content: dict[str, Any]
    raw: dict[str, Any]


def validate_response(request: GenerationRequest, content: dict[str, Any]) -> None:
    validator_class = validator_for(request.response_schema)
    validator_class.check_schema(request.response_schema)
    validator_class(request.response_schema).validate(content)


class InferenceBackend(Protocol):
    model_id: str
    model_revision: str

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError
