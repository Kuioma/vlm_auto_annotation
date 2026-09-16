import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    model_validator,
)


_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingConfig(ConfigModel):
    coarse_fps: float = Field(default=2.0, gt=0)
    refine_fps: float = Field(default=6.0, gt=0)
    refine_window_ms: int = Field(default=4000, gt=0)


class TimelineConfig(ConfigModel):
    merge_gap_ms: int = Field(default=250, ge=0)
    min_segment_ms: int = Field(default=300, ge=1)
    max_overlap_resolution_ms: int = Field(default=0, ge=0)


class MockBackendConfig(ConfigModel):
    kind: Literal["mock"]
    fixture_path: Path


class VllmBackendConfig(ConfigModel):
    kind: Literal["vllm"]
    base_url: AnyHttpUrl
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    vllm_version: str = Field(min_length=1)
    timeout_s: int = Field(default=180, gt=0)
    max_completion_tokens: int = Field(default=4096, gt=0)
    boundary_max_completion_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 0
    enable_thinking: bool = False
    media_staging_root: Path
    api_key_env: Annotated[
        str,
        Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
    ] | None = None


class OpenAICompatibleBackendConfig(ConfigModel):
    """DashScope Qwen vision OpenAI-compatible completion settings."""

    _resolved_base_url: AnyHttpUrl | None = PrivateAttr(default=None)

    kind: Literal["openai_compatible"]
    base_url: AnyHttpUrl | None = None
    base_url_env: Annotated[
        str,
        Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
    ] | None = None
    model_id: str = Field(min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    media_staging_root: Path
    api_key_env: Annotated[
        str,
        Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
    ]
    timeout_s: int = Field(default=180, gt=0)
    max_completion_tokens: int = Field(default=4096, gt=0)
    boundary_max_completion_tokens: int = Field(default=512, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, lt=2.0)

    @model_validator(mode="after")
    def validate_base_url_source(self) -> "OpenAICompatibleBackendConfig":
        if (self.base_url is None) == (self.base_url_env is None):
            raise ValueError(
                "exactly one of base_url and base_url_env must be configured"
            )
        return self

    @property
    def profile(self) -> str:
        return "dashscope_qwen_vision"

    @property
    def effective_base_url(self) -> AnyHttpUrl:
        if self.base_url is not None:
            return self.base_url
        if self._resolved_base_url is not None:
            return self._resolved_base_url
        assert self.base_url_env is not None
        raw_value = os.environ.get(self.base_url_env)
        if not raw_value or not raw_value.strip():
            raise ValueError(
                "missing base URL environment variable: "
                f"{self.base_url_env}"
            )
        try:
            resolved = _HTTP_URL_ADAPTER.validate_python(raw_value.strip())
        except ValueError:
            raise ValueError(
                "invalid base URL environment variable: "
                f"{self.base_url_env}"
            ) from None
        self._resolved_base_url = resolved
        return resolved

    @property
    def effective_model_revision(self) -> str:
        return self.model_revision or self.model_id


BackendConfig = Annotated[
    MockBackendConfig | VllmBackendConfig | OpenAICompatibleBackendConfig,
    Field(discriminator="kind"),
]


class RunConfig(ConfigModel):
    _source_path: Path | None = PrivateAttr(default=None)

    schema_path: Path
    ontology_path: Path
    manifest_path: Path
    artifact_root: Path
    output_path: Path
    backend: BackendConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)

    @property
    def source_path(self) -> Path | None:
        return self._source_path
