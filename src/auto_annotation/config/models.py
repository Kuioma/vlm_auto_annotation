from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, PrivateAttr


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


BackendConfig = Annotated[
    MockBackendConfig | VllmBackendConfig,
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
