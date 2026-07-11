from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SamplingConfig(ConfigModel):
    coarse_fps: float = Field(default=2.0, gt=0)
    refine_fps: float = Field(default=6.0, gt=0)
    refine_window_ms: int = Field(default=4000, gt=0)


class TimelineConfig(ConfigModel):
    merge_gap_ms: int = Field(default=250, ge=0)
    min_segment_ms: int = Field(default=300, ge=1)


class MockBackendConfig(ConfigModel):
    kind: Literal["mock"]
    fixture_path: Path


class RunConfig(ConfigModel):
    _source_path: Path | None = PrivateAttr(default=None)

    schema_path: Path
    ontology_path: Path
    manifest_path: Path
    artifact_root: Path
    output_path: Path
    backend: MockBackendConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)

    @property
    def source_path(self) -> Path | None:
        return self._source_path
