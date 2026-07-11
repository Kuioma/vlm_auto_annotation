from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestItem(DomainModel):
    video_id: str = Field(min_length=1)
    video_uri: Path
    dataset_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    ontology_id: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoInfo(DomainModel):
    path: Path
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    nominal_fps: float = Field(gt=0)
    frame_timestamps_ms: tuple[int, ...] = Field(min_length=1)


class VideoChunk(DomainModel):
    chunk_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "VideoChunk":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class SamplePoint(DomainModel):
    source_timestamp_ms: int = Field(ge=0)
    chunk_timestamp_ms: int = Field(ge=0)


class SamplePlan(DomainModel):
    chunk: VideoChunk
    target_fps: float = Field(gt=0)
    points: tuple[SamplePoint, ...] = Field(min_length=1)


class Segment(DomainModel):
    segment_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    values: dict[str, Any]
    sentence: str = Field(min_length=1)
    evidence_timestamps_ms: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interval(self) -> "Segment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class CoarseAnnotation(DomainModel):
    task_summary: str = Field(min_length=1)
    segments: list[Segment]


class Quality(DomainModel):
    score: float = Field(ge=0, le=1)
    signals: dict[str, Any] = Field(default_factory=dict)
    review_reasons: list[str] = Field(default_factory=list)


class Provenance(DomainModel):
    schema_version: str
    ontology_hash: str
    prompt_versions: dict[str, str]
    model_id: str
    model_revision: str
    config_hash: str


class FinalizedAnnotation(DomainModel):
    video_id: str
    duration_ms: int = Field(gt=0)
    task_summary: str = Field(min_length=1)
    segments: list[Segment]
    quality: Quality

    @model_validator(mode="after")
    def validate_timeline(self) -> "FinalizedAnnotation":
        previous_end = 0
        for segment in self.segments:
            if segment.end_ms > self.duration_ms:
                raise ValueError("segment exceeds video duration")
            if segment.start_ms < previous_end:
                raise ValueError("segments overlap or are not sorted")
            previous_end = segment.end_ms
        return self


class AnnotationDocument(FinalizedAnnotation):
    provenance: Provenance
