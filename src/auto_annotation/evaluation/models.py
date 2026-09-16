"""Strict external and normalized evaluator models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def reject_nested_non_finite_numbers(self) -> "EvaluationModel":
        def validate(value: Any, path: str) -> None:
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite number is not allowed at {path}"
                    )
                return
            if isinstance(value, BaseModel):
                for key, item in value.__dict__.items():
                    validate(item, f"{path}.{key}")
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    validate(item, f"{path}.{key}")
                return
            if isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for index, item in enumerate(value):
                    validate(item, f"{path}[{index}]")

        for key, value in self.__dict__.items():
            validate(value, key)
        return self


class GoldEpisodeSelection(EvaluationModel):
    episode_indices: list[StrictInt] = Field(min_length=1)
    mode: str | None = None
    sample_size: StrictInt | None = None
    seed: StrictInt | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "GoldEpisodeSelection":
        if len(set(self.episode_indices)) != len(self.episode_indices):
            raise ValueError("selected episode indices must be unique")
        if any(type(index) is not int or index < 0 for index in self.episode_indices):
            raise ValueError("selected episode indices must be non-negative integers")
        if self.sample_size is not None and self.sample_size != len(self.episode_indices):
            raise ValueError("selection sample_size does not match episode_indices")
        return self


class GoldOntology(EvaluationModel):
    hash: str | None = None
    id: str | None = None
    interval_semantics: str | None = None
    ordered_segments: list[dict[str, Any]] = Field(default_factory=list)
    require_full_coverage: bool = False


class GoldSetManifest(EvaluationModel):
    version: Literal[1]
    gold_set_id: str = Field(min_length=1)
    dataset_id: str | None = None
    created_at: str | None = None
    episode_selection: GoldEpisodeSelection
    episodes_sha256: str | None = None
    fps: float | None = None
    ontology: GoldOntology = Field(default_factory=GoldOntology)
    record_defaults: dict[str, Any] = Field(default_factory=dict)
    vlm_source: dict[str, Any] = Field(default_factory=dict)


class EvalSegment(EvaluationModel):
    segment_id: str = Field(min_length=1)
    start_ms: StrictInt = Field(ge=0)
    end_ms: StrictInt = Field(gt=0)
    values: dict[str, Any]
    sentence: str = Field(min_length=1)
    evidence_timestamps_ms: list[StrictInt] = Field(default_factory=list)
    start_frame: StrictInt | None = None
    end_frame_exclusive: StrictInt | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "EvalSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("segment end_ms must be greater than start_ms")
        return self


class EvalEvent(EvaluationModel):
    event_id: str = Field(min_length=1)
    timestamp_ms: StrictInt = Field(ge=0)
    values: dict[str, Any]
    sentence: str = Field(min_length=1)
    evidence_timestamps_ms: list[StrictInt] = Field(default_factory=list)
    timestamp_frame: StrictInt | None = None


class HumanAnnotationMetadata(EvaluationModel):
    version: StrictInt
    gold_set_id: str = Field(min_length=1)
    episode_index: StrictInt = Field(ge=0)
    status: str = Field(min_length=1)
    exclusion_reason: str | None = None
    annotator_id: str | None = None
    updated_at: str | None = None


class GoldAnnotation(EvaluationModel):
    video_id: str = Field(min_length=1)
    duration_ms: StrictInt = Field(gt=0)
    task_summary: str = Field(min_length=1)
    events: list[EvalEvent] = Field(default_factory=list)
    segments: list[EvalSegment] = Field(default_factory=list)
    human_annotation: HumanAnnotationMetadata


class NormalizedPrediction(EvaluationModel):
    video_id: str = Field(min_length=1)
    duration_ms: StrictInt = Field(gt=0)
    task_summary: str = Field(min_length=1)
    events: list[EvalEvent] = Field(default_factory=list)
    segments: list[EvalSegment] = Field(default_factory=list)
    ontology_id: str | None = None
    ontology_hash: str | None = None
    provenance_verified: bool = False


class FailureRecord(EvaluationModel):
    category: str
    severity: Literal["info", "warning", "error"]
    message: str
    episode_index: int | None = None
    video_id: str | None = None


class ConfidenceInterval(EvaluationModel):
    status: Literal["computed", "not_applicable", "unavailable"]
    confidence_level: float | None = None
    lower: float | None = None
    upper: float | None = None
    valid_resamples: int = 0
    reason: str | None = None


class MetricResult(EvaluationModel):
    status: Literal["computed", "skipped", "not_implemented", "not_applicable"]
    version: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    aggregation: dict[str, Any] = Field(default_factory=dict)
    denominator: dict[str, int] = Field(default_factory=dict)
    missing_count: int = 0
    confidence_intervals: dict[str, ConfidenceInterval] = Field(default_factory=dict)
    reason: str | None = None


class EligibilitySummary(EvaluationModel):
    selected: int
    complete: int
    included: int
    excluded: int
    incomplete: int
    predicted: int
    missing_predictions: int
    invalid_predictions: int
    extra_predictions: int
    exclusion_reasons: dict[str, int] = Field(default_factory=dict)


class EvaluationSummary(EvaluationModel):
    result_schema_version: str
    evaluator_version: str
    evaluation_id: str
    profile_id: str
    evaluation_mode: Literal["official", "partial"]
    report_label: Literal["OFFICIAL", "PARTIAL"]
    config_sha256: str
    gold_manifest_sha256: str
    included_annotations_sha256: str
    prediction_sha256: str
    metric_versions: dict[str, str]
    eligibility: EligibilitySummary
    prediction_coverage: dict[str, Any]
    prediction_schema_validity: dict[str, Any]
    metrics: dict[str, MetricResult]
    decision: Literal["report_only"] = "report_only"


class EpisodeEvaluation(EvaluationModel):
    episode_index: int
    video_id: str
    status: Literal["evaluated", "missing_prediction", "invalid_prediction"]
    provenance_status: Literal["verified", "unverified", "missing", "invalid"]
    boundaries: dict[str, dict[str, Any]]
    segments: dict[str, dict[str, Any]]
    temporal_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    extra_segment_ids: list[str] = Field(default_factory=list)
    sequence: dict[str, Any] | None = None
    labels: dict[str, Any] | None = None
    segmentation: dict[str, Any] | None = None
    max_boundary_error_ms: int | None = None
    mean_tiou: float


class ReviewQueueItem(EvaluationModel):
    rank: int
    episode_index: int
    video_id: str
    prediction_status: Literal["evaluated", "missing_prediction", "invalid_prediction"]
    max_boundary_error_ms: int | None
    mean_tiou: float
    reasons: list[str]
