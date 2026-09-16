"""Strict, path-aware configuration for Gold-set evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unique(values: list[object], label: str) -> list[object]:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")
    return values


def _strictly_increasing(values: list[int | float], label: str):
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label} must be strictly increasing")
    return values


class EvaluationInputs(ConfigModel):
    gold_set_path: Path
    annotations_dir: Path
    prediction_path: Path


class EligibilityConfig(ConfigModel):
    require_complete_gold_set: bool = True
    included_statuses: list[str] = Field(default_factory=lambda: ["complete"], min_length=1)
    missing_gold_annotation: Literal["fail"] = "fail"
    missing_prediction: Literal["count_as_missing"] = "count_as_missing"
    extra_prediction: Literal["ignore_and_report"] = "ignore_and_report"
    duplicate_video_id: Literal["fail"] = "fail"
    duration_mismatch: Literal["fail"] = "fail"

    @field_validator("included_statuses")
    @classmethod
    def validate_statuses(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("included_statuses may not contain blank values")
        return list(_unique(value, "included_statuses"))


class BoundaryMetricConfig(ConfigModel):
    enabled: bool = True
    metric_version: Literal["boundary-error-v1"] = "boundary-error-v1"
    event_ids: list[str] = Field(min_length=1)
    thresholds_ms: list[StrictInt] = Field(min_length=1)
    report: list[Literal["mae", "signed_bias", "hit_rate", "per_event"]] = Field(
        default_factory=lambda: ["mae", "signed_bias", "hit_rate", "per_event"]
    )
    reason: str | None = None

    @field_validator("event_ids")
    @classmethod
    def validate_events(cls, value: list[str]) -> list[str]:
        if any(not event.strip() for event in value):
            raise ValueError("event_ids may not contain blank values")
        return list(_unique(value, "event_ids"))

    @field_validator("thresholds_ms")
    @classmethod
    def validate_thresholds(cls, value: list[int]) -> list[int]:
        if any(type(item) is not int or item <= 0 for item in value):
            raise ValueError("boundary thresholds must be positive integer milliseconds")
        return list(_strictly_increasing(value, "boundary thresholds"))


class TemporalIouMetricConfig(ConfigModel):
    enabled: bool = True
    metric_version: Literal["temporal-iou-v1"] = "temporal-iou-v1"
    matching: Literal["segment_id"] = "segment_id"
    thresholds: list[float] = Field(min_length=1)
    report: list[Literal["mean", "per_segment", "precision", "recall", "f1"]] = Field(
        default_factory=lambda: ["mean", "per_segment", "precision", "recall", "f1"]
    )
    reason: str | None = None

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, value: list[float]) -> list[float]:
        if any(not 0 < item <= 1 for item in value):
            raise ValueError("tIoU thresholds must be in (0, 1]")
        return list(_strictly_increasing(value, "tIoU thresholds"))


class SequenceMetricConfig(ConfigModel):
    enabled: bool = False
    metric_version: Literal["sequence-edit-distance-v1"] = "sequence-edit-distance-v1"
    field: str = "atomic_action"
    reason: str | None = None

    @field_validator("field")
    @classmethod
    def validate_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sequence field may not be blank")
        return value


class LabelMetricConfig(ConfigModel):
    enabled: bool = False
    metric_version: Literal["label-accuracy-v1"] = "label-accuracy-v1"
    fields: list[str] = Field(default_factory=lambda: ["atomic_action"], min_length=1)
    exact_match_groups: list[list[str]] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LabelMetricConfig":
        if any(not field.strip() for field in self.fields):
            raise ValueError("label fields may not be blank")
        _unique(self.fields, "label fields")
        seen: set[tuple[str, ...]] = set()
        for group in self.exact_match_groups:
            if not group or any(not field.strip() for field in group):
                raise ValueError("exact-match groups must contain nonblank fields")
            _unique(group, "exact-match group fields")
            key = tuple(group)
            if key in seen:
                raise ValueError("exact-match groups must be unique")
            seen.add(key)
        return self


class SegmentationMetricConfig(ConfigModel):
    enabled: bool = False
    metric_version: Literal["segmentation-errors-v1"] = "segmentation-errors-v1"
    substantial_overlap_threshold: float = Field(default=0.5, gt=0, le=1)
    report: list[
        Literal[
            "over_split_count",
            "missed_segment_count",
            "incorrect_merge_count",
            "extra_segment_count",
        ]
    ] = Field(
        default_factory=lambda: [
            "over_split_count",
            "missed_segment_count",
            "incorrect_merge_count",
            "extra_segment_count",
        ]
    )
    reason: str | None = None


class DeferredMetricConfig(ConfigModel):
    enabled: Literal[False] = False
    implementation: Literal["deferred"] = "deferred"
    reason: str | None = None


class MetricsConfig(ConfigModel):
    boundary_error: BoundaryMetricConfig
    temporal_iou: TemporalIouMetricConfig
    sequence_edit_distance: SequenceMetricConfig = Field(default_factory=SequenceMetricConfig)
    label_accuracy: LabelMetricConfig = Field(default_factory=LabelMetricConfig)
    segmentation_errors: SegmentationMetricConfig = Field(default_factory=SegmentationMetricConfig)
    semantic_quality: DeferredMetricConfig = Field(default_factory=DeferredMetricConfig)
    gold_reliability: DeferredMetricConfig = Field(default_factory=DeferredMetricConfig)


class AggregationConfig(ConfigModel):
    primary: Literal["macro_episode"] = "macro_episode"
    also_report: list[Literal["micro", "per_event", "per_segment"]] = Field(
        default_factory=lambda: ["micro", "per_event", "per_segment"]
    )
    percentiles: list[float] = Field(default_factory=lambda: [50, 90], min_length=1)
    report_max: bool = True

    @field_validator("also_report")
    @classmethod
    def validate_reports(cls, value: list[str]) -> list[str]:
        return list(_unique(value, "aggregation views"))

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, value: list[float]) -> list[float]:
        if any(not 0 <= item <= 100 for item in value):
            raise ValueError("percentiles must be in [0, 100]")
        _unique(value, "percentiles")
        return list(_strictly_increasing(value, "percentiles"))


class MissingPredictionsConfig(ConfigModel):
    report_coverage: bool = True
    boundary_mae: Literal["exclude_with_count"] = "exclude_with_count"
    boundary_hit_rate: Literal["count_as_miss"] = "count_as_miss"


class UncertaintyConfig(ConfigModel):
    enabled: bool = True
    method: Literal["episode_bootstrap"] = "episode_bootstrap"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    resamples: StrictInt = Field(default=10000, gt=0)
    seed: StrictInt = 20260721


class DecisionConfig(ConfigModel):
    mode: Literal["report_only"] = "report_only"
    composite_score: None = None
    thresholds: None = None


class ReviewQueueConfig(ConfigModel):
    enabled: bool = True
    top_k: int = Field(default=20, gt=0)
    order_by: list[
        Literal["missing_prediction", "max_boundary_error_ms", "mean_tiou_ascending"]
    ] = Field(
        default_factory=lambda: [
            "missing_prediction",
            "max_boundary_error_ms",
            "mean_tiou_ascending",
        ]
    )

    @field_validator("order_by")
    @classmethod
    def validate_order(cls, value: list[str]) -> list[str]:
        return list(_unique(value, "review queue order"))


class OutputConfig(ConfigModel):
    root: Path
    write_summary_json: Literal[True] = True
    write_episodes_jsonl: bool = True
    write_failures_jsonl: bool = True
    write_review_queue_jsonl: bool = True
    write_markdown_report: bool = True


class EvaluationConfig(ConfigModel):
    _source_path: Path | None = PrivateAttr(default=None)

    version: Literal[1]
    profile_id: str = Field(min_length=1)
    inputs: EvaluationInputs
    evaluation_mode: Literal["official", "partial"] = "official"
    eligibility: EligibilityConfig = Field(default_factory=EligibilityConfig)
    metrics: MetricsConfig
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    missing_predictions: MissingPredictionsConfig = Field(default_factory=MissingPredictionsConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    review_queue: ReviewQueueConfig = Field(default_factory=ReviewQueueConfig)
    output: OutputConfig

    @property
    def source_path(self) -> Path | None:
        return self._source_path


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def load_evaluation_config(path: Path) -> EvaluationConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load evaluation config: {path}") from error
    config = EvaluationConfig.model_validate(raw)
    base = path.resolve().parent
    inputs = config.inputs.model_copy(
        update={
            "gold_set_path": _resolve(base, config.inputs.gold_set_path),
            "annotations_dir": _resolve(base, config.inputs.annotations_dir),
            "prediction_path": _resolve(base, config.inputs.prediction_path),
        }
    )
    output = config.output.model_copy(update={"root": _resolve(base, config.output.root)})
    resolved = config.model_copy(update={"inputs": inputs, "output": output})
    resolved._source_path = path.resolve()
    return resolved


def canonical_config_bytes(config: EvaluationConfig) -> bytes:
    return json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def evaluation_config_hash(config: EvaluationConfig) -> str:
    return f"sha256:{hashlib.sha256(canonical_config_bytes(config)).hexdigest()}"
