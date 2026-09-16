"""Load, normalize, and fully prevalidate evaluator inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from auto_annotation.evaluation.config import EvaluationConfig
from auto_annotation.evaluation.errors import EvaluationError
from auto_annotation.evaluation.models import (
    EvalEvent,
    EvalSegment,
    FailureRecord,
    GoldAnnotation,
    GoldSetManifest,
    NormalizedPrediction,
)


@dataclass(frozen=True)
class LoadedInputs:
    manifest: GoldSetManifest
    manifest_bytes: bytes
    selected_annotation_bytes: dict[int, bytes]
    included: tuple[GoldAnnotation, ...]
    excluded: tuple[GoldAnnotation, ...]
    incomplete_indices: tuple[int, ...]
    predictions: dict[str, NormalizedPrediction]
    invalid_predictions: dict[str, str]
    declared_prediction_durations: dict[str, int]
    prediction_bytes: bytes
    failures: tuple[FailureRecord, ...]
    extra_prediction_ids: tuple[str, ...]


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise EvaluationError(f"cannot read {label}: {path}") from error


def _load_json(raw: bytes, path: Path, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid {label} JSON: {path}") from error


def _validate_timeline(
    *,
    duration_ms: int,
    segments: list[EvalSegment],
    label: str,
    require_full_coverage: bool,
) -> None:
    ids = [segment.segment_id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} segment IDs must be unique")
    previous_end = 0
    for position, segment in enumerate(segments):
        if segment.end_ms > duration_ms:
            raise ValueError(f"{label} segment exceeds duration: {segment.segment_id}")
        if position and segment.start_ms < previous_end:
            raise ValueError(f"{label} segments overlap or are not sorted")
        if require_full_coverage and segment.start_ms != previous_end:
            raise ValueError(f"{label} full-coverage timeline contains a gap")
        previous_end = segment.end_ms
    if require_full_coverage and (
        not segments or segments[0].start_ms != 0 or previous_end != duration_ms
    ):
        raise ValueError(f"{label} timeline does not fully cover the video")


def _validate_events(
    *, duration_ms: int, events: list[EvalEvent], label: str
) -> None:
    ids = [event.event_id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} event IDs must be unique")
    if any(event.timestamp_ms > duration_ms for event in events):
        raise ValueError(f"{label} event exceeds duration")


def _project_segment(raw: Any) -> EvalSegment:
    if not isinstance(raw, dict):
        raise ValueError("segment must be an object")
    return EvalSegment.model_validate(
        {
            key: raw[key]
            for key in (
                "segment_id",
                "start_ms",
                "end_ms",
                "values",
                "sentence",
                "evidence_timestamps_ms",
                "start_frame",
                "end_frame_exclusive",
            )
            if key in raw
        }
    )


def _project_event(raw: Any) -> EvalEvent:
    if not isinstance(raw, dict):
        raise ValueError("event must be an object")
    return EvalEvent.model_validate(
        {
            key: raw[key]
            for key in (
                "event_id",
                "timestamp_ms",
                "values",
                "sentence",
                "evidence_timestamps_ms",
                "timestamp_frame",
            )
            if key in raw
        }
    )


def _normalize_prediction(payload: Any) -> NormalizedPrediction:
    if not isinstance(payload, dict):
        raise ValueError("prediction must be an object")
    video_id = payload.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        raise EvaluationError("prediction record has no nonblank video_id")
    duration_ms = payload.get("duration_ms")
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("prediction duration_ms must be a positive integer")
    raw_segments = payload.get("segments", [])
    raw_events = payload.get("events", [])
    if not isinstance(raw_segments, list) or not isinstance(raw_events, list):
        raise ValueError("prediction events and segments must be arrays")
    segments = [_project_segment(item) for item in raw_segments]
    events = [_project_event(item) for item in raw_events]
    experiment = payload.get("experiment")
    provenance = payload.get("provenance")
    ontology_id = None
    ontology_hash = None
    if isinstance(experiment, dict):
        raw_id = experiment.get("ontology_id")
        raw_hash = experiment.get("ontology_hash")
        ontology_id = raw_id if isinstance(raw_id, str) else None
        ontology_hash = raw_hash if isinstance(raw_hash, str) else None
    if isinstance(provenance, dict):
        raw_id = provenance.get("ontology_id")
        raw_hash = provenance.get("ontology_hash")
        if isinstance(raw_id, str):
            ontology_id = raw_id
        if isinstance(raw_hash, str):
            ontology_hash = raw_hash
    task_summary = payload.get("task_summary")
    if not isinstance(task_summary, str) or not task_summary.strip():
        raise ValueError("prediction task_summary must be a nonblank string")
    return NormalizedPrediction(
        video_id=video_id,
        duration_ms=duration_ms,
        task_summary=task_summary,
        events=events,
        segments=segments,
        ontology_id=ontology_id,
        ontology_hash=ontology_hash,
        provenance_verified=ontology_id is not None or ontology_hash is not None,
    )


def _gold_boundaries(annotation: GoldAnnotation, event_ids: list[str]) -> dict[str, int]:
    events = {event.event_id: event.timestamp_ms for event in annotation.events}
    segments = {segment.segment_id: segment.start_ms for segment in annotation.segments}
    resolved: dict[str, int] = {}
    for event_id in event_ids:
        if event_id not in events:
            raise ValueError(f"Gold boundary event is missing: {event_id}")
        if event_id not in segments:
            raise ValueError(f"Gold boundary segment is missing: {event_id}")
        if events[event_id] != segments[event_id]:
            raise ValueError(f"Gold event/segment boundary mismatch: {event_id}")
        if not 0 < events[event_id] < annotation.duration_ms:
            raise ValueError(f"Gold boundary must be strictly inside the video: {event_id}")
        resolved[event_id] = events[event_id]
    return resolved


def resolve_prediction_boundaries(
    prediction: NormalizedPrediction, event_ids: list[str]
) -> dict[str, int | None]:
    events = {event.event_id: event.timestamp_ms for event in prediction.events}
    segments = {segment.segment_id: segment.start_ms for segment in prediction.segments}
    resolved: dict[str, int | None] = {}
    for event_id in event_ids:
        event_value = events.get(event_id)
        segment_value = segments.get(event_id)
        if event_value is not None and segment_value is not None and event_value != segment_value:
            raise ValueError(f"prediction event/segment boundary mismatch: {event_id}")
        value = event_value if event_value is not None else segment_value
        if value is not None and not 0 < value < prediction.duration_ms:
            raise ValueError(
                f"prediction boundary must be strictly inside the video: {event_id}"
            )
        resolved[event_id] = value
    return resolved


def _parse_gold_annotations(
    *, config: EvaluationConfig, manifest: GoldSetManifest
) -> tuple[dict[int, GoldAnnotation], dict[int, bytes], list[FailureRecord]]:
    directory = config.inputs.annotations_dir
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as error:
        raise EvaluationError(f"cannot scan annotations directory: {directory}") from error
    if not directory.is_dir():
        raise EvaluationError(f"annotations_dir is not a directory: {directory}")
    by_index: dict[int, GoldAnnotation] = {}
    bytes_by_index: dict[int, bytes] = {}
    failures: list[FailureRecord] = []
    selected = set(manifest.episode_selection.episode_indices)
    seen_indices: set[int] = set()
    for path in paths:
        raw = _read_bytes(path, "Gold annotation")
        payload = _load_json(raw, path, "Gold annotation")
        try:
            annotation = GoldAnnotation.model_validate(payload)
        except ValidationError as error:
            raise EvaluationError(f"invalid Gold annotation contract: {path}: {error}") from error
        index = annotation.human_annotation.episode_index
        if index in seen_indices:
            raise EvaluationError(f"duplicate Gold episode_index: {index}")
        seen_indices.add(index)
        if index not in selected:
            failures.append(
                FailureRecord(
                    category="extra_gold_annotation",
                    severity="info",
                    episode_index=index,
                    video_id=annotation.video_id,
                    message="annotation is outside the manifest selection and was ignored",
                )
            )
            continue
        by_index[index] = annotation
        bytes_by_index[index] = raw
    return by_index, bytes_by_index, failures


def _parse_predictions(
    path: Path,
) -> tuple[
    dict[str, NormalizedPrediction],
    dict[str, str],
    dict[str, int],
    bytes,
]:
    raw = _read_bytes(path, "prediction JSONL")
    predictions: dict[str, NormalizedPrediction] = {}
    invalid: dict[str, str] = {}
    declared_durations: dict[str, int] = {}
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise EvaluationError(f"prediction JSONL is not UTF-8: {path}") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid prediction JSON on line {line_number}: {path}") from error
        if not isinstance(payload, dict):
            raise EvaluationError(f"prediction line {line_number} must be an object")
        video_id = payload.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            raise EvaluationError(f"prediction line {line_number} has no nonblank video_id")
        if video_id in predictions or video_id in invalid:
            raise EvaluationError(f"duplicate prediction video_id: {video_id}")
        declared_duration = payload.get("duration_ms")
        if type(declared_duration) is int and declared_duration > 0:
            declared_durations[video_id] = declared_duration
        try:
            predictions[video_id] = _normalize_prediction(payload)
        except EvaluationError:
            raise
        except (ValueError, ValidationError) as error:
            invalid[video_id] = str(error)
    return predictions, invalid, declared_durations, raw


def load_and_prevalidate_inputs(config: EvaluationConfig) -> LoadedInputs:
    manifest_bytes = _read_bytes(config.inputs.gold_set_path, "Gold manifest")
    payload = _load_json(manifest_bytes, config.inputs.gold_set_path, "Gold manifest")
    try:
        manifest = GoldSetManifest.model_validate(payload)
    except ValidationError as error:
        raise EvaluationError(f"invalid Gold manifest contract: {error}") from error

    gold_by_index, selected_bytes, failures = _parse_gold_annotations(
        config=config, manifest=manifest
    )
    (
        predictions,
        invalid_predictions,
        declared_prediction_durations,
        prediction_bytes,
    ) = _parse_predictions(config.inputs.prediction_path)

    included: list[GoldAnnotation] = []
    excluded: list[GoldAnnotation] = []
    incomplete: list[int] = []
    selected_indices = manifest.episode_selection.episode_indices
    seen_gold_video_ids: set[str] = set()
    for index in selected_indices:
        annotation = gold_by_index.get(index)
        if annotation is None:
            incomplete.append(index)
            failures.append(
                FailureRecord(
                    category="incomplete_gold",
                    severity="warning",
                    episode_index=index,
                    message="selected episode has no Gold annotation",
                )
            )
            continue
        if annotation.video_id in seen_gold_video_ids:
            raise EvaluationError(f"duplicate Gold video_id: {annotation.video_id}")
        seen_gold_video_ids.add(annotation.video_id)
        if annotation.human_annotation.gold_set_id != manifest.gold_set_id:
            raise EvaluationError(
                f"Gold annotation gold_set_id mismatch for episode {index}"
            )
        try:
            _validate_timeline(
                duration_ms=annotation.duration_ms,
                segments=annotation.segments,
                label=f"Gold episode {index}",
                require_full_coverage=manifest.ontology.require_full_coverage,
            )
            _validate_events(
                duration_ms=annotation.duration_ms,
                events=annotation.events,
                label=f"Gold episode {index}",
            )
            _gold_boundaries(
                annotation, config.metrics.boundary_error.event_ids
            )
        except ValueError as error:
            raise EvaluationError(str(error)) from error
        reason = annotation.human_annotation.exclusion_reason
        if reason is not None and reason.strip():
            excluded.append(annotation)
            failures.append(
                FailureRecord(
                    category="excluded_gold",
                    severity="info",
                    episode_index=index,
                    video_id=annotation.video_id,
                    message=reason.strip(),
                )
            )
        elif annotation.human_annotation.status in config.eligibility.included_statuses:
            included.append(annotation)
        else:
            incomplete.append(index)
            failures.append(
                FailureRecord(
                    category="incomplete_gold",
                    severity="warning",
                    episode_index=index,
                    video_id=annotation.video_id,
                    message=f"Gold status is {annotation.human_annotation.status!r}",
                )
            )

    if config.evaluation_mode == "official" and incomplete:
        raise EvaluationError(
            "official evaluation requires every selected episode to be complete or explicitly excluded; "
            f"incomplete indices: {sorted(set(incomplete))}"
        )

    included_video_ids = {annotation.video_id for annotation in included}
    extra_prediction_ids = sorted((set(predictions) | set(invalid_predictions)) - included_video_ids)
    for video_id in extra_prediction_ids:
        failures.append(
            FailureRecord(
                category="extra_prediction",
                severity="info",
                video_id=video_id,
                message="prediction is outside the included Gold set and was ignored",
            )
        )

    for annotation in included:
        video_id = annotation.video_id
        prediction = predictions.get(video_id)
        declared_duration = declared_prediction_durations.get(video_id)
        if (
            declared_duration is not None
            and declared_duration != annotation.duration_ms
        ):
            raise EvaluationError(
                f"duration mismatch for {video_id}: "
                f"Gold={annotation.duration_ms}, "
                f"prediction={declared_duration}"
            )
        if video_id in invalid_predictions:
            failures.append(
                FailureRecord(
                    category="invalid_prediction",
                    severity="error",
                    episode_index=annotation.human_annotation.episode_index,
                    video_id=video_id,
                    message=invalid_predictions[video_id],
                )
            )
            continue
        if prediction is None:
            failures.append(
                FailureRecord(
                    category="missing_prediction",
                    severity="error",
                    episode_index=annotation.human_annotation.episode_index,
                    video_id=video_id,
                    message="included Gold episode has no prediction",
                )
            )
            continue
        if prediction.duration_ms != annotation.duration_ms:
            raise EvaluationError(
                f"duration mismatch for {video_id}: Gold={annotation.duration_ms}, prediction={prediction.duration_ms}"
            )
        try:
            _validate_timeline(
                duration_ms=prediction.duration_ms,
                segments=prediction.segments,
                label=f"prediction {video_id}",
                require_full_coverage=False,
            )
            _validate_events(
                duration_ms=prediction.duration_ms,
                events=prediction.events,
                label=f"prediction {video_id}",
            )
            resolve_prediction_boundaries(
                prediction, config.metrics.boundary_error.event_ids
            )
            gold_ontology = manifest.ontology
            if prediction.ontology_id is not None and gold_ontology.id is not None and prediction.ontology_id != gold_ontology.id:
                raise ValueError("prediction ontology ID conflicts with Gold")
            if prediction.ontology_hash is not None and gold_ontology.hash is not None and prediction.ontology_hash != gold_ontology.hash:
                raise ValueError("prediction ontology hash conflicts with Gold")
        except ValueError as error:
            invalid_predictions[video_id] = str(error)
            predictions.pop(video_id, None)
            failures.append(
                FailureRecord(
                    category="invalid_prediction",
                    severity="error",
                    episode_index=annotation.human_annotation.episode_index,
                    video_id=video_id,
                    message=str(error),
                )
            )
        else:
            if not prediction.provenance_verified:
                failures.append(
                    FailureRecord(
                        category="unverified_prediction_provenance",
                        severity="warning",
                        episode_index=annotation.human_annotation.episode_index,
                        video_id=video_id,
                        message="prediction does not identify its ontology provenance",
                    )
                )

    return LoadedInputs(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        selected_annotation_bytes=selected_bytes,
        included=tuple(sorted(included, key=lambda item: item.human_annotation.episode_index)),
        excluded=tuple(sorted(excluded, key=lambda item: item.human_annotation.episode_index)),
        incomplete_indices=tuple(sorted(set(incomplete))),
        predictions=predictions,
        invalid_predictions=invalid_predictions,
        declared_prediction_durations=declared_prediction_durations,
        prediction_bytes=prediction_bytes,
        failures=tuple(failures),
        extra_prediction_ids=tuple(extra_prediction_ids),
    )


def gold_boundary_map(annotation: GoldAnnotation, event_ids: list[str]) -> dict[str, int]:
    return _gold_boundaries(annotation, event_ids)
