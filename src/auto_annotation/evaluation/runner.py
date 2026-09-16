"""End-to-end Gold-set evaluator orchestration."""

from __future__ import annotations

from collections import Counter

from auto_annotation.evaluation.aggregation import aggregate_all
from auto_annotation.evaluation.config import EvaluationConfig, evaluation_config_hash
from auto_annotation.evaluation.identity import build_evaluation_identity
from auto_annotation.evaluation.inputs import (
    gold_boundary_map,
    load_and_prevalidate_inputs,
    resolve_prediction_boundaries,
)
from auto_annotation.evaluation.metrics.boundary import evaluate_boundaries
from auto_annotation.evaluation.metrics.labels import evaluate_labels
from auto_annotation.evaluation.metrics.segmentation import evaluate_segmentation
from auto_annotation.evaluation.metrics.sequence import evaluate_sequence
from auto_annotation.evaluation.metrics.temporal_iou import evaluate_temporal_iou
from auto_annotation.evaluation.models import (
    EligibilitySummary,
    EpisodeEvaluation,
    EvaluationSummary,
)
from auto_annotation.evaluation.reporting import build_report_files
from auto_annotation.evaluation.store import publish_evaluation, validate_output_topology
from auto_annotation.evaluation.versions import (
    EVALUATOR_VERSION,
    METRIC_VERSIONS,
    RESULT_SCHEMA_VERSION,
)


def _evaluate_episodes(config: EvaluationConfig, loaded) -> list[EpisodeEvaluation]:
    episodes: list[EpisodeEvaluation] = []
    for gold in loaded.included:
        video_id = gold.video_id
        prediction = loaded.predictions.get(video_id)
        invalid = video_id in loaded.invalid_predictions
        status = "invalid_prediction" if invalid else ("missing_prediction" if prediction is None else "evaluated")
        if invalid:
            provenance_status = "invalid"
        elif prediction is None:
            provenance_status = "missing"
        elif prediction.provenance_verified:
            provenance_status = "verified"
        else:
            provenance_status = "unverified"
        gold_boundaries = gold_boundary_map(gold, config.metrics.boundary_error.event_ids)
        predicted_boundaries = (
            None
            if prediction is None
            else resolve_prediction_boundaries(
                prediction, config.metrics.boundary_error.event_ids
            )
        )
        boundaries = evaluate_boundaries(
            gold_boundaries,
            predicted_boundaries,
            config.metrics.boundary_error.thresholds_ms,
        )
        segment_details, temporal_counts, extras = evaluate_temporal_iou(
            gold.segments,
            None if prediction is None else prediction.segments,
            config.metrics.temporal_iou.thresholds,
        )
        sequence = (
            evaluate_sequence(
                gold.segments,
                None if prediction is None else prediction.segments,
                config.metrics.sequence_edit_distance.field,
            )
            if config.metrics.sequence_edit_distance.enabled
            else None
        )
        labels = (
            evaluate_labels(
                gold.segments,
                None if prediction is None else prediction.segments,
                config.metrics.label_accuracy.fields,
                config.metrics.label_accuracy.exact_match_groups,
            )
            if config.metrics.label_accuracy.enabled
            else None
        )
        segmentation = (
            evaluate_segmentation(
                gold.segments,
                None if prediction is None else prediction.segments,
                config.metrics.segmentation_errors.substantial_overlap_threshold,
            )
            if config.metrics.segmentation_errors.enabled
            else None
        )
        observed_errors = [
            int(detail["absolute_error_ms"])
            for detail in boundaries.values()
            if detail["absolute_error_ms"] is not None
        ]
        scores = [float(detail["tiou"]) for detail in segment_details.values()]
        episodes.append(
            EpisodeEvaluation(
                episode_index=gold.human_annotation.episode_index,
                video_id=video_id,
                status=status,
                provenance_status=provenance_status,
                boundaries=boundaries,
                segments=segment_details,
                temporal_counts=temporal_counts,
                extra_segment_ids=extras,
                sequence=sequence,
                labels=labels,
                segmentation=segmentation,
                max_boundary_error_ms=max(observed_errors) if observed_errors else None,
                mean_tiou=sum(scores) / len(scores) if scores else 0.0,
            )
        )
    return episodes


def run_evaluation(config: EvaluationConfig) -> EvaluationSummary:
    validate_output_topology(config)
    loaded = load_and_prevalidate_inputs(config)
    versions = {
        "evaluator_version": EVALUATOR_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "metric_versions": METRIC_VERSIONS,
    }
    evaluation_id, identity = build_evaluation_identity(
        config=config,
        versions=versions,
        gold_manifest_bytes=loaded.manifest_bytes,
        selected_annotation_bytes=loaded.selected_annotation_bytes,
        prediction_bytes=loaded.prediction_bytes,
    )
    validate_output_topology(config, evaluation_id)
    episodes = _evaluate_episodes(config, loaded)
    metrics = aggregate_all(config, episodes)
    included_ids = {item.video_id for item in loaded.included}
    invalid_count = len(included_ids & set(loaded.invalid_predictions))
    valid_count = len(included_ids & set(loaded.predictions))
    missing_count = len(included_ids) - valid_count - invalid_count
    eligibility = EligibilitySummary(
        selected=len(loaded.manifest.episode_selection.episode_indices),
        complete=len(loaded.included),
        included=len(loaded.included),
        excluded=len(loaded.excluded),
        incomplete=len(loaded.incomplete_indices),
        predicted=valid_count,
        missing_predictions=missing_count,
        invalid_predictions=invalid_count,
        extra_predictions=len(loaded.extra_prediction_ids),
        exclusion_reasons=dict(
            sorted(
                Counter(
                    item.human_annotation.exclusion_reason.strip()
                    for item in loaded.excluded
                    if item.human_annotation.exclusion_reason is not None
                    and item.human_annotation.exclusion_reason.strip()
                ).items()
            )
        ),
    )
    summary = EvaluationSummary(
        result_schema_version=RESULT_SCHEMA_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        evaluation_id=evaluation_id,
        profile_id=config.profile_id,
        evaluation_mode=config.evaluation_mode,
        report_label="PARTIAL" if config.evaluation_mode == "partial" else "OFFICIAL",
        config_sha256=evaluation_config_hash(config),
        gold_manifest_sha256=identity["gold_manifest_sha256"],
        included_annotations_sha256=identity["included_annotations_sha256"],
        prediction_sha256=identity["prediction_sha256"],
        metric_versions=METRIC_VERSIONS,
        eligibility=eligibility,
        prediction_coverage={
            "valid": valid_count,
            "invalid": invalid_count,
            "missing": missing_count,
            "total": len(loaded.included),
            "rate": valid_count / len(loaded.included) if loaded.included else 0.0,
        },
        prediction_schema_validity={
            "valid": valid_count,
            "invalid": invalid_count,
            "total_present": valid_count + invalid_count,
            "rate": (
                valid_count / (valid_count + invalid_count)
                if valid_count + invalid_count
                else 0.0
            ),
        },
        metrics=metrics,
    )
    files = build_report_files(
        config=config,
        summary=summary,
        episodes=episodes,
        failures=loaded.failures,
    )
    publish_evaluation(config, evaluation_id, files)
    return summary
