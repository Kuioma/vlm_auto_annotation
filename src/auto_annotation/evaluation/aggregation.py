"""Episode-aware metric aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from auto_annotation.evaluation.bootstrap import (
    episode_bootstrap,
    quantile,
    suppressed_interval,
)
from auto_annotation.evaluation.config import EvaluationConfig
from auto_annotation.evaluation.models import (
    ConfidenceInterval,
    EpisodeEvaluation,
    MetricResult,
)
from auto_annotation.evaluation.versions import METRIC_VERSIONS


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(values) / len(values) if values else None


def _distribution(values: Sequence[float], config: EvaluationConfig) -> dict[str, float | None]:
    result = {
        f"p{percentile:g}": quantile(values, percentile)
        for percentile in config.aggregation.percentiles
    }
    if config.aggregation.report_max:
        result["max"] = max(values) if values else None
    return result


def _boundary_stat(
    episodes: Sequence[EpisodeEvaluation],
    *,
    kind: str,
    event_id: str | None = None,
    threshold: int | None = None,
) -> float | None:
    per_episode: list[float] = []
    for episode in episodes:
        boundaries = (
            [episode.boundaries[event_id]]
            if event_id is not None
            else list(episode.boundaries.values())
        )
        if kind in {"mae", "bias"}:
            field = "absolute_error_ms" if kind == "mae" else "signed_error_ms"
            values = [float(item[field]) for item in boundaries if item[field] is not None]
            if values:
                per_episode.append(sum(values) / len(values))
        elif kind == "hit":
            assert threshold is not None
            per_episode.append(
                sum(bool(item["hits"][str(threshold)]) for item in boundaries)
                / len(boundaries)
            )
        else:
            raise AssertionError(kind)
    return _mean(per_episode)


def _ci(
    config: EvaluationConfig,
    episodes: Sequence[EpisodeEvaluation],
    statistic: Callable[[Sequence[EpisodeEvaluation]], float | None],
    *,
    seed_offset: int,
) -> ConfidenceInterval:
    if config.evaluation_mode == "partial":
        return suppressed_interval("partial evaluations do not publish confidence intervals")
    if not config.uncertainty.enabled:
        return suppressed_interval("uncertainty is disabled")
    return episode_bootstrap(
        episodes,
        statistic,
        confidence_level=config.uncertainty.confidence_level,
        resamples=config.uncertainty.resamples,
        seed=config.uncertainty.seed,
    )


def aggregate_boundary(
    config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]
) -> MetricResult:
    metric_config = config.metrics.boundary_error
    if not metric_config.enabled:
        return MetricResult(
            status="skipped",
            version=METRIC_VERSIONS["boundary_error"],
            reason=metric_config.reason or "disabled by configuration",
            parameters={
                "event_ids": metric_config.event_ids,
                "thresholds_ms": metric_config.thresholds_ms,
            },
        )
    absolute: list[float] = []
    signed: list[float] = []
    per_event_values: dict[str, dict[str, list[float]]] = {
        event_id: {"absolute": [], "signed": []}
        for event_id in metric_config.event_ids
    }
    total = 0
    hit_counts = {threshold: 0 for threshold in metric_config.thresholds_ms}
    episode_mae: list[float] = []
    episode_bias: list[float] = []
    episode_hit: dict[int, list[float]] = defaultdict(list)
    for episode in episodes:
        ep_abs: list[float] = []
        ep_signed: list[float] = []
        for event_id, detail in episode.boundaries.items():
            total += 1
            if detail["absolute_error_ms"] is not None:
                value = float(detail["absolute_error_ms"])
                absolute.append(value)
                ep_abs.append(value)
                per_event_values[event_id]["absolute"].append(value)
            if detail["signed_error_ms"] is not None:
                value = float(detail["signed_error_ms"])
                signed.append(value)
                ep_signed.append(value)
                per_event_values[event_id]["signed"].append(value)
            for threshold in metric_config.thresholds_ms:
                hit = bool(detail["hits"][str(threshold)])
                hit_counts[threshold] += int(hit)
        if ep_abs:
            episode_mae.append(sum(ep_abs) / len(ep_abs))
            episode_bias.append(sum(ep_signed) / len(ep_signed))
        for threshold in metric_config.thresholds_ms:
            episode_hit[threshold].append(
                sum(
                    bool(detail["hits"][str(threshold)])
                    for detail in episode.boundaries.values()
                )
                / len(episode.boundaries)
            )
    per_event: dict[str, Any] = {}
    for event_id, values in per_event_values.items():
        event_total = len(episodes)
        event_hits = {
            str(threshold): sum(
                bool(episode.boundaries[event_id]["hits"][str(threshold)])
                for episode in episodes
            )
            for threshold in metric_config.thresholds_ms
        }
        per_event[event_id] = {
            "mae_ms": _mean(values["absolute"]),
            "signed_bias_ms": _mean(values["signed"]),
            "observed": len(values["absolute"]),
            "missing": event_total - len(values["absolute"]),
            "hit_rate": {
                key: count / event_total if event_total else 0.0
                for key, count in event_hits.items()
            },
        }
    aggregation = {
        "macro_episode": {
            "mae_ms": _mean(episode_mae),
            "signed_bias_ms": _mean(episode_bias),
            "hit_rate": {
                str(threshold): _mean(episode_hit[threshold])
                for threshold in metric_config.thresholds_ms
            },
        },
        "micro": {
            "mae_ms": _mean(absolute),
            "signed_bias_ms": _mean(signed),
            "hit_rate": {
                str(threshold): hit_counts[threshold] / total if total else 0.0
                for threshold in metric_config.thresholds_ms
            },
        },
        "per_event": per_event,
        "episode_distribution": {
            "mae_ms": _distribution(episode_mae, config),
            "signed_bias_ms": _distribution(episode_bias, config),
            "hit_rate": {
                str(threshold): _distribution(episode_hit[threshold], config)
                for threshold in metric_config.thresholds_ms
            },
        },
    }
    cis: dict[str, ConfidenceInterval] = {
        "macro_episode.mae_ms": _ci(
            config,
            episodes,
            lambda sample: _boundary_stat(sample, kind="mae"),
            seed_offset=1,
        ),
        "macro_episode.signed_bias_ms": _ci(
            config,
            episodes,
            lambda sample: _boundary_stat(sample, kind="bias"),
            seed_offset=2,
        ),
    }
    offset = 10
    for threshold in metric_config.thresholds_ms:
        cis[f"macro_episode.hit_rate.{threshold}"] = _ci(
            config,
            episodes,
            lambda sample, threshold=threshold: _boundary_stat(
                sample, kind="hit", threshold=threshold
            ),
            seed_offset=offset,
        )
        offset += 1
    for event_id in metric_config.event_ids:
        cis[f"per_event.{event_id}.mae_ms"] = _ci(
            config,
            episodes,
            lambda sample, event_id=event_id: _boundary_stat(
                sample, kind="mae", event_id=event_id
            ),
            seed_offset=offset,
        )
        offset += 1
        for threshold in metric_config.thresholds_ms:
            cis[f"per_event.{event_id}.hit_rate.{threshold}"] = _ci(
                config,
                episodes,
                lambda sample, event_id=event_id, threshold=threshold: _boundary_stat(
                    sample, kind="hit", event_id=event_id, threshold=threshold
                ),
                seed_offset=offset,
            )
            offset += 1
    return MetricResult(
        status="computed",
        version=METRIC_VERSIONS["boundary_error"],
        parameters={
            "event_ids": metric_config.event_ids,
            "thresholds_ms": metric_config.thresholds_ms,
            "hit_comparison": "absolute_error_ms <= threshold_ms",
        },
        aggregation=aggregation,
        denominator={
            "episodes": len(episodes),
            "gold_boundaries": total,
            "observed_boundaries": len(absolute),
        },
        missing_count=total - len(absolute),
        confidence_intervals=cis,
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def aggregate_temporal_iou(
    config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]
) -> MetricResult:
    metric_config = config.metrics.temporal_iou
    if not metric_config.enabled:
        return MetricResult(
            status="skipped",
            version=METRIC_VERSIONS["temporal_iou"],
            reason=metric_config.reason or "disabled by configuration",
        )
    all_scores: list[float] = []
    per_segment: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for segment_id, detail in episode.segments.items():
            score = float(detail["tiou"])
            all_scores.append(score)
            per_segment[segment_id].append(score)
    micro_counts: dict[str, dict[str, int | float]] = {}
    macro_counts: dict[str, dict[str, float]] = {}
    for threshold in metric_config.thresholds:
        key = str(threshold)
        tp = sum(episode.temporal_counts[key]["tp"] for episode in episodes)
        fp = sum(episode.temporal_counts[key]["fp"] for episode in episodes)
        fn = sum(episode.temporal_counts[key]["fn"] for episode in episodes)
        micro_counts[key] = _prf(tp, fp, fn)
        episode_rates = [
            _prf(
                episode.temporal_counts[key]["tp"],
                episode.temporal_counts[key]["fp"],
                episode.temporal_counts[key]["fn"],
            )
            for episode in episodes
        ]
        macro_counts[key] = {
            metric: _mean([float(rate[metric]) for rate in episode_rates]) or 0.0
            for metric in ("precision", "recall", "f1")
        }
    episode_scores = [episode.mean_tiou for episode in episodes]
    ci = _ci(
        config,
        episodes,
        lambda sample: _mean([episode.mean_tiou for episode in sample]),
        seed_offset=1000,
    )
    return MetricResult(
        status="computed",
        version=METRIC_VERSIONS["temporal_iou"],
        parameters={"matching": "segment_id", "thresholds": metric_config.thresholds},
        aggregation={
            "macro_episode": {"mean_tiou": _mean(episode_scores), "thresholds": macro_counts},
            "micro": {"mean_tiou": _mean(all_scores), "thresholds": micro_counts},
            "per_segment": {
                segment_id: {"mean_tiou": _mean(scores), "count": len(scores)}
                for segment_id, scores in sorted(per_segment.items())
            },
            "episode_distribution": {"mean_tiou": _distribution(episode_scores, config)},
        },
        denominator={
            "episodes": len(episodes),
            "gold_segments": len(all_scores),
            "extra_prediction_segments": sum(len(item.extra_segment_ids) for item in episodes),
        },
        missing_count=sum(
            detail["status"] == "missing"
            for episode in episodes
            for detail in episode.segments.values()
        ),
        confidence_intervals={"macro_episode.mean_tiou": ci},
    )


def aggregate_sequence(config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]) -> MetricResult:
    metric_config = config.metrics.sequence_edit_distance
    if not metric_config.enabled:
        return MetricResult(
            status="skipped",
            version=METRIC_VERSIONS["sequence_edit_distance"],
            reason=metric_config.reason or "disabled by configuration",
            parameters={"field": metric_config.field},
        )
    details = [episode.sequence for episode in episodes if episode.sequence is not None]
    raw = [float(item["distance"]) for item in details]
    normalized = [float(item["normalized_distance"]) for item in details]
    total_gold = sum(int(item["gold_length"]) for item in details)
    normalized_denominator = sum(
        max(int(item["gold_length"]), int(item["prediction_length"]), 1)
        for item in details
    )
    return MetricResult(
        status="computed",
        version=METRIC_VERSIONS["sequence_edit_distance"],
        parameters={"field": metric_config.field},
        aggregation={
            "macro_episode": {
                "distance": _mean(raw),
                "normalized_distance": _mean(normalized),
                "exact_sequence_rate": _mean([float(bool(item["exact"])) for item in details]),
            },
            "micro": {
                "distance": sum(raw),
                "normalized_distance": sum(raw) / max(normalized_denominator, 1),
                "exact_sequence_rate": _mean([float(bool(item["exact"])) for item in details]),
            },
            "episode_distribution": {
                "distance": _distribution(raw, config),
                "normalized_distance": _distribution(normalized, config),
            },
        },
        denominator={
            "episodes": len(details),
            "gold_tokens": total_gold,
            "normalization_tokens": normalized_denominator,
        },
    )


def aggregate_labels(config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]) -> MetricResult:
    metric_config = config.metrics.label_accuracy
    if not metric_config.enabled:
        return MetricResult(
            status="skipped",
            version=METRIC_VERSIONS["label_accuracy"],
            reason=metric_config.reason or "disabled by configuration",
            parameters={"fields": metric_config.fields, "exact_match_groups": metric_config.exact_match_groups},
        )
    field_totals = {field: {"correct": 0, "total": 0} for field in metric_config.fields}
    group_totals = {"+".join(group): {"correct": 0, "total": 0} for group in metric_config.exact_match_groups}
    missing = extras = 0
    for episode in episodes:
        assert episode.labels is not None
        missing += int(episode.labels["missing_pairs"])
        extras += int(episode.labels["extra_predictions"])
        for field, counts in episode.labels["fields"].items():
            field_totals[field]["correct"] += int(counts["correct"])
            field_totals[field]["total"] += int(counts["total"])
        for group, counts in episode.labels["groups"].items():
            group_totals[group]["correct"] += int(counts["correct"])
            group_totals[group]["total"] += int(counts["total"])
    for counts in [*field_totals.values(), *group_totals.values()]:
        counts["accuracy"] = counts["correct"] / counts["total"] if counts["total"] else 0.0
    return MetricResult(
        status="computed",
        version=METRIC_VERSIONS["label_accuracy"],
        parameters={"fields": metric_config.fields, "exact_match_groups": metric_config.exact_match_groups},
        aggregation={"micro": {"fields": field_totals, "groups": group_totals}},
        denominator={"episodes": len(episodes), "gold_segments": sum(item["total"] for item in field_totals.values()) // max(len(field_totals), 1), "extra_prediction_segments": extras},
        missing_count=missing,
    )


def aggregate_segmentation(config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]) -> MetricResult:
    metric_config = config.metrics.segmentation_errors
    if not metric_config.enabled:
        return MetricResult(
            status="skipped",
            version=METRIC_VERSIONS["segmentation_errors"],
            reason=metric_config.reason or "disabled by configuration",
            parameters={"substantial_overlap_threshold": metric_config.substantial_overlap_threshold},
        )
    keys = ("missed_segment_count", "extra_segment_count", "over_split_count", "incorrect_merge_count")
    totals = {key: 0 for key in keys}
    affected = {key: 0 for key in keys}
    for episode in episodes:
        assert episode.segmentation is not None
        for key in keys:
            value = int(episode.segmentation[key])
            totals[key] += value
            affected[key] += int(value > 0)
    return MetricResult(
        status="computed",
        version=METRIC_VERSIONS["segmentation_errors"],
        parameters={"substantial_overlap_threshold": metric_config.substantial_overlap_threshold},
        aggregation={
            "totals": totals,
            "affected_episodes": affected,
            "episode_rates": {
                key: affected[key] / len(episodes) if episodes else 0.0 for key in keys
            },
        },
        denominator={"episodes": len(episodes)},
    )


def deferred_metric(name: str, reason: str | None) -> MetricResult:
    return MetricResult(
        status="not_implemented",
        version=METRIC_VERSIONS[name],
        reason=reason or "capability is deferred",
    )


def aggregate_all(config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]) -> dict[str, MetricResult]:
    return {
        "boundary_error": aggregate_boundary(config, episodes),
        "temporal_iou": aggregate_temporal_iou(config, episodes),
        "sequence_edit_distance": aggregate_sequence(config, episodes),
        "label_accuracy": aggregate_labels(config, episodes),
        "segmentation_errors": aggregate_segmentation(config, episodes),
        "semantic_quality": deferred_metric("semantic_quality", config.metrics.semantic_quality.reason),
        "gold_reliability": deferred_metric("gold_reliability", config.metrics.gold_reliability.reason),
    }
