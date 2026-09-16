"""Deterministic machine and human-readable evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Sequence

from auto_annotation.evaluation.config import EvaluationConfig
from auto_annotation.evaluation.models import (
    EpisodeEvaluation,
    EvaluationSummary,
    FailureRecord,
    ReviewQueueItem,
)


def _json_bytes(value: object, *, indent: int | None = 2) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Sequence[object]) -> bytes:
    return b"".join(_json_bytes(value, indent=None) for value in values)


def build_review_queue(
    config: EvaluationConfig, episodes: Sequence[EpisodeEvaluation]
) -> list[ReviewQueueItem]:
    def key(episode: EpisodeEvaluation) -> tuple[object, ...]:
        status_priority = 0 if episode.status in {"missing_prediction", "invalid_prediction"} else 1
        boundary_key = -episode.max_boundary_error_ms if episode.max_boundary_error_ms is not None else float("inf")
        return (status_priority, boundary_key, episode.mean_tiou, episode.video_id)

    queue: list[ReviewQueueItem] = []
    for rank, episode in enumerate(sorted(episodes, key=key)[: config.review_queue.top_k], start=1):
        reasons: list[str] = []
        if episode.status != "evaluated":
            reasons.append(episode.status)
        if episode.max_boundary_error_ms is not None:
            reasons.append(f"max_boundary_error_ms={episode.max_boundary_error_ms}")
        reasons.append(f"mean_tiou={episode.mean_tiou:.6f}")
        queue.append(
            ReviewQueueItem(
                rank=rank,
                episode_index=episode.episode_index,
                video_id=episode.video_id,
                prediction_status=episode.status,
                max_boundary_error_ms=episode.max_boundary_error_ms,
                mean_tiou=episode.mean_tiou,
                reasons=reasons,
            )
        )
    return queue


def _markdown(summary: EvaluationSummary) -> bytes:
    eligibility = summary.eligibility
    boundary = summary.metrics["boundary_error"]
    tiou = summary.metrics["temporal_iou"]
    marker = "\n> **PARTIAL EVALUATION — not an official baseline.**\n" if summary.evaluation_mode == "partial" else ""
    lines = [
        "# Gold Set Evaluation",
        marker.rstrip(),
        f"- Evaluation ID: `{summary.evaluation_id}`",
        f"- Profile: `{summary.profile_id}`",
        f"- Mode: **{summary.report_label}**",
        f"- Gold progress: {eligibility.complete}/{eligibility.selected} complete; {eligibility.included} included",
        f"- Prediction coverage: {summary.prediction_coverage['valid']}/{summary.prediction_coverage['total']} ({summary.prediction_coverage['rate']:.2%})",
        "",
        "## Metrics",
        "",
        f"- Boundary error: `{boundary.status}`",
        f"- Temporal IoU: `{tiou.status}`",
        f"- Sequence edit distance: `{summary.metrics['sequence_edit_distance'].status}`",
        f"- Label accuracy: `{summary.metrics['label_accuracy'].status}`",
        f"- Segmentation errors: `{summary.metrics['segmentation_errors'].status}`",
        f"- Semantic quality: `{summary.metrics['semantic_quality'].status}`",
        f"- Gold reliability: `{summary.metrics['gold_reliability'].status}`",
        "",
        "No composite score, quality gate, or model pass/fail decision is produced.",
    ]
    return ("\n".join(line for line in lines if line is not None) + "\n").encode("utf-8")


def build_report_files(
    *,
    config: EvaluationConfig,
    summary: EvaluationSummary,
    episodes: Sequence[EpisodeEvaluation],
    failures: Sequence[FailureRecord],
) -> dict[str, bytes]:
    files = {"summary.json": _json_bytes(summary.model_dump(mode="json"))}
    if config.output.write_episodes_jsonl:
        files["episodes.jsonl"] = _jsonl_bytes(
            [
                episode.model_dump(mode="json")
                for episode in sorted(episodes, key=lambda item: (item.episode_index, item.video_id))
            ]
        )
    if config.output.write_failures_jsonl:
        ordered_failures = sorted(
            failures,
            key=lambda item: (
                item.category,
                -1 if item.episode_index is None else item.episode_index,
                item.video_id or "",
                item.message,
            ),
        )
        files["failures.jsonl"] = _jsonl_bytes(
            [item.model_dump(mode="json") for item in ordered_failures]
        )
    if config.output.write_review_queue_jsonl:
        queue = build_review_queue(config, episodes) if config.review_queue.enabled else []
        files["review_queue.jsonl"] = _jsonl_bytes(
            [item.model_dump(mode="json") for item in queue]
        )
    if config.output.write_markdown_report:
        files["report.md"] = _markdown(summary)
    return files

