from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def segment(segment_id: str, start: int, end: int) -> dict:
    return {
        "segment_id": segment_id,
        "start_ms": start,
        "end_ms": end,
        "values": {
            "atomic_action": segment_id,
            "interaction_object": "workpiece",
            "target": "bin",
        },
        "sentence": segment_id,
        "evidence_timestamps_ms": [],
    }


def annotation(index: int, *, status: str = "complete", exclusion_reason=None) -> dict:
    video_id = f"episode_{index:06d}_multiview"
    return {
        "video_id": video_id,
        "duration_ms": 3000,
        "task_summary": "pick and place",
        "events": [
            {"event_id": "transfer", "timestamp_ms": 1000, "values": {}, "sentence": "transfer"},
            {"event_id": "place", "timestamp_ms": 2000, "values": {}, "sentence": "place"},
        ],
        "segments": [
            segment("pick_up", 0, 1000),
            segment("transfer", 1000, 2000),
            segment("place", 2000, 3000),
        ],
        "human_annotation": {
            "version": 1,
            "gold_set_id": "synthetic-v1",
            "episode_index": index,
            "status": status,
            "exclusion_reason": exclusion_reason,
        },
    }


def prediction(index: int, *, transfer: int = 1100, place: int = 1800) -> dict:
    video_id = f"episode_{index:06d}_multiview"
    return {
        "video_id": video_id,
        "duration_ms": 3000,
        "task_summary": "pick and place",
        "events": [
            {"event_id": "transfer", "timestamp_ms": transfer, "values": {}, "sentence": "transfer"},
            {"event_id": "place", "timestamp_ms": place, "values": {}, "sentence": "place"},
        ],
        "segments": [
            segment("pick_up", 0, transfer),
            segment("transfer", transfer, place),
            segment("place", place, 3000),
        ],
        "experiment": {"ontology_id": "pick-place-v1", "ontology_hash": "sha256:ontology"},
    }


def config_payload(root: Path, *, mode: str = "official") -> dict:
    return {
        "version": 1,
        "profile_id": "synthetic-boundary-v1",
        "inputs": {
            "gold_set_path": "gold_set.json",
            "annotations_dir": "annotations",
            "prediction_path": "predictions.jsonl",
        },
        "evaluation_mode": mode,
        "eligibility": {
            "require_complete_gold_set": True,
            "included_statuses": ["complete"],
            "missing_gold_annotation": "fail",
            "missing_prediction": "count_as_missing",
            "extra_prediction": "ignore_and_report",
            "duplicate_video_id": "fail",
            "duration_mismatch": "fail",
        },
        "metrics": {
            "boundary_error": {
                "enabled": True,
                "event_ids": ["transfer", "place"],
                "thresholds_ms": [100, 500, 1000],
            },
            "temporal_iou": {"enabled": True, "matching": "segment_id", "thresholds": [0.3, 0.5, 0.7]},
            "sequence_edit_distance": {"enabled": False, "reason": "fixed"},
            "label_accuracy": {"enabled": False, "reason": "fixed"},
            "segmentation_errors": {"enabled": False, "reason": "fixed"},
            "semantic_quality": {"enabled": False, "implementation": "deferred"},
            "gold_reliability": {"enabled": False, "implementation": "deferred"},
        },
        "aggregation": {
            "primary": "macro_episode",
            "also_report": ["micro", "per_event", "per_segment"],
            "percentiles": [50, 90],
            "report_max": True,
        },
        "missing_predictions": {
            "report_coverage": True,
            "boundary_mae": "exclude_with_count",
            "boundary_hit_rate": "count_as_miss",
        },
        "uncertainty": {
            "enabled": True,
            "method": "episode_bootstrap",
            "confidence_level": 0.95,
            "resamples": 100,
            "seed": 7,
        },
        "decision": {"mode": "report_only", "composite_score": None, "thresholds": None},
        "review_queue": {
            "enabled": True,
            "top_k": 20,
            "order_by": ["missing_prediction", "max_boundary_error_ms", "mean_tiou_ascending"],
        },
        "output": {
            "root": str(root / "evaluation-output"),
            "write_summary_json": True,
            "write_episodes_jsonl": True,
            "write_failures_jsonl": True,
            "write_review_queue_jsonl": True,
            "write_markdown_report": True,
        },
    }


def write_fixture(
    root: Path,
    *,
    indices=(1, 2),
    annotation_indices=None,
    predictions=None,
    mode="official",
) -> Path:
    (root / "annotations").mkdir()
    manifest = {
        "version": 1,
        "gold_set_id": "synthetic-v1",
        "episode_selection": {"episode_indices": list(indices), "sample_size": len(indices)},
        "ontology": {
            "id": "pick-place-v1",
            "hash": "sha256:ontology",
            "interval_semantics": "[start_ms,end_ms)",
            "ordered_segments": [],
            "require_full_coverage": True,
        },
    }
    (root / "gold_set.json").write_text(json.dumps(manifest), encoding="utf-8")
    actual_annotations = indices if annotation_indices is None else annotation_indices
    for index in actual_annotations:
        (root / "annotations" / f"arbitrary-{index}.json").write_text(
            json.dumps(annotation(index)), encoding="utf-8"
        )
    actual_predictions = [prediction(index) for index in indices] if predictions is None else predictions
    (root / "predictions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in actual_predictions), encoding="utf-8"
    )
    path = root / "evaluation.yaml"
    path.write_text(yaml.safe_dump(config_payload(root, mode=mode), sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_fixture(tmp_path: Path) -> Path:
    return write_fixture(tmp_path)

