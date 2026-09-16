import json

import pytest
from pydantic import ValidationError

from auto_annotation.domain.models import (
    FinalizedAnnotation,
    Provenance,
    Quality,
    Segment,
)


def segment(segment_id: str, start_ms: int, end_ms: int) -> Segment:
    return Segment(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        values={
            "atomic_action": "grasp",
            "interaction_object": "cup",
            "target": "杯柄",
        },
        sentence="抓住杯子的把手。",
        evidence_timestamps_ms=[start_ms],
    )


def test_final_annotation_allows_gaps() -> None:
    annotation = FinalizedAnnotation(
        video_id="video-1",
        duration_ms=10_000,
        task_summary="拿起杯子",
        segments=[segment("s1", 1000, 2000), segment("s2", 4000, 5000)],
        quality=Quality(score=1.0),
    )
    assert [item.start_ms for item in annotation.segments] == [1000, 4000]


def test_final_annotation_rejects_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        FinalizedAnnotation(
            video_id="video-1",
            duration_ms=10_000,
            task_summary="拿起杯子",
            segments=[segment("s1", 1000, 3000), segment("s2", 2500, 5000)],
            quality=Quality(score=1.0),
        )


def test_segment_rejects_empty_interval() -> None:
    with pytest.raises(ValidationError, match="end_ms"):
        segment("s1", 1000, 1000)


def test_dashscope_provenance_contains_identity_without_secret_fields() -> None:
    provenance = Provenance(
        schema_version="schema-v1",
        ontology_hash="sha256:ontology",
        prompt_versions={"dashscope": "prompt-v1"},
        model_id="qwen3-vl-plus",
        model_revision="qwen3-vl-plus",
        config_hash="sha256:config",
        backend_kind="openai_compatible",
        backend_profile="dashscope_qwen_vision",
        base_url="https://workspace-id.example.test/v1",
        adapter_version="adapter-v1",
        materializer_version="materializer-v1",
        runner_version="runner-v1",
        manifest_sha256="sha256:manifest",
        video_sha256={"video-1": "sha256:video"},
    )
    serialized = provenance.model_dump(mode="json")
    assert serialized["backend_profile"] == "dashscope_qwen_vision"
    assert "api_key" not in serialized
    assert "synthetic-secret" not in json.dumps(serialized)
