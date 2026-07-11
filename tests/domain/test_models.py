import pytest
from pydantic import ValidationError

from auto_annotation.domain.models import FinalizedAnnotation, Quality, Segment


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
