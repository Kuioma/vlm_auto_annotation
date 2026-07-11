from pathlib import Path

import pytest

from auto_annotation.config.models import TimelineConfig
from auto_annotation.domain.models import CoarseAnnotation, Segment
from auto_annotation.ontology.registry import ResolvedContract, load_contract
from auto_annotation.pipeline.finalize import TimelineConflict, finalize_annotation


FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def contract() -> ResolvedContract:
    return load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )


def make_segment(
    segment_id: str,
    start_ms: int,
    end_ms: int,
    action: str,
    *,
    sentence: str | None = None,
    evidence_timestamps_ms: list[int] | None = None,
) -> Segment:
    return Segment(
        segment_id=segment_id,
        start_ms=start_ms,
        end_ms=end_ms,
        values={
            "atomic_action": action,
            "interaction_object": "cup",
            "target": "托盘",
        },
        sentence=sentence
        if sentence is not None
        else ("抓住杯子。" if action == "grasp" else "将杯子放到托盘。"),
        evidence_timestamps_ms=evidence_timestamps_ms or [],
    )


def test_finalize_sorts_and_merges_same_values_with_short_gap(
    contract: ResolvedContract,
) -> None:
    refined = [
        make_segment("s3", 4000, 5000, "place"),
        make_segment(
            "s2",
            2250,
            3000,
            "grasp",
            evidence_timestamps_ms=[2800, 2300],
        ),
        make_segment(
            "s1",
            1000,
            2000,
            "grasp",
            evidence_timestamps_ms=[1500, 1500],
        ),
    ]
    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(task_summary="移动杯子", segments=refined),
        refined=refined,
        contract=contract,
        timeline=TimelineConfig(merge_gap_ms=250),
    )

    assert [segment.segment_id for segment in document.segments] == ["s1", "s3"]
    assert (document.segments[0].start_ms, document.segments[0].end_ms) == (
        1000,
        3000,
    )
    assert document.segments[0].evidence_timestamps_ms == [1500, 2300, 2800]
    assert document.quality.signals["merged_segment_sources"] == {
        "s1": ["s1", "s2"]
    }


def test_finalize_rejects_mismatched_segment_ids(
    contract: ResolvedContract,
) -> None:
    with pytest.raises(ValueError, match=r"segment IDs"):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=CoarseAnnotation(
                task_summary="移动杯子",
                segments=[make_segment("coarse-only", 1000, 2000, "grasp")],
            ),
            refined=[make_segment("refined-only", 1000, 2000, "grasp")],
            contract=contract,
            timeline=TimelineConfig(),
        )


@pytest.mark.parametrize("duplicate_side", ["coarse", "refined"])
def test_finalize_rejects_duplicate_segment_ids(
    contract: ResolvedContract,
    duplicate_side: str,
) -> None:
    segment = make_segment("s1", 1000, 2000, "grasp")
    coarse_segments = [segment, segment] if duplicate_side == "coarse" else [segment]
    refined = [segment, segment] if duplicate_side == "refined" else [segment]

    with pytest.raises(ValueError, match=r"segment IDs"):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=CoarseAnnotation(
                task_summary="移动杯子",
                segments=coarse_segments,
            ),
            refined=refined,
            contract=contract,
            timeline=TimelineConfig(),
        )


def test_finalize_calculates_boundary_shifts_before_merging(
    contract: ResolvedContract,
) -> None:
    unchanged_segments = [
        make_segment("s1", 1000, 2000, "grasp"),
        make_segment("s2", 2250, 5000, "grasp"),
    ]

    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(
            task_summary="移动杯子",
            segments=unchanged_segments,
        ),
        refined=unchanged_segments,
        contract=contract,
        timeline=TimelineConfig(merge_gap_ms=250),
    )

    assert document.quality.signals["max_boundary_shift_ms"] == 0
    assert "boundary_shift_gt_2s" not in document.quality.review_reasons


def test_finalize_does_not_merge_same_values_with_different_sentences(
    contract: ResolvedContract,
) -> None:
    segments = [
        make_segment("s1", 1000, 2000, "grasp"),
        make_segment(
            "s2",
            2250,
            3000,
            "grasp",
            sentence="继续抓住杯子。",
        ),
    ]

    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(task_summary="移动杯子", segments=segments),
        refined=segments,
        contract=contract,
        timeline=TimelineConfig(merge_gap_ms=250),
    )

    assert [segment.segment_id for segment in document.segments] == ["s1", "s2"]


def test_finalize_rejects_overlapping_same_values_with_different_sentences(
    contract: ResolvedContract,
) -> None:
    segments = [
        make_segment("s1", 1000, 4000, "grasp"),
        make_segment(
            "s2",
            3000,
            5000,
            "grasp",
            sentence="继续抓住杯子。",
        ),
    ]

    with pytest.raises(TimelineConflict, match=r"overlap: s1, s2"):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=CoarseAnnotation(task_summary="移动杯子", segments=segments),
            refined=segments,
            contract=contract,
            timeline=TimelineConfig(),
        )


def test_finalize_uses_segment_id_to_break_equal_boundary_ties(
    contract: ResolvedContract,
) -> None:
    first = make_segment("s1", 1000, 3000, "grasp")
    second = make_segment("s2", 1000, 3000, "grasp")
    coarse = CoarseAnnotation(task_summary="移动杯子", segments=[first, second])

    forward = finalize_annotation(
        "video-1",
        10_000,
        coarse,
        [first, second],
        contract,
        TimelineConfig(),
    )
    reverse = finalize_annotation(
        "video-1",
        10_000,
        coarse,
        [second, first],
        contract,
        TimelineConfig(),
    )

    assert forward.model_dump() == reverse.model_dump()
    assert forward.segments[0].segment_id == "s1"


def test_finalize_does_not_merge_same_values_beyond_gap_threshold(
    contract: ResolvedContract,
) -> None:
    segments = [
        make_segment("s1", 1000, 2000, "grasp"),
        make_segment("s2", 2251, 3000, "grasp"),
    ]
    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(task_summary="移动杯子", segments=segments),
        refined=segments,
        contract=contract,
        timeline=TimelineConfig(merge_gap_ms=250),
    )

    assert [segment.segment_id for segment in document.segments] == ["s1", "s2"]


def test_finalize_keeps_gap_and_reports_large_boundary_shift(
    contract: ResolvedContract,
) -> None:
    coarse = CoarseAnnotation(
        task_summary="移动杯子",
        segments=[make_segment("s1", 1000, 3000, "grasp")],
    )
    refined = [make_segment("s1", 3500, 5000, "grasp")]

    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=coarse,
        refined=refined,
        contract=contract,
        timeline=TimelineConfig(),
    )

    assert document.segments[0].start_ms == 3500
    assert document.quality.review_reasons == ["boundary_shift_gt_2s"]
    assert document.quality.signals == {
        "max_boundary_shift_ms": 2500,
        "segment_count": 1,
        "empty_timeline": False,
        "merged_segment_sources": {},
    }
    assert document.quality.score == pytest.approx(0.8)


def test_finalize_reports_segment_shorter_than_minimum(
    contract: ResolvedContract,
) -> None:
    segment = make_segment("s1", 1000, 1299, "grasp")
    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(task_summary="移动杯子", segments=[segment]),
        refined=[segment],
        contract=contract,
        timeline=TimelineConfig(min_segment_ms=300),
    )

    assert document.quality.review_reasons == ["segment_shorter_than_minimum"]
    assert document.quality.signals["segment_count"] == 1
    assert document.quality.score == pytest.approx(0.8)


def test_finalize_rejects_different_label_overlap(
    contract: ResolvedContract,
) -> None:
    segments = [
        make_segment("s1", 1000, 4000, "grasp"),
        make_segment("s2", 3000, 5000, "place"),
    ]
    coarse = CoarseAnnotation(task_summary="移动杯子", segments=segments)

    with pytest.raises(
        TimelineConflict,
        match=r"different-label overlap: s1, s2",
    ):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=coarse,
            refined=segments,
            contract=contract,
            timeline=TimelineConfig(),
        )


def test_finalize_validates_values_against_contract(
    contract: ResolvedContract,
) -> None:
    segment = make_segment("s1", 1000, 2000, "pour")
    with pytest.raises(ValueError, match=r"unknown atomic_action: pour"):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=CoarseAnnotation(task_summary="移动杯子", segments=[segment]),
            refined=[segment],
            contract=contract,
            timeline=TimelineConfig(),
        )


def test_finalize_rejects_blank_sentence(contract: ResolvedContract) -> None:
    segment = make_segment("s1", 1000, 2000, "grasp", sentence="   ")
    with pytest.raises(ValueError, match=r"empty sentence: s1"):
        finalize_annotation(
            video_id="video-1",
            duration_ms=10_000,
            coarse=CoarseAnnotation(task_summary="移动杯子", segments=[segment]),
            refined=[segment],
            contract=contract,
            timeline=TimelineConfig(),
        )


def test_finalize_allows_empty_timeline(contract: ResolvedContract) -> None:
    document = finalize_annotation(
        video_id="video-1",
        duration_ms=10_000,
        coarse=CoarseAnnotation(
            task_summary="没有可识别的闭集动作",
            segments=[],
        ),
        refined=[],
        contract=contract,
        timeline=TimelineConfig(),
    )

    assert document.segments == []
    assert document.quality.signals == {
        "max_boundary_shift_ms": 0,
        "segment_count": 0,
        "empty_timeline": True,
        "merged_segment_sources": {},
    }
    assert document.quality.review_reasons == []
    assert document.quality.score == 1.0
