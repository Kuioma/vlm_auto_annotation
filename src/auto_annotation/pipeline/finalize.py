from auto_annotation.config.models import TimelineConfig
from auto_annotation.domain.models import (
    CoarseAnnotation,
    FinalizedAnnotation,
    Quality,
    Segment,
)
from auto_annotation.ontology.registry import ResolvedContract


class TimelineConflict(ValueError):
    pass


def _same_values(left: Segment, right: Segment) -> bool:
    return left.values == right.values


def _validate_segment_ids(coarse: list[Segment], refined: list[Segment]) -> None:
    coarse_ids = [segment.segment_id for segment in coarse]
    refined_ids = [segment.segment_id for segment in refined]

    duplicate_coarse_ids = sorted(
        segment_id
        for segment_id in set(coarse_ids)
        if coarse_ids.count(segment_id) > 1
    )
    if duplicate_coarse_ids:
        raise ValueError(
            f"coarse segment IDs must be unique: {duplicate_coarse_ids}"
        )

    duplicate_refined_ids = sorted(
        segment_id
        for segment_id in set(refined_ids)
        if refined_ids.count(segment_id) > 1
    )
    if duplicate_refined_ids:
        raise ValueError(
            f"refined segment IDs must be unique: {duplicate_refined_ids}"
        )

    coarse_id_set = set(coarse_ids)
    refined_id_set = set(refined_ids)
    if coarse_id_set != refined_id_set:
        raise ValueError(
            "coarse/refined segment IDs must match exactly; "
            f"coarse-only={sorted(coarse_id_set - refined_id_set)}, "
            f"refined-only={sorted(refined_id_set - coarse_id_set)}"
        )


def _merge_same_label(
    segments: list[Segment],
    merge_gap_ms: int,
) -> tuple[list[Segment], dict[str, list[str]]]:
    merged: list[Segment] = []
    source_ids: list[list[str]] = []
    for segment in sorted(
        segments,
        key=lambda item: (item.start_ms, item.end_ms, item.segment_id),
    ):
        if not merged:
            merged.append(segment)
            source_ids.append([segment.segment_id])
            continue

        previous = merged[-1]
        gap = segment.start_ms - previous.end_ms
        if (
            _same_values(previous, segment)
            and previous.sentence == segment.sentence
            and gap <= merge_gap_ms
        ):
            merged[-1] = previous.model_copy(
                update={
                    "end_ms": max(previous.end_ms, segment.end_ms),
                    "evidence_timestamps_ms": sorted(
                        set(
                            previous.evidence_timestamps_ms
                            + segment.evidence_timestamps_ms
                        )
                    ),
                }
            )
            source_ids[-1].append(segment.segment_id)
            continue

        if segment.start_ms < previous.end_ms:
            conflict_kind = (
                "different-sentence"
                if _same_values(previous, segment)
                else "different-label"
            )
            raise TimelineConflict(
                f"{conflict_kind} overlap: {previous.segment_id}, "
                f"{segment.segment_id}"
            )
        merged.append(segment)
        source_ids.append([segment.segment_id])

    merged_segment_sources = {
        segment.segment_id: sources
        for segment, sources in zip(merged, source_ids, strict=True)
        if len(sources) > 1
    }
    return merged, merged_segment_sources


def _resolve_different_label_overlaps(
    segments: list[Segment],
    max_overlap_ms: int,
) -> tuple[list[Segment], list[dict[str, object]]]:
    resolved: list[Segment] = []
    resolutions: list[dict[str, object]] = []
    for segment in sorted(
        segments,
        key=lambda item: (item.start_ms, item.end_ms, item.segment_id),
    ):
        if not resolved:
            resolved.append(segment)
            continue

        previous = resolved[-1]
        overlap_ms = previous.end_ms - segment.start_ms
        if overlap_ms <= 0 or _same_values(previous, segment):
            resolved.append(segment)
            continue
        if max_overlap_ms <= 0 or overlap_ms > max_overlap_ms:
            raise TimelineConflict(
                f"different-label overlap: {previous.segment_id}, "
                f"{segment.segment_id}"
            )

        overlap_start = segment.start_ms
        overlap_end = previous.end_ms
        midpoint = (overlap_start + overlap_end) // 2
        common_evidence = sorted(
            timestamp
            for timestamp in set(previous.evidence_timestamps_ms).intersection(
                segment.evidence_timestamps_ms
            )
            if overlap_start <= timestamp <= overlap_end
        )
        if not common_evidence:
            raise TimelineConflict(
                "different-label overlap has no shared boundary evidence: "
                f"{previous.segment_id}, {segment.segment_id}"
            )
        boundary_ms = min(
            common_evidence,
            key=lambda timestamp: (abs(timestamp - midpoint), timestamp),
        )
        if boundary_ms <= previous.start_ms or boundary_ms >= segment.end_ms:
            raise TimelineConflict(
                "different-label overlap resolution would create an invalid "
                f"segment: {previous.segment_id}, {segment.segment_id}"
            )

        resolved[-1] = previous.model_copy(update={"end_ms": boundary_ms})
        resolved.append(segment.model_copy(update={"start_ms": boundary_ms}))
        resolutions.append(
            {
                "left_segment_id": previous.segment_id,
                "right_segment_id": segment.segment_id,
                "original_left_end_ms": previous.end_ms,
                "original_right_start_ms": segment.start_ms,
                "overlap_ms": overlap_ms,
                "resolved_boundary_ms": boundary_ms,
                "strategy": "shared_evidence_nearest_midpoint",
            }
        )
    return resolved, resolutions


def finalize_annotation(
    video_id: str,
    duration_ms: int,
    coarse: CoarseAnnotation,
    refined: list[Segment],
    contract: ResolvedContract,
    timeline: TimelineConfig,
) -> FinalizedAnnotation:
    _validate_segment_ids(coarse.segments, refined)
    for segment in refined:
        contract.validate_values(segment.values)
        if not segment.sentence.strip():
            raise ValueError(f"empty sentence: {segment.segment_id}")

    coarse_by_id = {segment.segment_id: segment for segment in coarse.segments}
    boundary_shifts = [
        max(
            abs(segment.start_ms - coarse_by_id[segment.segment_id].start_ms),
            abs(segment.end_ms - coarse_by_id[segment.segment_id].end_ms),
        )
        for segment in refined
    ]
    resolved_refined, overlap_resolutions = _resolve_different_label_overlaps(
        refined,
        timeline.max_overlap_resolution_ms,
    )
    segments, merged_segment_sources = _merge_same_label(
        resolved_refined,
        timeline.merge_gap_ms,
    )

    review_reasons: list[str] = []
    if any(shift > 2000 for shift in boundary_shifts):
        review_reasons.append("boundary_shift_gt_2s")
    if any(
        segment.end_ms - segment.start_ms < timeline.min_segment_ms
        for segment in segments
    ):
        review_reasons.append("segment_shorter_than_minimum")
    if overlap_resolutions:
        review_reasons.append("boundary_overlap_resolved")

    signals = {
        "max_boundary_shift_ms": max(boundary_shifts, default=0),
        "segment_count": len(segments),
        "empty_timeline": not segments,
        "merged_segment_sources": merged_segment_sources,
    }
    if overlap_resolutions:
        signals["overlap_resolutions"] = overlap_resolutions
    score = max(0.0, 1.0 - 0.2 * len(review_reasons))
    return FinalizedAnnotation(
        video_id=video_id,
        duration_ms=duration_ms,
        task_summary=coarse.task_summary,
        segments=segments,
        quality=Quality(
            score=score,
            signals=signals,
            review_reasons=review_reasons,
        ),
    )
