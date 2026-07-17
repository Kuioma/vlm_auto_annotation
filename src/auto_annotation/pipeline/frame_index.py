from math import isfinite

from pydantic import BaseModel, ConfigDict

from auto_annotation.domain.models import (
    Event,
    SamplePlan,
    SamplePoint,
    Segment,
    TaskStep,
    TemporalAnnotation,
    VideoChunk,
    VideoInfo,
)
from auto_annotation.media.sampling import build_sample_plan
from auto_annotation.ontology.registry import ResolvedContract
from auto_annotation.ordered_steps import resolve_ordered_step_entries


FRAME_INDEX_STAGE_VERSION = "ordered-step-frame-boundary-stage-v1"
FRAME_INDEX_REFINE_STAGE_VERSION = "ordered-step-frame-boundary-refine-stage-v1"


class FrameEventDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_frame_indices: dict[str, int]
    evidence_frame_indices: list[int]


class FrameEventRefinementDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_frame_index: int
    evidence_frame_indices: list[int]


def build_event_refine_plan(
    info: VideoInfo,
    chunk: VideoChunk,
    coarse_event: Event,
    fps: float,
    window_ms: int,
) -> SamplePlan:
    if window_ms <= 0:
        raise ValueError("event refine window_ms must be positive")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("event refine fps must be finite and positive")
    if not chunk.start_ms <= coarse_event.timestamp_ms < chunk.end_ms:
        raise ValueError("coarse event timestamp is outside the source chunk")

    half_window = window_ms // 2
    start_ms = max(chunk.start_ms, coarse_event.timestamp_ms - half_window)
    end_ms = min(chunk.end_ms, start_ms + window_ms)
    start_ms = max(chunk.start_ms, end_ms - window_ms)
    window = VideoChunk(
        chunk_id=f"frame-index-refine-{coarse_event.event_id}",
        start_ms=start_ms,
        end_ms=end_ms,
    )
    if coarse_event.timestamp_ms not in info.frame_timestamps_ms:
        raise ValueError("coarse event timestamp is not a source frame PTS")
    plan = build_sample_plan(info, window, fps)
    if any(
        point.source_timestamp_ms == coarse_event.timestamp_ms
        for point in plan.points
    ):
        return plan
    points = tuple(
        sorted(
            (
                *plan.points,
                SamplePoint(
                    source_timestamp_ms=coarse_event.timestamp_ms,
                    chunk_timestamp_ms=(
                        coarse_event.timestamp_ms - window.start_ms
                    ),
                ),
            ),
            key=lambda point: point.source_timestamp_ms,
        )
    )
    return SamplePlan(chunk=window, target_fps=fps, points=points)


def event_from_refinement_frame(
    *,
    coarse_event: Event,
    plan: SamplePlan,
    content: dict[str, object],
) -> Event:
    decision = FrameEventRefinementDecision.model_validate(content, strict=True)
    if decision.event_id != coarse_event.event_id:
        raise ValueError("event refinement response event_id mismatch")

    frame_count = len(plan.points)
    if not 0 <= decision.event_frame_index < frame_count:
        raise ValueError("event frame index is outside the refinement frame sheet")
    if len(set(decision.evidence_frame_indices)) != len(
        decision.evidence_frame_indices
    ):
        raise ValueError("evidence frame indices must be unique")
    if any(
        frame_index < 0 or frame_index >= frame_count
        for frame_index in decision.evidence_frame_indices
    ):
        raise ValueError("evidence frame index is outside the refinement frame sheet")

    return coarse_event.model_copy(
        update={
            "timestamp_ms": plan.points[
                decision.event_frame_index
            ].source_timestamp_ms,
            "evidence_timestamps_ms": sorted(
                plan.points[frame_index].source_timestamp_ms
                for frame_index in decision.evidence_frame_indices
            ),
        }
    )


def _annotation_from_events(
    *,
    task_summary: str,
    events: list[Event],
    steps: tuple[TaskStep, ...],
    contract: ResolvedContract,
    timeline: VideoChunk,
    interval_evidence_timestamps_ms: list[int],
) -> TemporalAnnotation:
    resolve_ordered_step_entries(steps, contract)
    # Each event is the boundary that starts the corresponding non-first step.
    # This yields exactly len(steps) - 1 boundaries for any ordered action set.
    boundary_steps = steps[1:]
    expected_event_ids = [step.step_id for step in boundary_steps]
    actual_event_ids = [event.event_id for event in events]
    if actual_event_ids != expected_event_ids:
        raise ValueError(
            "events must exactly match ordered boundary IDs; "
            f"expected={expected_event_ids}, actual={actual_event_ids}"
        )

    event_timestamps = [event.timestamp_ms for event in events]
    if event_timestamps != sorted(set(event_timestamps)):
        raise ValueError("event timestamps must be unique and increasing")
    if any(
        not timeline.start_ms < timestamp < timeline.end_ms
        for timestamp in event_timestamps
    ):
        raise ValueError(
            "event timestamps must be strictly inside the annotation timeline"
        )

    for event, step in zip(events, boundary_steps, strict=True):
        contract.validate_values(event.values)
        if event.values != step.values or event.sentence != step.sentence:
            raise ValueError(
                "event semantics must match the step started by the boundary: "
                f"{event.event_id}"
            )

    boundaries = (
        timeline.start_ms,
        *(event.timestamp_ms for event in events),
        timeline.end_ms,
    )
    segments = []
    for step, start_ms, end_ms in zip(
        steps,
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        segments.append(
            Segment(
                segment_id=step.step_id,
                start_ms=start_ms,
                end_ms=end_ms,
                values=step.values,
                sentence=step.sentence,
                evidence_timestamps_ms=[
                    timestamp
                    for timestamp in interval_evidence_timestamps_ms
                    if start_ms <= timestamp < end_ms
                ],
            )
        )
    return TemporalAnnotation(
        task_summary=task_summary,
        events=events,
        segments=segments,
    )


def annotation_from_events(
    *,
    task_summary: str,
    events: list[Event],
    steps: tuple[TaskStep, ...],
    contract: ResolvedContract,
    timeline: VideoChunk,
) -> TemporalAnnotation:
    evidence_timestamps = sorted(
        {
            timestamp
            for event in events
            for timestamp in event.evidence_timestamps_ms
        }
    )
    return _annotation_from_events(
        task_summary=task_summary,
        events=events,
        steps=steps,
        contract=contract,
        timeline=timeline,
        interval_evidence_timestamps_ms=evidence_timestamps,
    )


def annotation_from_event_frames(
    *,
    task_summary: str,
    plan: SamplePlan,
    steps: tuple[TaskStep, ...],
    contract: ResolvedContract,
    content: dict[str, object],
) -> TemporalAnnotation:
    resolve_ordered_step_entries(steps, contract)
    decision = FrameEventDecision.model_validate(content, strict=True)
    boundary_steps = steps[1:]
    expected_event_ids = tuple(step.step_id for step in boundary_steps)
    actual_event_ids = set(decision.event_frame_indices)
    if actual_event_ids != set(expected_event_ids):
        missing = sorted(set(expected_event_ids) - actual_event_ids)
        unexpected = sorted(actual_event_ids - set(expected_event_ids))
        raise ValueError(
            "event frame keys must exactly match point-event IDs; "
            f"missing={missing}, unexpected={unexpected}"
        )

    frame_count = len(plan.points)
    ordered_event_indices = [
        decision.event_frame_indices[event_id]
        for event_id in expected_event_ids
    ]
    if any(
        frame_index < 0 or frame_index >= frame_count
        for frame_index in ordered_event_indices
    ):
        raise ValueError("event frame index is outside the frame sheet")
    if ordered_event_indices != sorted(set(ordered_event_indices)):
        raise ValueError("event frame indices must be unique and increasing")
    if len(set(decision.evidence_frame_indices)) != len(
        decision.evidence_frame_indices
    ):
        raise ValueError("evidence frame indices must be unique")
    if any(
        frame_index < 0 or frame_index >= frame_count
        for frame_index in decision.evidence_frame_indices
    ):
        raise ValueError("evidence frame index is outside the frame sheet")

    event_timestamps = {
        event_id: plan.points[
            decision.event_frame_indices[event_id]
        ].source_timestamp_ms
        for event_id in expected_event_ids
    }
    evidence_timestamps = sorted(
        plan.points[frame_index].source_timestamp_ms
        for frame_index in decision.evidence_frame_indices
    )

    events = [
        Event(
            event_id=step.step_id,
            timestamp_ms=event_timestamps[step.step_id],
            values=step.values,
            sentence=step.sentence,
            evidence_timestamps_ms=[
                timestamp
                for timestamp in evidence_timestamps
                if timestamp == event_timestamps[step.step_id]
            ],
        )
        for step in boundary_steps
    ]
    return _annotation_from_events(
        task_summary=task_summary,
        events=events,
        steps=steps,
        contract=contract,
        timeline=plan.chunk,
        interval_evidence_timestamps_ms=evidence_timestamps,
    )
