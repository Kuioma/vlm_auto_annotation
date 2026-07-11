from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict

from auto_annotation.domain.models import ManifestItem, Segment, VideoChunk, VideoInfo
from auto_annotation.inference.base import InferenceBackend, validate_response
from auto_annotation.media.sampling import build_sample_plan
from auto_annotation.prompts.builders import build_boundary_request


class BoundaryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    boundary_kind: Literal["start", "end"]
    boundary_ms: int
    evidence_timestamps_ms: list[int]


def _window_around(
    chunk: VideoChunk,
    center_ms: int,
    window_ms: int,
    boundary_kind: Literal["start", "end"],
    segment_id: str,
) -> VideoChunk:
    half_window = window_ms // 2
    start_ms = max(chunk.start_ms, center_ms - half_window)
    end_ms = min(chunk.end_ms, start_ms + window_ms)
    start_ms = max(chunk.start_ms, end_ms - window_ms)
    return VideoChunk(
        chunk_id=f"boundary-{boundary_kind}-{segment_id}",
        start_ms=start_ms,
        end_ms=end_ms,
    )


async def run_boundary_stage(
    item: ManifestItem,
    chunk: VideoChunk,
    info: VideoInfo,
    segments: list[Segment],
    backend: InferenceBackend,
    refine_fps: float,
    window_ms: int,
) -> list[Segment]:
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if not isfinite(refine_fps) or refine_fps <= 0:
        raise ValueError("refine_fps must be finite and positive")

    refined: list[Segment] = []
    ordered_segments = sorted(
        segments,
        key=lambda segment: (segment.start_ms, segment.end_ms, segment.segment_id),
    )
    for index, segment in enumerate(ordered_segments):
        previous_segment = ordered_segments[index - 1] if index > 0 else None
        next_segment = (
            ordered_segments[index + 1]
            if index + 1 < len(ordered_segments)
            else None
        )
        boundaries: dict[str, int] = {}
        absolute_evidence: list[int] = []
        for boundary_kind, center_ms in (
            ("start", segment.start_ms),
            ("end", segment.end_ms),
        ):
            window = _window_around(
                chunk,
                center_ms,
                window_ms,
                boundary_kind,
                segment.segment_id,
            )
            if not any(
                window.start_ms <= timestamp < window.end_ms
                for timestamp in info.frame_timestamps_ms
            ):
                raise ValueError(
                    "no PTS in boundary window for "
                    f"segment {segment.segment_id}, kind {boundary_kind}"
                )
            plan = build_sample_plan(info, window, refine_fps)
            request = build_boundary_request(
                item,
                segment,
                boundary_kind,
                plan,
                previous_segment=previous_segment,
                next_segment=next_segment,
            )
            response = await backend.generate(request)
            validate_response(request, response.content)
            content = BoundaryDecision.model_validate(
                response.content, strict=True
            )
            if content.segment_id != segment.segment_id:
                raise ValueError("boundary response segment_id mismatch")
            if content.boundary_kind != boundary_kind:
                raise ValueError("boundary response kind mismatch")
            window_duration_ms = window.end_ms - window.start_ms
            local_boundary = content.boundary_ms
            if not 0 <= local_boundary <= window_duration_ms:
                raise ValueError("boundary response outside local window")
            local_evidence = content.evidence_timestamps_ms
            if any(
                timestamp < 0 or timestamp >= window_duration_ms
                for timestamp in local_evidence
            ):
                raise ValueError("boundary evidence outside local window")
            boundaries[boundary_kind] = window.start_ms + local_boundary
            absolute_evidence.extend(
                window.start_ms + timestamp for timestamp in local_evidence
            )
        refined.append(
            Segment(
                segment_id=segment.segment_id,
                start_ms=boundaries["start"],
                end_ms=boundaries["end"],
                values=segment.values,
                sentence=segment.sentence,
                evidence_timestamps_ms=sorted(set(absolute_evidence)),
            )
        )
    return refined
