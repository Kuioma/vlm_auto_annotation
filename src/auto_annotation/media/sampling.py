from bisect import bisect_left
from math import isfinite

from auto_annotation.domain.models import SamplePlan, SamplePoint, VideoChunk, VideoInfo

MAX_SAMPLE_FPS = 240.0
MAX_SAMPLE_TARGETS = 100_000


def build_chunks(
    info: VideoInfo, max_chunk_ms: int, overlap_ms: int
) -> tuple[VideoChunk, ...]:
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be positive")
    if not 0 <= overlap_ms < max_chunk_ms:
        raise ValueError("overlap_ms must satisfy 0 <= overlap_ms < max_chunk_ms")
    chunks: list[VideoChunk] = []
    start = 0
    index = 0
    while start < info.duration_ms:
        end = min(start + max_chunk_ms, info.duration_ms)
        chunks.append(VideoChunk(chunk_id=f"chunk-{index:04d}", start_ms=start, end_ms=end))
        if end == info.duration_ms:
            break
        start = end - overlap_ms
        index += 1
    return tuple(chunks)


def _nearest_timestamp(timestamps: tuple[int, ...], target: int) -> int:
    index = bisect_left(timestamps, target)
    candidates = timestamps[max(0, index - 1) : min(len(timestamps), index + 1)]
    return min(candidates, key=lambda value: abs(value - target))


def build_sample_plan(info: VideoInfo, chunk: VideoChunk, fps: float) -> SamplePlan:
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if fps > MAX_SAMPLE_FPS:
        raise ValueError(f"fps must not exceed {MAX_SAMPLE_FPS:g}")
    step_ms = 1000 / fps
    duration_ms = chunk.end_ms - chunk.start_ms
    if duration_ms > MAX_SAMPLE_TARGETS * step_ms:
        raise ValueError(
            f"sample target count must not exceed {MAX_SAMPLE_TARGETS}"
        )
    chunk_timestamps = tuple(
        timestamp
        for timestamp in info.frame_timestamps_ms
        if chunk.start_ms <= timestamp < chunk.end_ms
    )
    if not chunk_timestamps:
        raise ValueError(f"no frame PTS in chunk: {chunk.chunk_id}")
    targets: list[int] = []
    current = float(chunk.start_ms)
    while current < chunk.end_ms:
        targets.append(round(current))
        current += step_ms
    selected = sorted(
        {_nearest_timestamp(chunk_timestamps, target) for target in targets}
    )
    points = tuple(
        SamplePoint(
            source_timestamp_ms=timestamp,
            chunk_timestamp_ms=timestamp - chunk.start_ms,
        )
        for timestamp in selected
    )
    return SamplePlan(chunk=chunk, target_fps=fps, points=points)
