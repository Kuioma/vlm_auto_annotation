import json
import subprocess
from fractions import Fraction
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Callable

from auto_annotation.domain.models import VideoInfo

Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _positive_duration_ms(value: object) -> int | None:
    seconds = _finite_float(value)
    if seconds is None or seconds <= 0:
        return None
    try:
        duration_ms = round(seconds * 1000)
    except OverflowError:
        return None
    return duration_ms if duration_ms > 0 else None


def _normalize_frame_timestamps(
    frames: object, stream_start_time: object
) -> tuple[int, ...]:
    if not isinstance(frames, list):
        frames = []
    raw_timestamps = [
        timestamp
        for frame in frames
        if isinstance(frame, dict)
        and (
            timestamp := _finite_float(frame.get("best_effort_timestamp_time"))
        )
        is not None
    ]
    if not raw_timestamps:
        raise ValueError("ffprobe returned no valid frame PTS")

    origin = _finite_float(stream_start_time)
    if origin is None:
        origin = min(raw_timestamps)
    normalized: set[int] = set()
    for timestamp in raw_timestamps:
        try:
            timestamp_ms = round((timestamp - origin) * 1000)
        except OverflowError:
            continue
        if timestamp_ms >= 0:
            normalized.add(timestamp_ms)
    if not normalized:
        raise ValueError("ffprobe returned no valid frame PTS")
    return tuple(sorted(normalized))


def _nominal_fps(value: object, timestamps_ms: tuple[int, ...]) -> float:
    try:
        parsed = float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        parsed = 0.0
    if isfinite(parsed) and parsed > 0:
        return parsed

    intervals_ms = [
        later - earlier
        for earlier, later in zip(timestamps_ms, timestamps_ms[1:])
        if later > earlier
    ]
    if intervals_ms:
        derived = 1000 / median(intervals_ms)
        if isfinite(derived) and derived > 0:
            return derived
    raise ValueError("could not determine nominal fps from metadata or frame PTS")


def probe_video(path: Path, runner: Runner = _default_runner) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,start_time:format=duration:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = runner(command)
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one selected video stream: {path}")
    stream = streams[0]
    duration_ms = _positive_duration_ms(stream.get("duration"))
    if duration_ms is None:
        format_metadata = payload.get("format", {})
        if not isinstance(format_metadata, dict):
            format_metadata = {}
        duration_ms = _positive_duration_ms(format_metadata.get("duration"))
    if duration_ms is None:
        raise ValueError(f"ffprobe returned no positive finite video duration: {path}")
    timestamps = _normalize_frame_timestamps(
        payload.get("frames", []), stream.get("start_time")
    )
    return VideoInfo(
        path=path,
        duration_ms=duration_ms,
        width=int(stream["width"]),
        height=int(stream["height"]),
        nominal_fps=_nominal_fps(stream.get("avg_frame_rate"), timestamps),
        frame_timestamps_ms=timestamps,
    )
