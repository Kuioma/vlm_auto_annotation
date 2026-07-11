import json
import math
import subprocess
from pathlib import Path

import pytest

import auto_annotation.media.sampling as sampling
from auto_annotation.domain.models import VideoChunk, VideoInfo
from auto_annotation.media.probe import probe_video
from auto_annotation.media.sampling import build_chunks, build_sample_plan


def _video_info(
    frame_timestamps_ms: tuple[int, ...], duration_ms: int = 2000
) -> VideoInfo:
    return VideoInfo(
        path=Path("clip.mp4"),
        duration_ms=duration_ms,
        width=640,
        height=480,
        nominal_fps=30.0,
        frame_timestamps_ms=frame_timestamps_ms,
    )


def test_probe_and_sampling_use_source_pts(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    payload = {
        "streams": [{"width": 640, "height": 480, "avg_frame_rate": "30/1"}],
        "format": {"duration": "2.0"},
        "frames": [
            {"best_effort_timestamp_time": "0.000"},
            {"best_effort_timestamp_time": "0.490"},
            {"best_effort_timestamp_time": "1.010"},
            {"best_effort_timestamp_time": "1.510"},
            {"best_effort_timestamp_time": "1.990"},
        ],
    }

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    info = probe_video(video, runner=runner)
    chunk = build_chunks(info, max_chunk_ms=120_000, overlap_ms=8000)[0]
    plan = build_sample_plan(info, chunk, fps=2.0)

    assert info.frame_timestamps_ms == (0, 490, 1010, 1510, 1990)
    assert chunk.start_ms == 0
    assert [point.source_timestamp_ms for point in plan.points] == [0, 490, 1010, 1510]


def _probe_from_payload(tmp_path: Path, payload: dict[str, object]) -> VideoInfo:
    video = tmp_path / "probe-input.mp4"
    video.touch()

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    return probe_video(video, runner=runner)


def test_probe_prefers_selected_video_stream_duration(tmp_path: Path) -> None:
    info = _probe_from_payload(
        tmp_path,
        {
            "streams": [
                {
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30/1",
                    "duration": "2.0",
                    "start_time": "0.0",
                }
            ],
            "format": {"duration": "8.0"},
            "frames": [
                {"best_effort_timestamp_time": "0.0"},
                {"best_effort_timestamp_time": "1.0"},
            ],
        },
    )

    assert info.duration_ms == 2000


def test_probe_normalizes_sorts_and_deduplicates_video_pts(tmp_path: Path) -> None:
    info = _probe_from_payload(
        tmp_path,
        {
            "streams": [
                {
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": "30/1",
                    "duration": "2.0",
                    "start_time": "5.0",
                }
            ],
            "format": {"duration": "7.0"},
            "frames": [
                {"best_effort_timestamp_time": "6.0"},
                {"best_effort_timestamp_time": "5.5"},
                {"best_effort_timestamp_time": "5.0"},
                {"best_effort_timestamp_time": "5.5"},
            ],
        },
    )

    assert info.frame_timestamps_ms == (0, 500, 1000)


@pytest.mark.parametrize(
    "stream_duration",
    [None, "N/A", "0", "-1", "NaN", "inf"],
    ids=["missing", "unknown", "zero", "negative", "nan", "infinity"],
)
def test_probe_falls_back_to_valid_format_duration(
    tmp_path: Path, stream_duration: str | None
) -> None:
    stream = {
        "width": 640,
        "height": 480,
        "avg_frame_rate": "30/1",
        "start_time": "0.0",
    }
    if stream_duration is not None:
        stream["duration"] = stream_duration

    info = _probe_from_payload(
        tmp_path,
        {
            "streams": [stream],
            "format": {"duration": "3.25"},
            "frames": [{"best_effort_timestamp_time": "0.0"}],
        },
    )

    assert info.duration_ms == 3250


@pytest.mark.parametrize(
    "format_duration",
    [None, "N/A", "0", "-1", "NaN", "inf"],
    ids=["missing", "unknown", "zero", "negative", "nan", "infinity"],
)
def test_probe_rejects_when_no_positive_finite_duration_exists(
    tmp_path: Path, format_duration: str | None
) -> None:
    format_metadata = {}
    if format_duration is not None:
        format_metadata["duration"] = format_duration

    with pytest.raises(ValueError, match="positive finite video duration"):
        _probe_from_payload(
            tmp_path,
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 480,
                        "avg_frame_rate": "30/1",
                        "duration": "NaN",
                        "start_time": "0.0",
                    }
                ],
                "format": format_metadata,
                "frames": [{"best_effort_timestamp_time": "0.0"}],
            },
        )


def test_probe_rejects_video_without_valid_frame_pts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no valid frame PTS"):
        _probe_from_payload(
            tmp_path,
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 480,
                        "avg_frame_rate": "30/1",
                        "duration": "2.0",
                        "start_time": "0.0",
                    }
                ],
                "format": {"duration": "2.0"},
                "frames": [
                    {},
                    {"best_effort_timestamp_time": "N/A"},
                    {"best_effort_timestamp_time": "NaN"},
                    {"best_effort_timestamp_time": "inf"},
                ],
            },
        )


@pytest.mark.parametrize("avg_frame_rate", ["0/0", "N/A"])
def test_probe_derives_nominal_fps_from_adjacent_pts(
    tmp_path: Path, avg_frame_rate: str
) -> None:
    info = _probe_from_payload(
        tmp_path,
        {
            "streams": [
                {
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": avg_frame_rate,
                    "duration": "1.0",
                    "start_time": "4.0",
                }
            ],
            "format": {"duration": "1.0"},
            "frames": [
                {"best_effort_timestamp_time": "4.08"},
                {"best_effort_timestamp_time": "4.0"},
                {"best_effort_timestamp_time": "4.04"},
            ],
        },
    )

    assert info.nominal_fps == pytest.approx(25.0)
    assert info.frame_timestamps_ms == (0, 40, 80)


def test_probe_rejects_unknown_fps_when_pts_cannot_establish_rate(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="could not determine nominal fps"):
        _probe_from_payload(
            tmp_path,
            {
                "streams": [
                    {
                        "width": 640,
                        "height": 480,
                        "avg_frame_rate": "0/0",
                        "duration": "1.0",
                        "start_time": "4.0",
                    }
                ],
                "format": {"duration": "1.0"},
                "frames": [
                    {"best_effort_timestamp_time": "4.0"},
                    {"best_effort_timestamp_time": "4.0"},
                ],
            },
        )


@pytest.mark.parametrize("max_chunk_ms", [0, -1])
def test_build_chunks_rejects_non_positive_max_chunk_ms(max_chunk_ms: int) -> None:
    info = _video_info((0, 1000))

    with pytest.raises(ValueError, match="max_chunk_ms must be positive"):
        build_chunks(info, max_chunk_ms=max_chunk_ms, overlap_ms=0)


@pytest.mark.parametrize("overlap_ms", [-1, 1000])
def test_build_chunks_rejects_overlap_outside_valid_range(overlap_ms: int) -> None:
    info = _video_info((0, 1000))

    with pytest.raises(
        ValueError,
        match="overlap_ms must satisfy 0 <= overlap_ms < max_chunk_ms",
    ):
        build_chunks(info, max_chunk_ms=1000, overlap_ms=overlap_ms)


@pytest.mark.parametrize(
    "fps",
    [0.0, -1.0, math.inf, -math.inf, math.nan],
    ids=["zero", "negative", "positive-infinity", "negative-infinity", "nan"],
)
def test_build_sample_plan_rejects_invalid_fps_before_sampling(
    monkeypatch: pytest.MonkeyPatch, fps: float
) -> None:
    info = _video_info((0, 1000))
    chunk = VideoChunk(chunk_id="chunk-0000", start_ms=0, end_ms=2000)

    def fail_if_sampling_starts(_: object) -> float:
        raise AssertionError("sampling loop setup reached")

    monkeypatch.setattr(sampling, "float", fail_if_sampling_starts, raising=False)

    with pytest.raises(ValueError, match="fps must be finite and positive"):
        build_sample_plan(info, chunk, fps=fps)


@pytest.mark.parametrize("fps", [240.0001, 1e300], ids=["just-over-limit", "huge"])
def test_build_sample_plan_rejects_fps_above_safety_limit_before_sampling(
    monkeypatch: pytest.MonkeyPatch, fps: float
) -> None:
    info = _video_info((0, 1000))
    chunk = VideoChunk(chunk_id="chunk-0000", start_ms=0, end_ms=2000)

    def fail_if_sampling_starts(_: object) -> float:
        raise AssertionError("sampling loop setup reached")

    monkeypatch.setattr(sampling, "float", fail_if_sampling_starts, raising=False)

    with pytest.raises(ValueError, match="fps must not exceed 240"):
        build_sample_plan(info, chunk, fps=fps)


def test_build_sample_plan_rejects_excessive_target_count_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration_ms = 100_001_000
    info = _video_info((0,), duration_ms=duration_ms)
    chunk = VideoChunk(
        chunk_id="chunk-0000",
        start_ms=0,
        end_ms=duration_ms,
    )

    def fail_if_sampling_starts(_: object) -> float:
        raise AssertionError("sampling loop setup reached")

    monkeypatch.setattr(sampling, "float", fail_if_sampling_starts, raising=False)

    with pytest.raises(ValueError, match="sample target count must not exceed 100000"):
        build_sample_plan(info, chunk, fps=1.0)


def test_build_sample_plan_rejects_chunk_without_pts_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _video_info((900, 2000), duration_ms=3000)
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=1000, end_ms=2000)

    def fail_if_sampling_starts(_: object) -> float:
        raise AssertionError("sampling loop setup reached")

    monkeypatch.setattr(sampling, "float", fail_if_sampling_starts, raising=False)

    with pytest.raises(ValueError, match="no frame PTS in chunk"):
        build_sample_plan(info, chunk, fps=2.0)


def test_sample_plan_snaps_chunk_start_to_first_later_pts() -> None:
    info = _video_info((100, 600, 1100), duration_ms=1200)
    chunk = VideoChunk(chunk_id="chunk-0000", start_ms=0, end_ms=1000)

    plan = build_sample_plan(info, chunk, fps=2.0)

    assert [point.source_timestamp_ms for point in plan.points] == [100, 600]


def test_sample_plan_tie_ignores_pts_before_nonzero_chunk() -> None:
    info = _video_info((900, 1100, 1900), duration_ms=2000)
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=1000, end_ms=2000)

    plan = build_sample_plan(info, chunk, fps=1.0)

    assert [
        (point.source_timestamp_ms, point.chunk_timestamp_ms) for point in plan.points
    ] == [(1100, 100)]
