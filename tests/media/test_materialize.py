import shutil
import subprocess
from pathlib import Path

import pytest

from auto_annotation.media.materialize import (
    MAX_CONTACT_SHEET_POINTS,
    MULTI_VIEW_FRAME_SHEET_MATERIALIZER_VERSION,
    FrameSheetView,
    LocalFrameSheetMaterializer,
    LocalMultiViewFrameSheetMaterializer,
    LocalVideoMaterializer,
    MaterializedFrameSheet,
)


def test_materializer_builds_local_time_clip_and_cleans_up(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"materialized")
        return subprocess.CompletedProcess(command, 0, "", "")

    materialized = LocalVideoMaterializer(staging, runner).materialize(
        source,
        start_ms=1250,
        end_ms=4750,
    )

    assert materialized.path.read_bytes() == b"materialized"
    assert materialized.path.parent.parent == staging
    assert materialized.uri.startswith("file://")
    assert calls[0][calls[0].index("-ss") + 1] == "1.250"
    assert calls[0][calls[0].index("-t") + 1] == "3.500"
    assert calls[0][calls[0].index("-vf") + 1] == "setpts=PTS-STARTPTS"

    workspace = materialized.workspace
    materialized.cleanup()
    assert not workspace.exists()


def test_materializer_removes_workspace_when_ffmpeg_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "decode failed")

    with pytest.raises(ValueError, match="decode failed"):
        LocalVideoMaterializer(staging, runner).materialize(source, 0, 1000)

    assert list(staging.iterdir()) == []


def test_materializer_rejects_symlink_staging_root(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    real_staging = tmp_path / "real-staging"
    real_staging.mkdir()
    staging = tmp_path / "staging"
    staging.symlink_to(real_staging, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        LocalVideoMaterializer(staging).materialize(source, 0, 1000)


def test_materializer_rejects_missing_staging_root(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    with pytest.raises(FileNotFoundError):
        LocalVideoMaterializer(tmp_path / "missing").validate()


def test_frame_sheet_materializer_numbers_exact_timestamps(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"image")
        return subprocess.CompletedProcess(command, 0, "", "")

    materialized = LocalFrameSheetMaterializer(staging, runner).materialize(
        source,
        (0, 500, 1000),
        columns=2,
    )

    assert materialized.frame_timestamps_ms == (0, 500, 1000)
    assert materialized.time_unit_width == 320
    assert materialized.time_unit_height == 180
    assert materialized.view_rects == ()
    assert materialized.path.read_bytes() == b"image"
    assert len(calls) == 4
    assert [call[call.index("-ss") + 1] for call in calls[:3]] == [
        "0.000",
        "0.500",
        "1.000",
    ]
    assert "F000 0ms" in calls[0][calls[0].index("-vf") + 1]
    assert "F002 1000ms" in calls[2][calls[2].index("-vf") + 1]
    assert "xstack=inputs=3" in calls[3][
        calls[3].index("-filter_complex") + 1
    ]

    workspace = materialized.workspace
    materialized.cleanup()
    assert not workspace.exists()


def test_frame_sheet_uses_stage_local_labels_without_changing_source_pts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"image")
        return subprocess.CompletedProcess(command, 0, "", "")

    materialized = LocalFrameSheetMaterializer(staging, runner).materialize(
        source,
        (500, 1000),
        label_timestamps_ms=(0, 500),
    )
    try:
        assert materialized.frame_timestamps_ms == (500, 1000)
        assert "F000 0ms" in calls[0][calls[0].index("-vf") + 1]
        assert "F001 500ms" in calls[1][calls[1].index("-vf") + 1]
        assert calls[0][calls[0].index("-ss") + 1] == "0.500"
    finally:
        materialized.cleanup()


def test_frame_sheet_rejects_41_points_before_workspace_or_ffmpeg(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ValueError, match="more than 40"):
        LocalFrameSheetMaterializer(staging, runner).materialize(
            source,
            tuple(range(MAX_CONTACT_SHEET_POINTS + 1)),
        )
    assert calls == []
    assert list(staging.iterdir()) == []


def test_materialized_frame_sheet_layout_metadata_has_safe_defaults(
    tmp_path: Path,
) -> None:
    materialized = MaterializedFrameSheet(
        path=tmp_path / "sheet.jpg",
        workspace=tmp_path,
        frame_timestamps_ms=(0,),
    )

    assert materialized.time_unit_width is None
    assert materialized.time_unit_height is None
    assert materialized.view_rects == ()


def test_frame_sheet_materializer_cleans_up_after_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()
    call_count = 0

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return subprocess.CompletedProcess(command, 1, "", "bad frame")
        Path(command[-1]).write_bytes(b"image")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(ValueError, match="bad frame"):
        LocalFrameSheetMaterializer(staging, runner).materialize(
            source,
            (0, 500),
        )

    assert list(staging.iterdir()) == []


def test_multi_view_frame_sheet_groups_synchronized_views_by_timestamp(
    tmp_path: Path,
) -> None:
    sources = []
    for label in ("HEAD_RGB", "LEFT_WRIST_RGB", "RIGHT_WRIST_RGB"):
        source = tmp_path / f"{label.lower()}.mp4"
        source.write_bytes(b"source")
        sources.append(
            FrameSheetView(
                label=label,
                source=source,
                is_primary=label == "HEAD_RGB",
            )
        )
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"image")
        return subprocess.CompletedProcess(command, 0, "", "")

    materialized = LocalMultiViewFrameSheetMaterializer(
        staging,
        runner,
    ).materialize(
        tuple(sources),
        (0, 500),
        columns=2,
    )

    assert materialized.frame_timestamps_ms == (0, 500)
    assert len(calls) == 3
    first_group = calls[0]
    first_seek_values = [
        first_group[index + 1]
        for index, value in enumerate(first_group)
        if value == "-ss"
    ]
    assert first_seek_values == ["0.000", "0.000", "0.000"]
    first_inputs = [
        first_group[index + 1]
        for index, value in enumerate(first_group)
        if value == "-i"
    ]
    assert first_inputs == [str(view.source.resolve()) for view in sources]
    group_filter = first_group[first_group.index("-filter_complex") + 1]
    assert "F000 0ms HEAD_RGB" in group_filter
    assert "F000 0ms LEFT_WRIST_RGB" in group_filter
    assert "F000 0ms RIGHT_WRIST_RGB" in group_filter
    assert group_filter.count("setpts=PTS-STARTPTS") == 3
    assert "scale=480:270" in group_filter
    assert group_filter.count("scale=240:135") == 2
    assert group_filter.count("fontsize=20") == 1
    assert group_filter.count("fontsize=12") == 2
    assert "xstack=inputs=3:layout=0_0|0_270|240_270" in group_filter
    sheet_filter = calls[2][calls[2].index("-filter_complex") + 1]
    assert "xstack=inputs=2:layout=0_0|480_0" in sheet_filter

    assert materialized.time_unit_width == 480
    assert materialized.time_unit_height == 405
    assert [
        (
            rect.label,
            rect.is_primary,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
        )
        for rect in materialized.view_rects
    ] == [
        ("HEAD_RGB", True, 0, 0, 480, 270),
        ("LEFT_WRIST_RGB", False, 0, 270, 240, 135),
        ("RIGHT_WRIST_RGB", False, 240, 270, 240, 135),
    ]

    workspace = materialized.workspace
    materialized.cleanup()
    assert not workspace.exists()


@pytest.mark.parametrize(
    ("specs", "expected_size", "expected_layout"),
    [
        (
            (("HEAD_RGB", True),),
            (480, 270),
            None,
        ),
        (
            (("RIGHT_WRIST_RGB", False), ("HEAD_RGB", True)),
            (480, 405),
            "xstack=inputs=2:layout=0_0|120_270",
        ),
        (
            (
                ("LEFT_WRIST_RGB", False),
                ("HEAD_RGB", True),
                ("RIGHT_WRIST_RGB", False),
            ),
            (480, 405),
            "xstack=inputs=3:layout=0_0|0_270|240_270",
        ),
    ],
)
def test_multi_view_frame_sheet_lays_out_one_to_three_views(
    tmp_path: Path,
    specs: tuple[tuple[str, bool], ...],
    expected_size: tuple[int, int],
    expected_layout: str | None,
) -> None:
    views = []
    for label, is_primary in specs:
        source = tmp_path / f"{label.lower()}.mp4"
        source.write_bytes(b"source")
        views.append(
            FrameSheetView(
                label=label,
                source=source,
                is_primary=is_primary,
            )
        )
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[-1]).write_bytes(b"image")
        return subprocess.CompletedProcess(command, 0, "", "")

    materialized = LocalMultiViewFrameSheetMaterializer(
        staging,
        runner,
    ).materialize(tuple(views), (250,))

    assert MULTI_VIEW_FRAME_SHEET_MATERIALIZER_VERSION.endswith("-v2")
    assert (materialized.time_unit_width, materialized.time_unit_height) == (
        expected_size
    )
    assert materialized.view_rects[0].is_primary
    assert materialized.view_rects[0].label == "HEAD_RGB"
    group_command = calls[0]
    group_inputs = [
        group_command[index + 1]
        for index, value in enumerate(group_command)
        if value == "-i"
    ]
    assert group_inputs[0].endswith("head_rgb.mp4")
    group_filter = group_command[group_command.index("-filter_complex") + 1]
    assert "scale=480:270" in group_filter
    if expected_layout is None:
        assert "xstack=" not in group_filter
        assert "boxborderw=4[group]" in group_filter
    else:
        assert expected_layout in group_filter

    materialized.cleanup()


def test_multi_view_frame_sheet_rejects_invalid_views(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="between one and three views"):
        LocalMultiViewFrameSheetMaterializer(staging).materialize(
            (),
            (0,),
        )
    with pytest.raises(ValueError, match="between one and three views"):
        LocalMultiViewFrameSheetMaterializer(staging).materialize(
            tuple(
                FrameSheetView(
                    label=f"VIEW_{index}",
                    source=source,
                    is_primary=index == 0,
                )
                for index in range(4)
            ),
            (0,),
        )
    with pytest.raises(ValueError, match="labels must be unique"):
        LocalMultiViewFrameSheetMaterializer(staging).materialize(
            (
                FrameSheetView(
                    label="HEAD_RGB",
                    source=source,
                    is_primary=True,
                ),
                FrameSheetView(label="HEAD_RGB", source=source),
            ),
            (0,),
        )
    with pytest.raises(ValueError, match="exactly one primary"):
        LocalMultiViewFrameSheetMaterializer(staging).materialize(
            (FrameSheetView(label="HEAD_RGB", source=source),),
            (0,),
        )
    with pytest.raises(ValueError, match="exactly one primary"):
        LocalMultiViewFrameSheetMaterializer(staging).materialize(
            (
                FrameSheetView(
                    label="HEAD_RGB",
                    source=source,
                    is_primary=True,
                ),
                FrameSheetView(
                    label="RIGHT_WRIST_RGB",
                    source=source,
                    is_primary=True,
                ),
            ),
            (0,),
        )
    with pytest.raises(ValueError, match="ASCII letters"):
        FrameSheetView(label="LEFT WRIST", source=source)


def test_multi_view_frame_sheet_cleans_up_after_failure(tmp_path: Path) -> None:
    views = []
    for label in ("HEAD_RGB", "LEFT_WRIST_RGB", "RIGHT_WRIST_RGB"):
        source = tmp_path / f"{label.lower()}.mp4"
        source.write_bytes(b"source")
        views.append(
            FrameSheetView(
                label=label,
                source=source,
                is_primary=label == "HEAD_RGB",
            )
        )
    staging = tmp_path / "staging"
    staging.mkdir()

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "bad camera frame")

    with pytest.raises(ValueError, match="bad camera frame"):
        LocalMultiViewFrameSheetMaterializer(staging, runner).materialize(
            tuple(views),
            (0, 500),
        )

    assert list(staging.iterdir()) == []


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")),
    reason="ffmpeg and ffprobe are required",
)
def test_materializer_creates_expected_duration_with_real_ffmpeg(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:d=2:r=30",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    materialized = LocalVideoMaterializer(staging).materialize(
        source,
        start_ms=500,
        end_ms=1500,
    )
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(materialized.path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        assert float(result.stdout.strip()) == pytest.approx(1.0, abs=0.05)
    finally:
        materialized.cleanup()


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")),
    reason="ffmpeg and ffprobe are required",
)
def test_frame_sheet_materializer_creates_expected_grid_with_real_ffmpeg(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:d=2:r=30",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    materialized = LocalFrameSheetMaterializer(staging).materialize(
        source,
        (0, 500, 1000, 1500),
        columns=2,
    )
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(materialized.path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        assert result.stdout.strip() == "640x360"
    finally:
        materialized.cleanup()


@pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")),
    reason="ffmpeg and ffprobe are required",
)
@pytest.mark.parametrize(
    ("view_count", "expected_size"),
    [
        (1, "960x270"),
        (2, "960x405"),
        (3, "960x405"),
    ],
)
def test_multi_view_frame_sheet_creates_expected_grid_with_real_ffmpeg(
    tmp_path: Path,
    view_count: int,
    expected_size: str,
) -> None:
    views = []
    for label, color in (
        ("HEAD_RGB", "red"),
        ("LEFT_WRIST_RGB", "green"),
        ("RIGHT_WRIST_RGB", "blue"),
    )[:view_count]:
        source = tmp_path / f"{label.lower()}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=320x240:d=1:r=30",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            check=True,
        )
        views.append(
            FrameSheetView(
                label=label,
                source=source,
                is_primary=label == "HEAD_RGB",
            )
        )
    staging = tmp_path / "staging"
    staging.mkdir()

    materialized = LocalMultiViewFrameSheetMaterializer(staging).materialize(
        tuple(views),
        (0, 500),
        columns=2,
    )
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(materialized.path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        assert result.stdout.strip() == expected_size
    finally:
        materialized.cleanup()
