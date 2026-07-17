import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]
MEDIA_MATERIALIZER_VERSION = "local-video-materializer-v1"
FRAME_SHEET_MATERIALIZER_VERSION = "local-frame-sheet-materializer-v1"
MULTI_VIEW_FRAME_SHEET_MATERIALIZER_VERSION = (
    "local-multi-view-frame-sheet-materializer-v2"
)


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )


def _seconds(milliseconds: int) -> str:
    return f"{milliseconds // 1000}.{milliseconds % 1000:03d}"


@dataclass(frozen=True)
class MaterializedMedia:
    path: Path
    workspace: Path

    @property
    def uri(self) -> str:
        return self.path.as_uri()

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


@dataclass(frozen=True)
class FrameSheetViewRect:
    label: str
    is_primary: bool
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class MaterializedFrameSheet(MaterializedMedia):
    frame_timestamps_ms: tuple[int, ...]
    time_unit_width: int | None = None
    time_unit_height: int | None = None
    view_rects: tuple[FrameSheetViewRect, ...] = ()


@dataclass(frozen=True)
class FrameSheetView:
    label: str
    source: Path
    is_primary: bool = False

    def __post_init__(self) -> None:
        if not self.label or any(
            not character.isascii()
            or not (character.isalnum() or character in {"_", "-"})
            for character in self.label
        ):
            raise ValueError(
                "frame-sheet view label must use ASCII letters, digits, "
                "underscores, or hyphens"
            )


class LocalVideoMaterializer:
    def __init__(self, staging_root: Path, runner: Runner = _default_runner) -> None:
        self.staging_root = staging_root
        self._runner = runner

    def _validated_root(self) -> Path:
        if self.staging_root.is_symlink():
            raise ValueError(
                f"media staging root must not be a symlink: {self.staging_root}"
            )
        root = self.staging_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"media staging root is not a directory: {root}")
        return root

    def validate(self) -> None:
        self._validated_root()

    def materialize(
        self,
        source: Path,
        start_ms: int,
        end_ms: int,
    ) -> MaterializedMedia:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("invalid media interval")
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"media source is not a file: {source}")

        root = self._validated_root()
        workspace = Path(
            tempfile.mkdtemp(prefix="auto-annotation-", dir=root)
        )
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_relative_to(root):
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError("media workspace escapes staging root")

        destination = workspace / "video.mp4"
        duration_ms = end_ms - start_ms
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-ss",
            _seconds(start_ms),
            "-i",
            str(source),
            "-t",
            _seconds(duration_ms),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            "setpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
        result = self._runner(command)
        if result.returncode != 0:
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError(
                f"ffmpeg media materialization failed: {result.stderr.strip()}"
            )
        if not destination.is_file() or destination.stat().st_size == 0:
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError("ffmpeg produced no materialized video")
        return MaterializedMedia(path=destination, workspace=workspace)


class LocalFrameSheetMaterializer(LocalVideoMaterializer):
    def _run_checked(self, command: list[str], workspace: Path) -> None:
        result = self._runner(command)
        if result.returncode != 0:
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError(
                f"ffmpeg frame-sheet materialization failed: "
                f"{result.stderr.strip()}"
            )

    def materialize(
        self,
        source: Path,
        frame_timestamps_ms: tuple[int, ...],
        *,
        columns: int = 5,
        cell_width: int = 320,
        cell_height: int = 180,
    ) -> MaterializedFrameSheet:
        if not frame_timestamps_ms:
            raise ValueError("frame sheet requires at least one timestamp")
        if any(timestamp < 0 for timestamp in frame_timestamps_ms):
            raise ValueError("frame sheet timestamps must be non-negative")
        if tuple(sorted(set(frame_timestamps_ms))) != frame_timestamps_ms:
            raise ValueError(
                "frame sheet timestamps must be unique and increasing"
            )
        if columns <= 0 or cell_width <= 0 or cell_height <= 0:
            raise ValueError("frame sheet dimensions must be positive")

        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"media source is not a file: {source}")
        root = self._validated_root()
        workspace = Path(
            tempfile.mkdtemp(prefix="auto-annotation-frames-", dir=root)
        )
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_relative_to(root):
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError("frame-sheet workspace escapes staging root")

        frame_paths: list[Path] = []
        try:
            for index, timestamp_ms in enumerate(frame_timestamps_ms):
                frame_path = workspace / f"frame-{index:03d}.jpg"
                label = f"F{index:03d} {timestamp_ms}ms"
                video_filter = (
                    f"scale={cell_width}:{cell_height}:"
                    "force_original_aspect_ratio=decrease,"
                    f"pad={cell_width}:{cell_height}:"
                    "(ow-iw)/2:(oh-ih)/2:color=black,"
                    f"drawtext=text='{label}':x=8:y=8:fontsize=22:"
                    "fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=4"
                )
                command = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-ss",
                    _seconds(timestamp_ms),
                    "-i",
                    str(source),
                    "-frames:v",
                    "1",
                    "-vf",
                    video_filter,
                    "-q:v",
                    "2",
                    str(frame_path),
                ]
                self._run_checked(command, workspace)
                if not frame_path.is_file() or frame_path.stat().st_size == 0:
                    raise ValueError(
                        f"ffmpeg produced no frame for F{index:03d}"
                    )
                frame_paths.append(frame_path)

            destination = workspace / "contact-sheet.jpg"
            if len(frame_paths) == 1:
                shutil.copyfile(frame_paths[0], destination)
            else:
                actual_columns = min(columns, len(frame_paths))
                layout = "|".join(
                    f"{(index % actual_columns) * cell_width}_"
                    f"{(index // actual_columns) * cell_height}"
                    for index in range(len(frame_paths))
                )
                command = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
                for frame_path in frame_paths:
                    command.extend(["-i", str(frame_path)])
                command.extend(
                    [
                        "-filter_complex",
                        f"xstack=inputs={len(frame_paths)}:"
                        f"layout={layout}:fill=black",
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(destination),
                    ]
                )
                self._run_checked(command, workspace)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError("ffmpeg produced no contact sheet")
            return MaterializedFrameSheet(
                path=destination,
                workspace=workspace,
                frame_timestamps_ms=frame_timestamps_ms,
                time_unit_width=cell_width,
                time_unit_height=cell_height,
            )
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise


class LocalMultiViewFrameSheetMaterializer(LocalFrameSheetMaterializer):
    """Build a time grid with one large primary and up to two auxiliaries."""

    def materialize(
        self,
        views: tuple[FrameSheetView, ...],
        frame_timestamps_ms: tuple[int, ...],
        *,
        columns: int = 5,
        auxiliary_width: int = 240,
        auxiliary_height: int = 135,
    ) -> MaterializedFrameSheet:
        if not 1 <= len(views) <= 3:
            raise ValueError(
                "multi-view frame sheet requires between one and three views"
            )
        labels = tuple(view.label for view in views)
        if len(set(labels)) != len(labels):
            raise ValueError("multi-view frame-sheet labels must be unique")
        if sum(view.is_primary for view in views) != 1:
            raise ValueError(
                "multi-view frame sheet requires exactly one primary view"
            )
        if not frame_timestamps_ms:
            raise ValueError("frame sheet requires at least one timestamp")
        if any(timestamp < 0 for timestamp in frame_timestamps_ms):
            raise ValueError("frame sheet timestamps must be non-negative")
        if tuple(sorted(set(frame_timestamps_ms))) != frame_timestamps_ms:
            raise ValueError(
                "frame sheet timestamps must be unique and increasing"
            )
        if columns <= 0 or auxiliary_width <= 0 or auxiliary_height <= 0:
            raise ValueError("frame sheet dimensions must be positive")

        resolved_by_input_order = tuple(
            FrameSheetView(
                label=view.label,
                source=view.source.resolve(strict=True),
                is_primary=view.is_primary,
            )
            for view in views
        )
        resolved_views = tuple(
            sorted(
                resolved_by_input_order,
                key=lambda view: not view.is_primary,
            )
        )
        for view in resolved_views:
            if not view.source.is_file():
                raise ValueError(
                    f"multi-view media source is not a file: {view.source}"
                )

        root = self._validated_root()
        workspace = Path(
            tempfile.mkdtemp(
                prefix="auto-annotation-multiview-frames-",
                dir=root,
            )
        )
        resolved_workspace = workspace.resolve(strict=True)
        if not resolved_workspace.is_relative_to(root):
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError("multi-view frame-sheet workspace escapes staging root")

        primary_width = auxiliary_width * 2
        primary_height = auxiliary_height * 2
        group_width = primary_width
        group_height = primary_height
        if len(resolved_views) > 1:
            group_height += auxiliary_height

        auxiliary_count = len(resolved_views) - 1
        view_rects = [
            FrameSheetViewRect(
                label=resolved_views[0].label,
                is_primary=True,
                x=0,
                y=0,
                width=primary_width,
                height=primary_height,
            )
        ]
        for auxiliary_index, view in enumerate(resolved_views[1:]):
            if auxiliary_count == 1:
                x = (primary_width - auxiliary_width) // 2
            else:
                x = auxiliary_index * auxiliary_width
            view_rects.append(
                FrameSheetViewRect(
                    label=view.label,
                    is_primary=False,
                    x=x,
                    y=primary_height,
                    width=auxiliary_width,
                    height=auxiliary_height,
                )
            )

        group_paths: list[Path] = []
        try:
            for frame_index, timestamp_ms in enumerate(frame_timestamps_ms):
                group_path = workspace / f"frame-group-{frame_index:03d}.jpg"
                command = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
                for view in resolved_views:
                    command.extend(
                        [
                            "-ss",
                            _seconds(timestamp_ms),
                            "-i",
                            str(view.source),
                        ]
                    )

                filter_parts: list[str] = []
                view_outputs: list[str] = []
                view_layout: list[str] = []
                for view_index, (view, rect) in enumerate(
                    zip(resolved_views, view_rects, strict=True)
                ):
                    output_name = (
                        "group" if len(resolved_views) == 1 else f"view{view_index}"
                    )
                    label = (
                        f"F{frame_index:03d} {timestamp_ms}ms {view.label}"
                    )
                    font_size = 20 if rect.is_primary else 12
                    filter_parts.append(
                        f"[{view_index}:v]"
                        "setpts=PTS-STARTPTS,"
                        f"scale={rect.width}:{rect.height}:"
                        "force_original_aspect_ratio=decrease:"
                        "force_divisible_by=2,"
                        f"pad={rect.width}:{rect.height}:"
                        "(ow-iw)/2:(oh-ih)/2:color=black,"
                        f"drawtext=text='{label}':x=8:y=8:fontsize={font_size}:"
                        "fontcolor=white:box=1:boxcolor=black@0.65:"
                        f"boxborderw=4[{output_name}]"
                    )
                    view_outputs.append(f"[{output_name}]")
                    view_layout.append(f"{rect.x}_{rect.y}")
                if len(resolved_views) > 1:
                    filter_parts.append(
                        f"{''.join(view_outputs)}"
                        f"xstack=inputs={len(resolved_views)}:"
                        f"layout={'|'.join(view_layout)}:fill=black[group]"
                    )
                command.extend(
                    [
                        "-filter_complex",
                        ";".join(filter_parts),
                        "-map",
                        "[group]",
                        "-frames:v",
                        "1",
                        "-pix_fmt",
                        "yuvj444p",
                        "-q:v",
                        "2",
                        str(group_path),
                    ]
                )
                self._run_checked(command, workspace)
                if not group_path.is_file() or group_path.stat().st_size == 0:
                    raise ValueError(
                        f"ffmpeg produced no multi-view group for "
                        f"F{frame_index:03d}"
                    )
                group_paths.append(group_path)

            destination = workspace / "multi-view-contact-sheet.jpg"
            if len(group_paths) == 1:
                shutil.copyfile(group_paths[0], destination)
            else:
                actual_columns = min(columns, len(group_paths))
                layout = "|".join(
                    f"{(index % actual_columns) * group_width}_"
                    f"{(index // actual_columns) * group_height}"
                    for index in range(len(group_paths))
                )
                command = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
                for group_path in group_paths:
                    command.extend(["-i", str(group_path)])
                command.extend(
                    [
                        "-filter_complex",
                        f"xstack=inputs={len(group_paths)}:"
                        f"layout={layout}:fill=black",
                        "-frames:v",
                        "1",
                        "-pix_fmt",
                        "yuvj444p",
                        "-q:v",
                        "2",
                        str(destination),
                    ]
                )
                self._run_checked(command, workspace)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError("ffmpeg produced no multi-view contact sheet")
            return MaterializedFrameSheet(
                path=destination,
                workspace=workspace,
                frame_timestamps_ms=frame_timestamps_ms,
                time_unit_width=group_width,
                time_unit_height=group_height,
                view_rects=tuple(view_rects),
            )
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
