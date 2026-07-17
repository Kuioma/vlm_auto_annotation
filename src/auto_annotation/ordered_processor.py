"""Run two-stage ordered closed-set segmentation with multiview frame sheets."""

import argparse
import asyncio
import hashlib
import json
import shutil
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from auto_annotation.config.models import VllmBackendConfig
from auto_annotation.domain.models import ManifestItem, VideoInfo
from auto_annotation.exporters.jsonl import write_jsonl
from auto_annotation.inference.base import GenerationRequest
from auto_annotation.inference.vllm import VLLM_ADAPTER_VERSION, VllmBackend
from auto_annotation.media.materialize import (
    MULTI_VIEW_FRAME_SHEET_MATERIALIZER_VERSION,
    FrameSheetView,
    LocalMultiViewFrameSheetMaterializer,
)
from auto_annotation.media.probe import probe_video
from auto_annotation.media.sampling import build_chunks, build_sample_plan
from auto_annotation.ontology.registry import load_contract
from auto_annotation.ordered_steps import validate_ordered_task_contract
from auto_annotation.pipeline.frame_index import (
    FRAME_INDEX_REFINE_STAGE_VERSION,
    FRAME_INDEX_STAGE_VERSION,
    annotation_from_event_frames,
    annotation_from_events,
    build_event_refine_plan,
    event_from_refinement_frame,
)
from auto_annotation.prompts.frame_index import (
    FRAME_INDEX_PROMPT_VERSION,
    FRAME_INDEX_REFINE_PROMPT_VERSION,
    build_event_refine_frame_request,
    build_fixed_step_frame_request,
)
from auto_annotation.task import load_ordered_task


VIEW_MODALITY_KEYS = {
    "HEAD_RGB": "head_rgb",
    "LEFT_WRIST_RGB": "left_wrist_rgb",
    "RIGHT_WRIST_RGB": "right_wrist_rgb",
}
AUXILIARY_VIEW_WIDTH = 240
AUXILIARY_VIEW_HEIGHT = 135
WA2_VIEW_SELECTION_VERSION = "wa2-active-action-view-selection-v2"
TRAINING_MASK_VERSION = 1
MAX_FRAME_SHEET_POINTS = 40


@dataclass(frozen=True)
class CameraViewSelection:
    views: tuple[FrameSheetView, ...]
    active_action_keys: tuple[str, ...]
    training_mask_path: Path
    training_mask_sha256: str
    modality_path: Path
    modality_sha256: str
    episodes_path: Path
    episodes_sha256: str
    episode_index: int
    video_sha256: tuple[tuple[str, str], ...]


def _validate_frame_sheet_size(
    frame_timestamps_ms: tuple[int, ...],
    *,
    stage: str,
) -> None:
    if len(frame_timestamps_ms) > MAX_FRAME_SHEET_POINTS:
        raise ValueError(
            f"{stage} frame sheet has {len(frame_timestamps_ms)} time units; "
            f"maximum is {MAX_FRAME_SHEET_POINTS}. Reduce FPS or window size."
        )


def _format_delta_ms(delta_ms: int) -> str:
    return f"{delta_ms:+d}ms"


def _refinement_status(*deltas_ms: int) -> str:
    return (
        "UNCHANGED"
        if all(delta == 0 for delta in deltas_ms)
        else "CHANGED"
    )


def _action_mask_is_active(
    key: str,
    value: Any,
    *,
    training_mask_path: Path,
) -> bool:
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, list) and all(
        isinstance(element, bool) for element in value
    ):
        return any(value)
    raise ValueError(
        "training action mask values must be bool, 0, 1, or a boolean "
        f"list: {key!r} in {training_mask_path}"
    )


def _load_active_action_keys(
    training_mask_path: Path,
) -> tuple[tuple[str, ...], str]:
    try:
        raw_mask = training_mask_path.read_bytes()
        payload = json.loads(raw_mask.decode("utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"training mask does not exist: {training_mask_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read training mask JSON: {training_mask_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"training mask must be a JSON object: {training_mask_path}"
        )
    version = payload.get("version")
    if type(version) is not int or version != TRAINING_MASK_VERSION:
        raise ValueError(
            f"training mask version must be {TRAINING_MASK_VERSION}: "
            f"{training_mask_path}"
        )
    action = payload.get("action")
    if not isinstance(action, dict):
        raise ValueError(
            "training mask must contain an action object: "
            f"{training_mask_path}"
        )
    return (
        tuple(
            sorted(
                key
                for key, value in action.items()
                if _action_mask_is_active(
                    key,
                    value,
                    training_mask_path=training_mask_path,
                )
            )
        ),
        hashlib.sha256(raw_mask).hexdigest(),
    )


def _derive_training_mask_path(head_video: Path) -> Path:
    try:
        chunk_directory = head_video.parents[1]
        videos_directory = head_video.parents[2]
        dataset_root = head_video.parents[3]
    except IndexError as error:
        raise ValueError(
            "head camera video must use the canonical dataset path "
            "videos/chunk-*/<video-key>/<episode>.mp4"
        ) from error
    if (
        videos_directory.name != "videos"
        or not chunk_directory.name.startswith("chunk-")
    ):
        raise ValueError(
            "head camera video must use the canonical dataset path "
            "videos/chunk-*/<video-key>/<episode>.mp4"
        )
    return dataset_root / "meta" / "training_mask.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_modality_original_keys(
    modality_path: Path,
    *,
    selected_labels: tuple[str, ...],
) -> tuple[dict[str, str], str]:
    try:
        raw_modality = modality_path.read_bytes()
        payload = json.loads(raw_modality.decode("utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"modality metadata does not exist: {modality_path}"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read modality metadata JSON: {modality_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"modality metadata must be a JSON object: {modality_path}"
        )
    video = payload.get("video")
    if not isinstance(video, dict):
        raise ValueError(
            f"modality metadata must contain a video object: {modality_path}"
        )

    original_keys: dict[str, str] = {}
    for label in selected_labels:
        modality_key = VIEW_MODALITY_KEYS[label]
        mapping = video.get(modality_key)
        if not isinstance(mapping, dict):
            raise ValueError(
                "modality metadata is missing the selected video mapping "
                f"video.{modality_key}: {modality_path}"
            )
        original_key = mapping.get("original_key")
        if not isinstance(original_key, str) or not original_key.strip():
            raise ValueError(
                "selected modality video mapping must contain a non-empty "
                f"original_key: video.{modality_key} in {modality_path}"
            )
        original_keys[label] = original_key
    if len(set(original_keys.values())) != len(original_keys):
        raise ValueError(
            "selected modality video original_key values must be distinct: "
            f"{modality_path}"
        )
    return original_keys, hashlib.sha256(raw_modality).hexdigest()


def _resolve_episode_video_path(
    dataset_root: Path,
    raw_path: Any,
    *,
    label: str,
    episodes_path: Path,
    require_file: bool,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(
            f"{label} episode video path must be a non-empty string: "
            f"{episodes_path}"
        )
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = dataset_root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"cannot resolve {label} episode video path: {candidate}"
        ) from error
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(
            f"{label} episode video path escapes dataset root "
            f"{dataset_root}: {raw_path}"
        )
    if require_file:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"{label} episode video does not exist: {candidate}"
            ) from error
        if not resolved.is_relative_to(dataset_root):
            raise ValueError(
                f"{label} episode video path escapes dataset root "
                f"{dataset_root}: {raw_path}"
            )
        if not resolved.is_file():
            raise ValueError(
                f"{label} episode video is not a file: {resolved}"
            )
    return resolved


def _load_matching_episode(
    episodes_path: Path,
    *,
    dataset_root: Path,
    head_video: Path,
    head_original_key: str,
) -> tuple[dict[str, Any], int, str]:
    try:
        raw_episodes = episodes_path.read_bytes()
        episodes_text = raw_episodes.decode("utf-8")
    except FileNotFoundError as error:
        raise ValueError(
            f"episodes metadata does not exist: {episodes_path}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ValueError(
            f"cannot read episodes metadata: {episodes_path}"
        ) from error

    matches: list[tuple[dict[str, Any], int]] = []
    for line_number, line in enumerate(episodes_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid episodes JSON on line {line_number}: "
                f"{episodes_path}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"episode row {line_number} must be a JSON object: "
                f"{episodes_path}"
            )
        videos = row.get("videos")
        if not isinstance(videos, dict):
            continue
        raw_head_path = videos.get(head_original_key)
        if raw_head_path is None:
            continue
        candidate = _resolve_episode_video_path(
            dataset_root,
            raw_head_path,
            label="HEAD_RGB",
            episodes_path=episodes_path,
            require_file=False,
        )
        if candidate == head_video:
            matches.append((row, line_number))

    if not matches:
        raise ValueError(
            "episodes metadata has no row whose mapped HEAD_RGB video "
            f"matches {head_video}: {episodes_path}"
        )
    if len(matches) != 1:
        matching_lines = ", ".join(str(line_number) for _, line_number in matches)
        raise ValueError(
            "episodes metadata must contain exactly one row for mapped "
            f"HEAD_RGB video {head_video}; matched lines {matching_lines}: "
            f"{episodes_path}"
        )
    row, line_number = matches[0]
    episode_index = row.get("episode_index")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError(
            f"episode row {line_number} must contain a non-negative integer "
            f"episode_index: {episodes_path}"
        )
    return row, episode_index, hashlib.sha256(raw_episodes).hexdigest()


def resolve_camera_views(
    head_video: Path,
    *,
    left_wrist_video: Path | None = None,
    right_wrist_video: Path | None = None,
) -> CameraViewSelection:
    try:
        head_video = head_video.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"head camera video does not exist: {head_video}") from error
    if not head_video.is_file():
        raise ValueError(f"head camera video is not a file: {head_video}")

    training_mask_path = _derive_training_mask_path(head_video)
    dataset_root = training_mask_path.parent.parent
    active_action_keys, training_mask_sha256 = _load_active_action_keys(
        training_mask_path
    )
    active_sides = {
        "LEFT_WRIST_RGB": any(
            key.startswith("left_") for key in active_action_keys
        ),
        "RIGHT_WRIST_RGB": any(
            key.startswith("right_") for key in active_action_keys
        ),
    }
    overrides = {
        "LEFT_WRIST_RGB": left_wrist_video,
        "RIGHT_WRIST_RGB": right_wrist_video,
    }
    for label, override in overrides.items():
        if not active_sides[label] and override is not None:
            side = label.removesuffix("_WRIST_RGB").lower()
            raise ValueError(
                f"{side} wrist override was provided, but no active "
                f"{side}_* action selects that view"
            )

    selected_labels = (
        "HEAD_RGB",
        *(
            label
            for label in ("LEFT_WRIST_RGB", "RIGHT_WRIST_RGB")
            if active_sides[label]
        ),
    )
    modality_path = dataset_root / "meta" / "modality.json"
    original_keys, modality_sha256 = _load_modality_original_keys(
        modality_path,
        selected_labels=selected_labels,
    )
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episode_row, episode_index, episodes_sha256 = _load_matching_episode(
        episodes_path,
        dataset_root=dataset_root,
        head_video=head_video,
        head_original_key=original_keys["HEAD_RGB"],
    )
    episode_videos = episode_row.get("videos")
    if not isinstance(episode_videos, dict):
        raise ValueError(
            f"matching episode row must contain a videos object: {episodes_path}"
        )

    views = [
        FrameSheetView(
            label="HEAD_RGB",
            source=head_video,
            is_primary=True,
        )
    ]
    for label, override in overrides.items():
        if not active_sides[label]:
            continue
        if override is None:
            original_key = original_keys[label]
            if original_key not in episode_videos:
                raise ValueError(
                    "matching episode row is missing the selected video path "
                    f"{original_key!r} for {label}: {episodes_path}"
                )
            candidate = _resolve_episode_video_path(
                dataset_root,
                episode_videos[original_key],
                label=label,
                episodes_path=episodes_path,
                require_file=True,
            )
        else:
            candidate = override.expanduser()
            try:
                candidate = candidate.resolve(strict=True)
            except FileNotFoundError as error:
                raise ValueError(
                    f"{label} override video does not exist: {candidate}"
                ) from error
            if not candidate.is_file():
                raise ValueError(
                    f"{label} override video is not a file: {candidate}"
                )
        views.append(
            FrameSheetView(
                label=label,
                source=candidate,
                is_primary=False,
            )
        )
    sources = tuple(view.source for view in views)
    if len(set(sources)) != len(sources):
        raise ValueError("camera view videos must be distinct")
    return CameraViewSelection(
        views=tuple(views),
        active_action_keys=active_action_keys,
        training_mask_path=training_mask_path,
        training_mask_sha256=training_mask_sha256,
        modality_path=modality_path,
        modality_sha256=modality_sha256,
        episodes_path=episodes_path,
        episodes_sha256=episodes_sha256,
        episode_index=episode_index,
        video_sha256=tuple(
            (view.label, _sha256_file(view.source)) for view in views
        ),
    )


def probe_synchronized_views(
    views: tuple[FrameSheetView, ...],
) -> VideoInfo:
    if not views:
        raise ValueError("camera synchronization requires at least one view")
    probed = tuple((view, probe_video(view.source)) for view in views)
    reference_view, reference_info = probed[0]
    for view, info in probed[1:]:
        if info.duration_ms != reference_info.duration_ms:
            raise ValueError(
                "camera durations are not synchronized: "
                f"{reference_view.label}={reference_info.duration_ms}ms, "
                f"{view.label}={info.duration_ms}ms"
            )
        if info.nominal_fps != reference_info.nominal_fps:
            raise ValueError(
                "camera frame rates are not synchronized: "
                f"{reference_view.label}={reference_info.nominal_fps:g}fps, "
                f"{view.label}={info.nominal_fps:g}fps"
            )
        if info.frame_timestamps_ms != reference_info.frame_timestamps_ms:
            raise ValueError(
                "camera frame PTS are not synchronized: "
                f"{reference_view.label}, {view.label}"
            )
    return reference_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help=(
            "head_rgb video; active left/right actions in the dataset training "
            "mask select wrist videos from modality and episode metadata"
        ),
    )
    parser.add_argument("--left-wrist-video", type=Path)
    parser.add_argument("--right-wrist-video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schema-path", type=Path, required=True)
    parser.add_argument("--ontology-path", type=Path, required=True)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument(
        "--fps",
        "--coarse-fps",
        dest="fps",
        type=float,
        default=2.0,
        help="global multiview frame-index sampling rate",
    )
    parser.add_argument(
        "--refine-fps",
        type=float,
        default=10.0,
        help="local multiview boundary refinement sampling rate",
    )
    parser.add_argument(
        "--refine-window-ms",
        type=int,
        default=2000,
        help="local refinement window centered on each coarse boundary",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--model-revision",
        default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
    parser.add_argument("--vllm-version", default="0.24.0")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "write only the global numbered contact sheet and prompt without "
            "calling vLLM; local sheets require the global model response"
        ),
    )
    parser.add_argument(
        "--media-staging-root",
        type=Path,
        default=Path("/home/cz-sjj/ws/vllm/media"),
    )
    parser.add_argument("--api-key-env")
    return parser.parse_args()


async def _generate(
    backend_config: VllmBackendConfig,
    request: GenerationRequest,
    image_uri: str,
) -> dict[str, object]:
    async with VllmBackend(backend_config) as backend:
        await backend.healthcheck()
        response = await backend.generate_with_image(request, image_uri)
    return response.content


def run() -> int:
    args = parse_args()
    if not isfinite(args.refine_fps) or args.refine_fps <= 0:
        raise ValueError("refine_fps must be finite and positive")
    if args.refine_fps <= args.fps:
        raise ValueError("refine_fps must be greater than the global fps")
    if args.refine_window_ms <= 0:
        raise ValueError("refine_window_ms must be positive")
    view_selection = resolve_camera_views(
        args.video,
        left_wrist_video=args.left_wrist_video,
        right_wrist_video=args.right_wrist_video,
    )
    views = view_selection.views
    head_video = views[0].source
    staging_root = args.media_staging_root.expanduser().resolve(strict=True)

    schema_path = args.schema_path.expanduser().resolve(strict=True)
    ontology_path = args.ontology_path.expanduser().resolve(strict=True)
    task_path = args.task_path.expanduser().resolve(strict=True)
    contract = load_contract(schema_path, ontology_path)
    task = load_ordered_task(task_path)
    steps = task.steps
    validate_ordered_task_contract(task, contract)

    output_dir = args.output_dir.expanduser()
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve(strict=True)

    info = probe_synchronized_views(views)
    chunks = build_chunks(info, max_chunk_ms=120_000, overlap_ms=8000)
    if len(chunks) != 1:
        raise ValueError("frame-index smoke accepts one short video chunk")
    plan = build_sample_plan(info, chunks[0], args.fps)
    _validate_frame_sheet_size(
        tuple(point.source_timestamp_ms for point in plan.points),
        stage="global",
    )
    video_id = f"{head_video.stem}_multiview"
    item = ManifestItem(
        video_id=video_id,
        video_uri=head_video,
        dataset_id="wa2-frame-index-multiview-smoke",
        schema_version=contract.schema.schema_id,
        ontology_id=contract.ontology.ontology_id,
    )
    view_labels = tuple(view.label for view in views)
    request = build_fixed_step_frame_request(
        item,
        plan,
        steps,
        contract,
        view_labels=view_labels,
        primary_view_label="HEAD_RGB",
    )
    materializer = LocalMultiViewFrameSheetMaterializer(staging_root)
    materialized = materializer.materialize(
        views,
        request.sample_timestamps_ms,
        columns=args.columns,
        auxiliary_width=AUXILIARY_VIEW_WIDTH,
        auxiliary_height=AUXILIARY_VIEW_HEIGHT,
    )
    try:
        contact_sheet_path = (
            output_dir / f"{video_id}-contact-sheet.jpg"
        )
        if contact_sheet_path.is_symlink():
            raise ValueError(
                "contact sheet output must not be a symlink: "
                f"{contact_sheet_path}"
            )
        shutil.copyfile(materialized.path, contact_sheet_path)
        prompt_path = output_dir / f"{video_id}-prompt.txt"
        if prompt_path.is_symlink():
            raise ValueError(
                f"prompt output must not be a symlink: {prompt_path}"
            )
        prompt_path.write_text(request.prompt + "\n", encoding="utf-8")
        if args.prepare_only:
            for view in views:
                print(f"{view.label.lower()}={view.source}")
            print(f"frame_count={len(request.sample_timestamps_ms)}")
            print(f"contact_sheet={contact_sheet_path}")
            print(f"prompt={prompt_path}")
            return 0
        backend_config = VllmBackendConfig(
            kind="vllm",
            base_url=args.base_url,
            model_id=args.model_id,
            model_revision=args.model_revision,
            vllm_version=args.vllm_version,
            timeout_s=args.timeout_s,
            max_completion_tokens=512,
            boundary_max_completion_tokens=512,
            temperature=0,
            seed=0,
            enable_thinking=False,
            media_staging_root=staging_root,
            api_key_env=args.api_key_env,
        )
        coarse_content = asyncio.run(
            _generate(backend_config, request, materialized.uri)
        )
    finally:
        materialized.cleanup()

    coarse_annotation = annotation_from_event_frames(
        task_summary=task.task_summary,
        plan=plan,
        steps=steps,
        contract=contract,
        content=coarse_content,
    )
    refined_events = []
    event_refinements = []
    refine_artifacts: list[tuple[str, Path, Path]] = []
    for index, coarse_event in enumerate(coarse_annotation.events):
        refine_plan = build_event_refine_plan(
            info,
            chunks[0],
            coarse_event,
            args.refine_fps,
            args.refine_window_ms,
        )
        _validate_frame_sheet_size(
            tuple(
                point.source_timestamp_ms for point in refine_plan.points
            ),
            stage=f"refine {coarse_event.event_id}",
        )
        previous_event = (
            coarse_annotation.events[index - 1] if index > 0 else None
        )
        next_event = (
            coarse_annotation.events[index + 1]
            if index + 1 < len(coarse_annotation.events)
            else None
        )
        refine_request = build_event_refine_frame_request(
            item,
            refine_plan,
            coarse_event,
            contract,
            steps=steps,
            previous_event=previous_event,
            next_event=next_event,
            view_labels=view_labels,
            primary_view_label="HEAD_RGB",
        )
        refine_materialized = materializer.materialize(
            views,
            refine_request.sample_timestamps_ms,
            columns=args.columns,
            auxiliary_width=AUXILIARY_VIEW_WIDTH,
            auxiliary_height=AUXILIARY_VIEW_HEIGHT,
        )
        try:
            refine_prefix = f"{video_id}-refine-{coarse_event.event_id}"
            refine_contact_sheet_path = (
                output_dir / f"{refine_prefix}-contact-sheet.jpg"
            )
            if refine_contact_sheet_path.is_symlink():
                raise ValueError(
                    "refinement contact sheet output must not be a symlink: "
                    f"{refine_contact_sheet_path}"
                )
            shutil.copyfile(
                refine_materialized.path,
                refine_contact_sheet_path,
            )
            refine_prompt_path = output_dir / f"{refine_prefix}-prompt.txt"
            if refine_prompt_path.is_symlink():
                raise ValueError(
                    "refinement prompt output must not be a symlink: "
                    f"{refine_prompt_path}"
                )
            refine_prompt_path.write_text(
                refine_request.prompt + "\n",
                encoding="utf-8",
            )
            refine_content = asyncio.run(
                _generate(
                    backend_config,
                    refine_request,
                    refine_materialized.uri,
                )
            )
        finally:
            refine_materialized.cleanup()

        refined_event = event_from_refinement_frame(
            coarse_event=coarse_event,
            plan=refine_plan,
            content=refine_content,
        )
        refined_events.append(refined_event)
        refine_artifacts.append(
            (
                coarse_event.event_id,
                refine_contact_sheet_path,
                refine_prompt_path,
            )
        )
        event_refinements.append(
            {
                "event_id": coarse_event.event_id,
                "coarse_timestamp_ms": coarse_event.timestamp_ms,
                "refined_timestamp_ms": refined_event.timestamp_ms,
                "shift_ms": (
                    refined_event.timestamp_ms - coarse_event.timestamp_ms
                ),
                "window_start_ms": refine_plan.chunk.start_ms,
                "window_end_ms": refine_plan.chunk.end_ms,
                "frame_timestamps_ms": (
                    refine_request.sample_timestamps_ms
                ),
                "model_response": refine_content,
            }
        )

    annotation = annotation_from_events(
        task_summary=task.task_summary,
        events=refined_events,
        steps=steps,
        contract=contract,
        timeline=chunks[0],
    )
    output_path = output_dir / "output.jsonl"
    write_jsonl(
        output_path,
        [
            {
                "video_id": video_id,
                "duration_ms": info.duration_ms,
                "task_summary": annotation.task_summary,
                "coarse_events": coarse_annotation.events,
                "coarse_segments": coarse_annotation.segments,
                "events": annotation.events,
                "segments": annotation.segments,
                "event_refinements": event_refinements,
                "frame_timestamps_ms": request.sample_timestamps_ms,
                "model_response": coarse_content,
                "experiment": {
                    "task_id": task.task_id,
                    "task_plan_sha256": _sha256_file(task_path),
                    "task_step_order": [step.step_id for step in steps],
                    "frame_index_prompt_version": FRAME_INDEX_PROMPT_VERSION,
                    "frame_index_stage_version": FRAME_INDEX_STAGE_VERSION,
                    "frame_index_refine_prompt_version": (
                        FRAME_INDEX_REFINE_PROMPT_VERSION
                    ),
                    "frame_index_refine_stage_version": (
                        FRAME_INDEX_REFINE_STAGE_VERSION
                    ),
                    "ontology_id": contract.ontology.ontology_id,
                    "ontology_revision": contract.ontology.revision,
                    "ontology_hash": contract.ontology_hash,
                    "frame_sheet_materializer_version": (
                        MULTI_VIEW_FRAME_SHEET_MATERIALIZER_VERSION
                    ),
                    "vllm_adapter_version": VLLM_ADAPTER_VERSION,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "vllm_version": args.vllm_version,
                    "sampling_fps": args.fps,
                    "refine_sampling_fps": args.refine_fps,
                    "refine_window_ms": args.refine_window_ms,
                    "view_selection": {
                        "version": WA2_VIEW_SELECTION_VERSION,
                        "training_mask_path": str(
                            view_selection.training_mask_path
                        ),
                        "training_mask_sha256": (
                            view_selection.training_mask_sha256
                        ),
                        "modality_path": str(view_selection.modality_path),
                        "modality_sha256": view_selection.modality_sha256,
                        "episodes_path": str(view_selection.episodes_path),
                        "episodes_sha256": view_selection.episodes_sha256,
                        "episode_index": view_selection.episode_index,
                        "active_action_keys": (
                            view_selection.active_action_keys
                        ),
                        "primary_view": "HEAD_RGB",
                        "views": view_labels,
                        "video_sha256": dict(view_selection.video_sha256),
                    },
                    "view_order": view_labels,
                    "view_videos": {
                        view.label: str(view.source) for view in views
                    },
                    "layout": {
                        "columns": min(
                            args.columns,
                            len(request.sample_timestamps_ms),
                        ),
                        "requested_columns": args.columns,
                        "time_unit_width": materialized.time_unit_width,
                        "time_unit_height": materialized.time_unit_height,
                        "view_rects": [
                            {
                                "label": rect.label,
                                "is_primary": rect.is_primary,
                                "x": rect.x,
                                "y": rect.y,
                                "width": rect.width,
                                "height": rect.height,
                            }
                            for rect in materialized.view_rects
                        ],
                    },
                },
            }
        ],
    )

    for view in views:
        print(f"{view.label.lower()}={view.source}")
    print(f"frame_count={len(request.sample_timestamps_ms)}")
    print(f"contact_sheet={contact_sheet_path}")
    print(f"prompt={prompt_path}")
    print(f"output={output_path}")
    print(f"coarse_model_response={coarse_content}")
    for event_id, refine_contact_sheet, refine_prompt in refine_artifacts:
        print(f"refine_contact_sheet[{event_id}]={refine_contact_sheet}")
        print(f"refine_prompt[{event_id}]={refine_prompt}")
    for refinement in event_refinements:
        shift_ms = int(refinement["shift_ms"])
        print(
            "refined_event="
            f"{refinement['event_id']} "
            f"coarse={refinement['coarse_timestamp_ms']}ms "
            f"refined={refinement['refined_timestamp_ms']}ms "
            f"delta={_format_delta_ms(shift_ms)} "
            f"[{_refinement_status(shift_ms)}]"
        )
    for coarse_segment, refined_segment in zip(
        coarse_annotation.segments,
        annotation.segments,
        strict=True,
    ):
        start_delta_ms = refined_segment.start_ms - coarse_segment.start_ms
        end_delta_ms = refined_segment.end_ms - coarse_segment.end_ms
        print(
            f"refined_segment={refined_segment.segment_id} "
            f"coarse=[{coarse_segment.start_ms}, {coarse_segment.end_ms}) "
            f"refined=[{refined_segment.start_ms}, {refined_segment.end_ms}) "
            f"start_delta={_format_delta_ms(start_delta_ms)} "
            f"end_delta={_format_delta_ms(end_delta_ms)} "
            f"[{_refinement_status(start_delta_ms, end_delta_ms)}]"
        )
    for event in annotation.events:
        print(
            f"{event.event_id}: @{event.timestamp_ms} "
            f"{event.sentence}"
        )
    for segment in annotation.segments:
        print(
            f"{segment.segment_id}: "
            f"[{segment.start_ms}, {segment.end_ms}) "
            f"{segment.sentence}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
