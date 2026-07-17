import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
CONFIG_ROOT = REPOSITORY_ROOT / "examples/pick_and_place_bolt"

from auto_annotation.domain.models import VideoInfo
from auto_annotation.media.materialize import (
    FrameSheetView,
    FrameSheetViewRect,
    MaterializedFrameSheet,
)
from auto_annotation import ordered_processor as processor


DEFAULT_VIDEO_ORIGINAL_KEYS = {
    "head_rgb": "observation.images.head_rgb",
    "left_wrist_rgb": "observation.images.left_wrist_rgb",
    "right_wrist_rgb": "observation.images.right_wrist_rgb",
}


def _write_camera_video(
    root: Path,
    camera_directory: str,
    name: str = "episode_000005.mp4",
) -> Path:
    video = root / "videos/chunk-000" / camera_directory / name
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    return video


def _write_video_meta(
    root: Path,
    *,
    original_keys: dict[str, str] | None = None,
    episode_videos: dict[str, str] | None = None,
    episode_index: int = 5,
) -> tuple[Path, Path]:
    original_keys = (
        DEFAULT_VIDEO_ORIGINAL_KEYS.copy()
        if original_keys is None
        else original_keys
    )
    meta_root = root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)
    modality = meta_root / "modality.json"
    modality.write_text(
        json.dumps(
            {
                "video": {
                    modality_key: {"original_key": original_key}
                    for modality_key, original_key in original_keys.items()
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    if episode_videos is None:
        episode_videos = {}
        for original_key in original_keys.values():
            candidates = sorted(
                (root / "videos").glob(
                    f"chunk-*/{original_key}/episode_000005.mp4"
                )
            )
            if candidates:
                episode_videos[original_key] = str(
                    candidates[0].relative_to(root)
                )
    episodes = meta_root / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_index": episode_index,
                "videos": episode_videos,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return modality, episodes


def _write_training_mask(
    root: Path,
    action: dict[str, object],
) -> Path:
    mask = root / "meta/training_mask.json"
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_text(
        json.dumps(
            {
                "version": 1,
                "video": {},
                "state": {},
                "action": action,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_video_meta(root)
    return mask


def test_resolve_camera_views_selects_right_only_and_ignores_missing_left(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(
        tmp_path,
        "observation.images.head_rgb",
    )
    right = _write_camera_video(
        tmp_path,
        "observation.images.right_wrist_rgb",
    )
    mask = _write_training_mask(
        tmp_path,
        {
            "left_action": False,
            "neck_joint_position": True,
            "right_action": [False, True],
        },
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.label for view in selection.views) == (
        "HEAD_RGB",
        "RIGHT_WRIST_RGB",
    )
    assert tuple(view.source for view in selection.views) == tuple(
        path.resolve() for path in (head, right)
    )
    assert tuple(view.is_primary for view in selection.views) == (True, False)
    assert selection.active_action_keys == (
        "neck_joint_position",
        "right_action",
    )
    assert selection.training_mask_path == mask
    assert selection.training_mask_sha256 == hashlib.sha256(
        mask.read_bytes()
    ).hexdigest()
    modality = tmp_path / "meta/modality.json"
    episodes = tmp_path / "meta/episodes.jsonl"
    assert selection.modality_path == modality
    assert selection.modality_sha256 == hashlib.sha256(
        modality.read_bytes()
    ).hexdigest()
    assert selection.episodes_path == episodes
    assert selection.episodes_sha256 == hashlib.sha256(
        episodes.read_bytes()
    ).hexdigest()
    assert selection.episode_index == 5


def test_resolve_camera_views_uses_non_sibling_episode_video_path(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    right = tmp_path / "recordings/camera-c/right-special.mp4"
    right.parent.mkdir(parents=True)
    right.write_bytes(b"video")
    _write_training_mask(tmp_path, {"right_action": True})
    _write_video_meta(
        tmp_path,
        episode_videos={
            "observation.images.head_rgb": str(head.relative_to(tmp_path)),
            "observation.images.right_wrist_rgb": str(
                right.relative_to(tmp_path)
            ),
        },
        episode_index=17,
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.source for view in selection.views) == (
        head.resolve(),
        right.resolve(),
    )
    assert selection.episode_index == 17


def test_resolve_camera_views_uses_modality_mapping_for_head_path(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "sensors.main_rgb")
    _write_training_mask(tmp_path, {"waist_joint_position": True})
    _write_video_meta(
        tmp_path,
        original_keys={"head_rgb": "sensors.main_rgb"},
        episode_videos={
            "sensors.main_rgb": str(head.relative_to(tmp_path)),
        },
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.source for view in selection.views) == (head.resolve(),)


def test_resolve_camera_views_selects_left_only_with_integer_masks(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(
        tmp_path,
        "observation.images.head_rgb",
    )
    left = _write_camera_video(
        tmp_path,
        "observation.images.left_wrist_rgb",
    )
    _write_training_mask(
        tmp_path,
        {"left_action": 1, "right_action": 0},
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.label for view in selection.views) == (
        "HEAD_RGB",
        "LEFT_WRIST_RGB",
    )
    assert selection.active_action_keys == ("left_action",)


def test_resolve_camera_views_selects_both_wrist_views(tmp_path: Path) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    left = _write_camera_video(
        tmp_path,
        "observation.images.left_wrist_rgb",
    )
    right = _write_camera_video(
        tmp_path,
        "observation.images.right_wrist_rgb",
    )
    _write_training_mask(
        tmp_path,
        {"left_action": [True, False], "right_action": True},
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.source for view in selection.views) == tuple(
        path.resolve() for path in (head, left, right)
    )


def test_resolve_camera_views_keeps_only_head_for_non_side_actions(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    _write_training_mask(
        tmp_path,
        {
            "left_action": [],
            "neck_joint_position": True,
            "waist_joint_position": 1,
        },
    )
    _write_video_meta(
        tmp_path,
        original_keys={"head_rgb": "observation.images.head_rgb"},
    )

    selection = processor.resolve_camera_views(head)

    assert tuple(view.label for view in selection.views) == ("HEAD_RGB",)
    assert selection.active_action_keys == (
        "neck_joint_position",
        "waist_joint_position",
    )


def test_resolve_camera_views_accepts_explicit_selected_wrist_paths(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_bytes(b"video")
    right.write_bytes(b"video")
    _write_training_mask(
        tmp_path,
        {"left_action": True, "right_action": True},
    )

    selection = processor.resolve_camera_views(
        head,
        left_wrist_video=left,
        right_wrist_video=right,
    )

    assert tuple(view.source for view in selection.views) == tuple(
        path.resolve() for path in (head, left, right)
    )


def test_resolve_camera_views_override_still_requires_modality_mapping(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    left = tmp_path / "left.mp4"
    left.write_bytes(b"video")
    _write_training_mask(tmp_path, {"left_action": True})
    _write_video_meta(
        tmp_path,
        original_keys={"head_rgb": "observation.images.head_rgb"},
    )

    with pytest.raises(ValueError, match=r"video\.left_wrist_rgb"):
        processor.resolve_camera_views(head, left_wrist_video=left)


def test_resolve_camera_views_rejects_duplicate_sources(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    _write_training_mask(tmp_path, {"left_action": True})

    with pytest.raises(ValueError, match="must be distinct"):
        processor.resolve_camera_views(
            head,
            left_wrist_video=head,
        )


def test_resolve_camera_views_rejects_missing_selected_episode_path(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(
        tmp_path,
        "observation.images.head_rgb",
    )
    _write_training_mask(tmp_path, {"left_action": True})

    with pytest.raises(ValueError, match="missing the selected video path"):
        processor.resolve_camera_views(head)


def test_resolve_camera_views_rejects_missing_selected_episode_file(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    _write_training_mask(tmp_path, {"left_action": True})
    _write_video_meta(
        tmp_path,
        episode_videos={
            "observation.images.head_rgb": str(head.relative_to(tmp_path)),
            "observation.images.left_wrist_rgb": (
                "recordings/missing-left.mp4"
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="LEFT_WRIST_RGB episode video does not exist",
    ):
        processor.resolve_camera_views(head)


def test_resolve_camera_views_rejects_episode_path_escape(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    _write_training_mask(tmp_path, {"right_action": True})
    _write_video_meta(
        tmp_path,
        episode_videos={
            "observation.images.head_rgb": str(head.relative_to(tmp_path)),
            "observation.images.right_wrist_rgb": "../outside.mp4",
        },
    )

    with pytest.raises(ValueError, match="path escapes dataset root"):
        processor.resolve_camera_views(head)


def test_resolve_camera_views_rejects_inactive_side_override(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    left = tmp_path / "left.mp4"
    left.write_bytes(b"video")
    _write_training_mask(tmp_path, {"right_action": False})

    with pytest.raises(ValueError, match="left wrist override was provided"):
        processor.resolve_camera_views(head, left_wrist_video=left)


@pytest.mark.parametrize(
    "invalid_value",
    ["true", 2, 1.0, [True, 1], None, {}],
)
def test_resolve_camera_views_rejects_invalid_action_mask_values(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    _write_training_mask(tmp_path, {"left_action": invalid_value})

    with pytest.raises(ValueError, match="must be bool, 0, 1"):
        processor.resolve_camera_views(head)


def test_resolve_camera_views_requires_derived_training_mask(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")

    with pytest.raises(ValueError, match="training mask does not exist"):
        processor.resolve_camera_views(head)


def test_resolve_camera_views_rejects_unknown_training_mask_version(
    tmp_path: Path,
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    mask = _write_training_mask(tmp_path, {"right_action": True})
    payload = json.loads(mask.read_text(encoding="utf-8"))
    payload["version"] = 2
    mask.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="version must be 1"):
        processor.resolve_camera_views(head)


def test_task_file_supplies_ordered_closed_set_steps() -> None:
    contract = processor.load_contract(
        CONFIG_ROOT / "schema.yaml",
        CONFIG_ROOT / "ontology.yaml",
    )

    task = processor.load_ordered_task(CONFIG_ROOT / "task.yaml")
    steps = task.steps

    assert task.task_summary == "将螺丝从蓝色料箱转移至目标容器"
    assert tuple(step.step_id for step in steps) == (
        "pick_up",
        "transfer",
        "place",
    )
    for step in steps:
        entry = contract.entry_for("ontology.atomic_actions", step.step_id)
        assert entry.temporal_type in {"point_event", "interval_action"}
        assert entry.boundary_rule
        assert entry.positive_cues
        assert entry.negative_cues
    assert "抓住并抬起" not in steps[0].sentence


def test_frame_sheet_size_is_bounded_before_materialization() -> None:
    with pytest.raises(ValueError, match="maximum is 40"):
        processor._validate_frame_sheet_size(
            tuple(range(processor.MAX_FRAME_SHEET_POINTS + 1)),
            stage="refine pick_up",
        )


def test_refinement_status_makes_zero_shift_explicit() -> None:
    assert processor._format_delta_ms(0) == "+0ms"
    assert processor._refinement_status(0) == "UNCHANGED"
    assert processor._refinement_status(0, 1) == "CHANGED"


def test_run_records_active_action_selection_and_asymmetric_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    head = _write_camera_video(tmp_path, "observation.images.head_rgb")
    right = _write_camera_video(
        tmp_path,
        "observation.images.right_wrist_rgb",
    )
    mask = _write_training_mask(
        tmp_path,
        {
            "left_action_chunk_tcp_in_head_cam": False,
            "right_action_chunk_tcp_in_head_cam": True,
        },
    )
    modality = tmp_path / "meta/modality.json"
    episodes = tmp_path / "meta/episodes.jsonl"
    staging = tmp_path / "staging"
    staging.mkdir()
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        processor,
        "parse_args",
        lambda: SimpleNamespace(
            video=head,
            left_wrist_video=None,
            right_wrist_video=None,
            output_dir=output_dir,
            schema_path=CONFIG_ROOT / "schema.yaml",
            ontology_path=CONFIG_ROOT / "ontology.yaml",
            task_path=CONFIG_ROOT / "task.yaml",
            fps=2.0,
            refine_fps=4.0,
            refine_window_ms=1000,
            columns=3,
            base_url="http://127.0.0.1:8000/v1",
            model_id="test-model",
            model_revision="test-revision",
            vllm_version="test-vllm",
            timeout_s=30,
            prepare_only=False,
            media_staging_root=staging,
            api_key_env=None,
        ),
    )
    info = VideoInfo(
        path=head.resolve(),
        duration_ms=3000,
        width=1280,
        height=720,
        nominal_fps=4.0,
        frame_timestamps_ms=tuple(range(0, 3000, 250)),
    )
    monkeypatch.setattr(processor, "probe_synchronized_views", lambda views: info)

    materialized_timestamps: list[tuple[int, ...]] = []
    materialized_workspaces: list[Path] = []

    class FakeMaterializer:
        def __init__(self, staging_root: Path) -> None:
            assert staging_root == staging.resolve()

        def materialize(
            self,
            views: tuple[FrameSheetView, ...],
            frame_timestamps_ms: tuple[int, ...],
            *,
            columns: int,
            auxiliary_width: int,
            auxiliary_height: int,
        ) -> MaterializedFrameSheet:
            assert tuple(view.label for view in views) == (
                "HEAD_RGB",
                "RIGHT_WRIST_RGB",
            )
            assert tuple(view.source for view in views) == (
                head.resolve(),
                right.resolve(),
            )
            call_index = len(materialized_timestamps)
            expected_timestamps = (
                (0, 500, 1000, 1500, 2000, 2500),
                (500, 750, 1000, 1250),
                (1500, 1750, 2000, 2250),
            )
            assert frame_timestamps_ms == expected_timestamps[call_index]
            assert columns == 3
            assert (auxiliary_width, auxiliary_height) == (240, 135)
            materialized_timestamps.append(frame_timestamps_ms)
            workspace = staging / f"workspace-{call_index}"
            workspace.mkdir()
            materialized_workspaces.append(workspace)
            sheet = workspace / "sheet.jpg"
            sheet.write_bytes(f"sheet-{call_index}".encode())
            return MaterializedFrameSheet(
                path=sheet,
                workspace=workspace,
                frame_timestamps_ms=frame_timestamps_ms,
                time_unit_width=480,
                time_unit_height=405,
                view_rects=(
                    FrameSheetViewRect(
                        label="HEAD_RGB",
                        is_primary=True,
                        x=0,
                        y=0,
                        width=480,
                        height=270,
                    ),
                    FrameSheetViewRect(
                        label="RIGHT_WRIST_RGB",
                        is_primary=False,
                        x=120,
                        y=270,
                        width=240,
                        height=135,
                    ),
                ),
            )

    responses = iter(
        (
            {
                "event_frame_indices": {"transfer": 2, "place": 4},
                "evidence_frame_indices": [2, 4],
            },
            {
                "event_id": "transfer",
                "event_frame_index": 1,
                "evidence_frame_indices": [1, 2],
            },
            {
                "event_id": "place",
                "event_frame_index": 3,
                "evidence_frame_indices": [2, 3],
            },
        )
    )
    generated_requests: list[object] = []
    generated_image_uris: list[str] = []

    async def fake_generate(
        backend_config: object,
        request: object,
        image_uri: str,
    ) -> dict[str, object]:
        generated_requests.append(request)
        generated_image_uris.append(image_uri)
        return next(responses)

    monkeypatch.setattr(
        processor,
        "LocalMultiViewFrameSheetMaterializer",
        FakeMaterializer,
    )
    monkeypatch.setattr(processor, "_generate", fake_generate)

    assert processor.run() == 0
    stdout = capsys.readouterr().out

    record = json.loads((output_dir / "output.jsonl").read_text(encoding="utf-8"))
    experiment = record["experiment"]
    assert [request.stage for request in generated_requests] == [
        "frame-index-fixed-steps",
        "frame-index-refine-event",
        "frame-index-refine-event",
    ]
    assert [request.request_id for request in generated_requests] == [
        "frame-index:episode_000005_multiview",
        "frame-index-refine:episode_000005_multiview:transfer",
        "frame-index-refine:episode_000005_multiview:place",
    ]
    assert [request.sample_timestamps_ms for request in generated_requests] == [
        (0, 500, 1000, 1500, 2000, 2500),
        (500, 750, 1000, 1250),
        (1500, 1750, 2000, 2250),
    ]
    assert len(set(generated_image_uris)) == 3
    assert materialized_timestamps == [
        (0, 500, 1000, 1500, 2000, 2500),
        (500, 750, 1000, 1250),
        (1500, 1750, 2000, 2250),
    ]
    assert all(not workspace.exists() for workspace in materialized_workspaces)
    assert [
        (event["event_id"], event["timestamp_ms"])
        for event in record["coarse_events"]
    ] == [("transfer", 1000), ("place", 2000)]
    assert [
        (segment["segment_id"], segment["start_ms"], segment["end_ms"])
        for segment in record["coarse_segments"]
    ] == [
        ("pick_up", 0, 1000),
        ("transfer", 1000, 2000),
        ("place", 2000, 3000),
    ]
    assert [
        (event["event_id"], event["timestamp_ms"])
        for event in record["events"]
    ] == [("transfer", 750), ("place", 2250)]
    assert [
        (segment["segment_id"], segment["start_ms"], segment["end_ms"])
        for segment in record["segments"]
    ] == [
        ("pick_up", 0, 750),
        ("transfer", 750, 2250),
        ("place", 2250, 3000),
    ]
    assert record["event_refinements"] == [
        {
            "event_id": "transfer",
            "coarse_timestamp_ms": 1000,
            "refined_timestamp_ms": 750,
            "shift_ms": -250,
            "window_start_ms": 500,
            "window_end_ms": 1500,
            "frame_timestamps_ms": [500, 750, 1000, 1250],
            "model_response": {
                "event_id": "transfer",
                "event_frame_index": 1,
                "evidence_frame_indices": [1, 2],
            },
        },
        {
            "event_id": "place",
            "coarse_timestamp_ms": 2000,
            "refined_timestamp_ms": 2250,
            "shift_ms": 250,
            "window_start_ms": 1500,
            "window_end_ms": 2500,
            "frame_timestamps_ms": [1500, 1750, 2000, 2250],
            "model_response": {
                "event_id": "place",
                "event_frame_index": 3,
                "evidence_frame_indices": [2, 3],
            },
        },
    ]
    assert {segment["segment_id"] for segment in record["segments"]} == {
        "pick_up",
        "transfer",
        "place",
    }
    assert record["segments"][0]["start_ms"] == 0
    assert record["segments"][-1]["end_ms"] == info.duration_ms
    assert all(
        current["end_ms"] == following["start_ms"]
        for current, following in zip(
            record["segments"][:-1], record["segments"][1:], strict=True
        )
    )
    video_id = "episode_000005_multiview"
    expected_artifacts = (
        output_dir / f"{video_id}-contact-sheet.jpg",
        output_dir / f"{video_id}-prompt.txt",
        output_dir / f"{video_id}-refine-transfer-contact-sheet.jpg",
        output_dir / f"{video_id}-refine-transfer-prompt.txt",
        output_dir / f"{video_id}-refine-place-contact-sheet.jpg",
        output_dir / f"{video_id}-refine-place-prompt.txt",
    )
    assert all(path.is_file() for path in expected_artifacts)
    assert (
        "refined_event=transfer coarse=1000ms refined=750ms "
        "delta=-250ms [CHANGED]"
    ) in stdout
    assert (
        "refined_event=place coarse=2000ms refined=2250ms "
        "delta=+250ms [CHANGED]"
    ) in stdout
    assert (
        "refined_segment=pick_up coarse=[0, 1000) "
        "refined=[0, 750) start_delta=+0ms "
        "end_delta=-250ms [CHANGED]"
    ) in stdout
    assert (
        "refined_segment=transfer coarse=[1000, 2000) "
        "refined=[750, 2250) start_delta=-250ms "
        "end_delta=+250ms [CHANGED]"
    ) in stdout
    assert (
        "refined_segment=place coarse=[2000, 3000) "
        "refined=[2250, 3000) start_delta=+250ms "
        "end_delta=+0ms [CHANGED]"
    ) in stdout
    for event_id, contact_sheet, prompt in (
        ("transfer", expected_artifacts[2], expected_artifacts[3]),
        ("place", expected_artifacts[4], expected_artifacts[5]),
    ):
        assert f"refine_contact_sheet[{event_id}]={contact_sheet}" in stdout
        assert f"refine_prompt[{event_id}]={prompt}" in stdout
    assert experiment["vllm_version"] == "test-vllm"
    assert experiment["frame_index_prompt_version"] == (
        "ordered-step-multiview-frame-boundary-v1"
    )
    assert experiment["frame_index_stage_version"] == (
        "ordered-step-frame-boundary-stage-v1"
    )
    assert experiment["frame_index_refine_prompt_version"] == (
        "ordered-step-multiview-frame-boundary-refine-v1"
    )
    assert experiment["frame_index_refine_stage_version"] == (
        "ordered-step-frame-boundary-refine-stage-v1"
    )
    assert experiment["task_id"] == "pick-and-place-bolt-v1"
    assert experiment["task_step_order"] == ["pick_up", "transfer", "place"]
    assert experiment["task_plan_sha256"] == hashlib.sha256(
        (CONFIG_ROOT / "task.yaml").read_bytes()
    ).hexdigest()
    assert experiment["sampling_fps"] == 2.0
    assert experiment["refine_sampling_fps"] == 4.0
    assert experiment["refine_window_ms"] == 1000
    assert experiment["ontology_id"] == "pick-and-place-bolt-v1"
    assert experiment["ontology_revision"] == "v1"
    assert experiment["ontology_hash"].startswith("sha256:")
    assert experiment["view_order"] == ["HEAD_RGB", "RIGHT_WRIST_RGB"]
    assert experiment["view_selection"] == {
        "version": processor.WA2_VIEW_SELECTION_VERSION,
        "training_mask_path": str(mask),
        "training_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
        "modality_path": str(modality),
        "modality_sha256": hashlib.sha256(
            modality.read_bytes()
        ).hexdigest(),
        "episodes_path": str(episodes),
        "episodes_sha256": hashlib.sha256(
            episodes.read_bytes()
        ).hexdigest(),
        "episode_index": 5,
        "active_action_keys": ["right_action_chunk_tcp_in_head_cam"],
        "primary_view": "HEAD_RGB",
        "views": ["HEAD_RGB", "RIGHT_WRIST_RGB"],
        "video_sha256": {
            "HEAD_RGB": hashlib.sha256(head.read_bytes()).hexdigest(),
            "RIGHT_WRIST_RGB": hashlib.sha256(right.read_bytes()).hexdigest(),
        },
    }
    assert experiment["layout"] == {
        "columns": 3,
        "requested_columns": 3,
        "time_unit_width": 480,
        "time_unit_height": 405,
        "view_rects": [
            {
                "label": "HEAD_RGB",
                "is_primary": True,
                "x": 0,
                "y": 0,
                "width": 480,
                "height": 270,
            },
            {
                "label": "RIGHT_WRIST_RGB",
                "is_primary": False,
                "x": 120,
                "y": 270,
                "width": 240,
                "height": 135,
            },
        ],
    }


def test_probe_synchronized_views_rejects_empty_views() -> None:
    with pytest.raises(ValueError, match="at least one view"):
        processor.probe_synchronized_views(())


def test_probe_synchronized_views_rejects_pts_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views = tuple(
        FrameSheetView(label=label, source=tmp_path / f"{label}.mp4")
        for label in ("HEAD_RGB", "LEFT_WRIST_RGB", "RIGHT_WRIST_RGB")
    )
    infos = {
        view.source: VideoInfo(
            path=view.source,
            duration_ms=1000,
            width=1280,
            height=720,
            nominal_fps=30,
            frame_timestamps_ms=(0, 500)
            if view.label != "RIGHT_WRIST_RGB"
            else (0, 533),
        )
        for view in views
    }
    monkeypatch.setattr(processor, "probe_video", lambda path: infos[path])

    with pytest.raises(ValueError, match="frame PTS are not synchronized"):
        processor.probe_synchronized_views(views)
