from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from auto_annotation.domain.models import (
    Event,
    ManifestItem,
    SamplePlan,
    SamplePoint,
    TaskStep,
    VideoChunk,
    VideoInfo,
)
from auto_annotation.inference.base import validate_response
from auto_annotation.ontology.models import (
    AnnotationSchema,
    Ontology,
    OntologyEntry,
    SegmentFieldSpec,
    TextOutputSpec,
)
from auto_annotation.ontology.registry import ResolvedContract
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


def _item() -> ManifestItem:
    return ManifestItem(
        video_id="video-1",
        video_uri=Path("/tmp/video.mp4"),
        dataset_id="demo",
        schema_version="schema-v1",
        ontology_id="ontology-v1",
    )


def _plan() -> SamplePlan:
    return SamplePlan(
        chunk=VideoChunk(
            chunk_id="chunk-0001",
            start_ms=10_000,
            end_ms=13_000,
        ),
        target_fps=2.0,
        points=tuple(
            SamplePoint(
                source_timestamp_ms=10_000 + offset,
                chunk_timestamp_ms=offset,
            )
            for offset in range(0, 3000, 500)
        ),
    )


def _entry(
    entry_id: str,
    temporal_type: str | None,
    *,
    definition: str | None = None,
) -> OntologyEntry:
    temporal = (
        {}
        if temporal_type is None
        else {
            "temporal_type": temporal_type,
            "boundary_rule": f"{entry_id} 的可观察边界规则",
            "positive_cues": [f"{entry_id} 正例线索"],
            "negative_cues": [f"{entry_id} 反例线索"],
        }
    )
    return OntologyEntry.model_validate(
        {
            "id": entry_id,
            "name": entry_id,
            "definition": definition or f"{entry_id} 的闭集定义",
            **temporal,
        }
    )


def _contract(*extra_entries: OntologyEntry) -> ResolvedContract:
    return ResolvedContract(
        AnnotationSchema(
            schema_id="schema-v1",
            segment_values={
                "atomic_action": SegmentFieldSpec(
                    type="enum",
                    source="ontology.atomic_actions",
                )
            },
            text_output=TextOutputSpec(
                field="sentence",
                language="zh-CN",
                style="imperative",
            ),
        ),
        Ontology(
            ontology_id="ontology-v1",
            vocabularies={
                "atomic_actions": [
                    _entry(
                        "pick_up",
                        "point_event",
                        definition="夹爪首次对物体建立稳定控制的状态变化",
                    ),
                    _entry("transfer", "interval_action"),
                    _entry("place", "point_event"),
                    *extra_entries,
                ]
            },
        ),
    )


def _steps() -> tuple[TaskStep, ...]:
    return tuple(
        TaskStep(
            step_id=step_id,
            values={"atomic_action": step_id},
            sentence=sentence,
        )
        for step_id, sentence in (
            ("pick_up", "确认已经稳定抓住物体。"),
            ("transfer", "保持控制并搬运物体。"),
            ("place", "确认物体已经释放。"),
        )
    )


def _event(
    event_id: str,
    timestamp_ms: int,
    *,
    evidence_timestamps_ms: list[int] | None = None,
) -> Event:
    step = next(step for step in _steps() if step.step_id == event_id)
    return Event(
        event_id=event_id,
        timestamp_ms=timestamp_ms,
        values=step.values,
        sentence=step.sentence,
        evidence_timestamps_ms=evidence_timestamps_ms or [],
    )


def _video_info() -> VideoInfo:
    return VideoInfo(
        path=Path("/tmp/video.mp4"),
        duration_ms=20_000,
        width=1280,
        height=720,
        nominal_fps=30.0,
        frame_timestamps_ms=tuple(range(0, 20_000, 100)),
    )


def _refine_plan() -> SamplePlan:
    return SamplePlan(
        chunk=VideoChunk(
            chunk_id="frame-index-refine-transfer",
            start_ms=11_500,
            end_ms=13_500,
        ),
        target_fps=10.0,
        points=tuple(
            SamplePoint(
                source_timestamp_ms=11_500 + offset,
                chunk_timestamp_ms=offset,
            )
            for offset in range(0, 2000, 100)
        ),
    )


def _request(
    *,
    view_labels: tuple[str, ...] = (
        "HEAD_RGB",
        "LEFT_WRIST_RGB",
        "RIGHT_WRIST_RGB",
    ),
):
    return build_fixed_step_frame_request(
        _item(),
        _plan(),
        _steps(),
        _contract(),
        view_labels=view_labels,
        primary_view_label="HEAD_RGB",
    )


def test_ordered_step_frame_request_uses_all_action_and_boundary_semantics() -> None:
    request = _request()

    assert request.stage == "frame-index-fixed-steps"
    assert request.sample_timestamps_ms == (
        10_000,
        10_500,
        11_000,
        11_500,
        12_000,
        12_500,
    )
    assert "F000" in request.prompt
    assert "F005" in request.prompt
    assert "夹爪首次对物体建立稳定控制的状态变化" in request.prompt
    assert "pick_up 的可观察边界规则" in request.prompt
    assert "pick_up 正例线索" in request.prompt
    assert "pick_up 反例线索" in request.prompt
    assert "第一张能够确认后一步已经开始且前一步已经结束的采样帧" in request.prompt
    assert "第一个边界不得选择 F000" in request.prompt
    assert "无缝划分为与动作数相同的半开区间" in request.prompt
    assert '"boundary_id": "transfer"' in request.prompt
    assert '"ends_step_id": "pick_up"' in request.prompt
    assert '"starts_step_id": "transfer"' in request.prompt
    assert "主视角 \"HEAD_RGB\" 位于上方并显示为大图" in request.prompt
    assert "两个辅助视角位于下方，从左到右依次为" in request.prompt
    assert "LEFT_WRIST_RGB" in request.prompt
    assert "RIGHT_WRIST_RGB" in request.prompt

    event_frames = request.response_schema["properties"][
        "event_frame_indices"
    ]
    assert event_frames == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "transfer": {"type": "integer", "minimum": 1, "maximum": 5},
            "place": {"type": "integer", "minimum": 0, "maximum": 5},
        },
        "required": ["transfer", "place"],
    }
    evidence = request.response_schema["properties"][
        "evidence_frame_indices"
    ]
    assert "uniqueItems" not in evidence
    assert FRAME_INDEX_PROMPT_VERSION == (
        "ordered-step-multiview-frame-boundary-v1"
    )


def test_ordered_step_frame_request_supports_more_than_three_steps() -> None:
    contract = _contract(
        _entry("inspect", "interval_action"),
        _entry("approved", "point_event"),
    )
    steps = (
        *_steps(),
        TaskStep(
            step_id="inspect",
            values={"atomic_action": "inspect"},
            sentence="检查物体。",
        ),
        TaskStep(
            step_id="approved",
            values={"atomic_action": "approved"},
            sentence="确认检查完成。",
        ),
    )

    request = build_fixed_step_frame_request(
        _item(),
        _plan(),
        steps,
        contract,
        view_labels=("HEAD_RGB",),
        primary_view_label="HEAD_RGB",
    )

    event_frames = request.response_schema["properties"]["event_frame_indices"]
    assert event_frames["required"] == [
        "transfer",
        "place",
        "inspect",
        "approved",
    ]
    assert set(event_frames["properties"]) == set(event_frames["required"])


def test_ordered_step_frame_request_requires_enough_distinct_frames() -> None:
    contract = _contract(
        _entry("inspect", "interval_action"),
        _entry("approved", "point_event"),
        _entry("archive", "state"),
        _entry("finish", "point_event"),
    )
    extra_steps = tuple(
        TaskStep(
            step_id=step_id,
            values={"atomic_action": step_id},
            sentence=f"{step_id}。",
        )
        for step_id in ("inspect", "approved", "archive", "finish")
    )

    with pytest.raises(ValueError, match="distinct frame per ordered step"):
        build_fixed_step_frame_request(
            _item(),
            _plan(),
            (*_steps(), *extra_steps),
            contract,
            view_labels=("HEAD_RGB",),
            primary_view_label="HEAD_RGB",
        )


def test_fixed_step_frame_request_schema_rejects_first_event_at_f000() -> None:
    with pytest.raises(JSONSchemaValidationError):
        validate_response(
            _request(),
            {
                "event_frame_indices": {"transfer": 0, "place": 4},
                "evidence_frame_indices": [],
            },
        )


def test_fixed_step_frame_request_describes_single_primary_view_only() -> None:
    request = _request(view_labels=("HEAD_RGB",))

    assert "单视角视频帧接触图" in request.prompt
    assert "主视角 \"HEAD_RGB\"" in request.prompt
    assert "多视角" not in request.prompt
    assert "交叉查看" not in request.prompt
    assert "腕部" not in request.prompt
    assert "LEFT_WRIST_RGB" not in request.prompt
    assert "RIGHT_WRIST_RGB" not in request.prompt


def test_fixed_step_frame_request_describes_one_centered_auxiliary_view() -> None:
    request = _request(view_labels=("HEAD_RGB", "RIGHT_WRIST_RGB"))

    assert "主视角 \"HEAD_RGB\" 位于上方并显示为大图" in request.prompt
    assert "唯一辅助视角位于下方并居中显示" in request.prompt
    assert '["RIGHT_WRIST_RGB"]' in request.prompt
    assert "交叉查看上述两个实际视角" in request.prompt
    assert "LEFT_WRIST_RGB" not in request.prompt


@pytest.mark.parametrize(
    "view_labels",
    ((), ("HEAD_RGB", "HEAD_RGB")),
)
def test_fixed_step_frame_request_rejects_invalid_view_labels(
    view_labels: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="view"):
        _request(view_labels=view_labels)


def test_fixed_step_frame_request_rejects_unknown_primary_view() -> None:
    with pytest.raises(ValueError, match="primary view label"):
        build_fixed_step_frame_request(
            _item(),
            _plan(),
            _steps(),
            _contract(),
            view_labels=("HEAD_RGB", "RIGHT_WRIST_RGB"),
            primary_view_label="LEFT_WRIST_RGB",
        )


def test_fixed_step_frame_request_rejects_more_than_three_views() -> None:
    with pytest.raises(ValueError, match="at most three views"):
        _request(
            view_labels=(
                "HEAD_RGB",
                "LEFT_WRIST_RGB",
                "RIGHT_WRIST_RGB",
                "REAR_RGB",
            )
        )


def test_ordered_step_frame_request_accepts_non_alternating_temporal_types() -> None:
    contract = _contract(_entry("wait", "state"))
    steps = (
        _steps()[0],
        TaskStep(
            step_id="wait",
            values={"atomic_action": "wait"},
            sentence="等待。",
        ),
        _steps()[2],
    )

    request = build_fixed_step_frame_request(
        _item(),
        _plan(),
        steps,
        contract,
        view_labels=("HEAD_RGB",),
        primary_view_label="HEAD_RGB",
    )

    assert request.response_schema["properties"]["event_frame_indices"][
        "required"
    ] == ["wait", "place"]


def test_fixed_step_frame_request_requires_temporal_metadata() -> None:
    contract = _contract(_entry("undefined", None))
    steps = (
        TaskStep(
            step_id="undefined",
            values={"atomic_action": "undefined"},
            sentence="未定义事件。",
        ),
        _steps()[1],
        _steps()[2],
    )

    with pytest.raises(ValueError, match="requires temporal metadata"):
        build_fixed_step_frame_request(
            _item(),
            _plan(),
            steps,
            contract,
            view_labels=("HEAD_RGB",),
            primary_view_label="HEAD_RGB",
        )


def test_event_frames_create_boundaries_and_full_coverage_segments() -> None:
    annotation = annotation_from_event_frames(
        task_summary="拾取并放置物体",
        plan=_plan(),
        steps=_steps(),
        contract=_contract(),
        content={
            "event_frame_indices": {"transfer": 1, "place": 4},
            "evidence_frame_indices": [0, 1, 3, 4],
        },
    )

    assert [
        (event.event_id, event.timestamp_ms)
        for event in annotation.events
    ] == [
        ("transfer", 10_500),
        ("place", 12_000),
    ]
    assert [event.evidence_timestamps_ms for event in annotation.events] == [
        [10_500],
        [12_000],
    ]
    assert [
        (segment.segment_id, segment.start_ms, segment.end_ms)
        for segment in annotation.segments
    ] == [
        ("pick_up", 10_000, 10_500),
        ("transfer", 10_500, 12_000),
        ("place", 12_000, 13_000),
    ]
    assert [
        segment.evidence_timestamps_ms for segment in annotation.segments
    ] == [
        [10_000],
        [10_500, 11_500],
        [12_000],
    ]
    assert annotation.segments[0].start_ms == _plan().chunk.start_ms
    assert annotation.segments[-1].end_ms == _plan().chunk.end_ms
    assert all(
        left.end_ms == right.start_ms
        for left, right in zip(
            annotation.segments,
            annotation.segments[1:],
        )
    )
    assert FRAME_INDEX_STAGE_VERSION == "ordered-step-frame-boundary-stage-v1"


@pytest.mark.parametrize(
    "event_frame_indices",
    (
        {"transfer": 4, "place": 2},
        {"transfer": 2, "place": 2},
        {"transfer": -1, "place": 4},
        {"transfer": 2, "place": 6},
    ),
)
def test_event_frames_reject_invalid_indices(
    event_frame_indices: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="event frame"):
        annotation_from_event_frames(
            task_summary="拾取并放置物体",
            plan=_plan(),
            steps=_steps(),
            contract=_contract(),
            content={
                "event_frame_indices": event_frame_indices,
                "evidence_frame_indices": [],
            },
        )


def test_event_frames_reject_first_event_at_timeline_start() -> None:
    with pytest.raises(ValueError, match="strictly inside .*timeline"):
        annotation_from_event_frames(
            task_summary="拾取并放置物体",
            plan=_plan(),
            steps=_steps(),
            contract=_contract(),
            content={
                "event_frame_indices": {"transfer": 0, "place": 4},
                "evidence_frame_indices": [],
            },
        )


@pytest.mark.parametrize(
    "event_frame_indices",
    (
        {"transfer": 0},
        {"transfer": 0, "place": 4, "unknown": 5},
    ),
)
def test_event_frames_require_exact_point_event_keys(
    event_frame_indices: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="exactly match"):
        annotation_from_event_frames(
            task_summary="拾取并放置物体",
            plan=_plan(),
            steps=_steps(),
            contract=_contract(),
            content={
                "event_frame_indices": event_frame_indices,
                "evidence_frame_indices": [],
            },
        )


def test_event_frames_reject_duplicate_evidence_indices() -> None:
    with pytest.raises(ValueError, match="evidence frame indices must be unique"):
        annotation_from_event_frames(
            task_summary="拾取并放置物体",
            plan=_plan(),
            steps=_steps(),
            contract=_contract(),
            content={
                "event_frame_indices": {"transfer": 1, "place": 4},
                "evidence_frame_indices": [1, 1],
            },
        )


@pytest.mark.parametrize("evidence_frame_index", (-1, 6))
def test_event_frames_reject_out_of_range_evidence_indices(
    evidence_frame_index: int,
) -> None:
    with pytest.raises(ValueError, match="evidence frame index"):
        annotation_from_event_frames(
            task_summary="拾取并放置物体",
            plan=_plan(),
            steps=_steps(),
            contract=_contract(),
            content={
                "event_frame_indices": {"transfer": 1, "place": 4},
                "evidence_frame_indices": [evidence_frame_index],
            },
        )


@pytest.mark.parametrize(
    ("coarse_timestamp_ms", "expected_window"),
    (
        (12_500, (11_500, 13_500)),
        (10_100, (10_000, 12_000)),
        (14_900, (13_000, 15_000)),
    ),
)
def test_event_refine_plan_centers_and_clips_window_at_chunk_edges(
    coarse_timestamp_ms: int,
    expected_window: tuple[int, int],
) -> None:
    chunk = VideoChunk(
        chunk_id="chunk-0001",
        start_ms=10_000,
        end_ms=15_000,
    )

    plan = build_event_refine_plan(
        _video_info(),
        chunk,
        _event("transfer", coarse_timestamp_ms),
        10.0,
        2000,
    )

    assert (plan.chunk.start_ms, plan.chunk.end_ms) == expected_window
    assert plan.target_fps == 10.0
    assert plan.points[0].source_timestamp_ms == expected_window[0]
    assert plan.points[-1].source_timestamp_ms == expected_window[1] - 100
    assert all(
        plan.chunk.start_ms <= point.source_timestamp_ms < plan.chunk.end_ms
        for point in plan.points
    )


def test_event_refine_plan_keeps_coarse_pts_in_shifted_tail_window() -> None:
    info = VideoInfo(
        path=Path("/tmp/video.mp4"),
        duration_ms=1033,
        width=1280,
        height=720,
        nominal_fps=10.0,
        frame_timestamps_ms=tuple(range(0, 1001, 100)),
    )

    plan = build_event_refine_plan(
        info,
        VideoChunk(chunk_id="chunk-0001", start_ms=0, end_ms=1033),
        _event("transfer", 1000),
        10.0,
        1000,
    )

    assert (plan.chunk.start_ms, plan.chunk.end_ms) == (33, 1033)
    assert plan.points[-1].source_timestamp_ms == 1000


def test_event_refine_request_has_singular_identity_and_closed_schema() -> None:
    event = _event(
        "transfer",
        12_500,
        evidence_timestamps_ms=[12_345],
    )
    request = build_event_refine_frame_request(
        _item(),
        _refine_plan(),
        event,
        _contract(),
        steps=_steps(),
        previous_event=None,
        next_event=_event(
            "place",
            14_001,
            evidence_timestamps_ms=[14_002],
        ),
        view_labels=("HEAD_RGB", "RIGHT_WRIST_RGB"),
        primary_view_label="HEAD_RGB",
    )

    assert request.request_id == "frame-index-refine:video-1:transfer"
    assert request.stage == "frame-index-refine-event"
    assert (request.media_start_ms, request.media_end_ms) == (11_500, 13_500)
    assert request.sample_timestamps_ms == tuple(
        point.source_timestamp_ms for point in _refine_plan().points
    )
    assert "夹爪首次对物体建立稳定控制的状态变化" in request.prompt
    assert "pick_up 的可观察边界规则" in request.prompt
    assert "transfer 的可观察边界规则" in request.prompt
    assert "pick_up 正例线索" in request.prompt
    assert "pick_up 反例线索" in request.prompt
    assert "确认已经稳定抓住物体。" in request.prompt
    assert "保持控制并搬运物体。" in request.prompt
    assert '"atomic_action": "pick_up"' in request.prompt
    assert '"atomic_action": "transfer"' in request.prompt
    assert '"target_boundary"' in request.prompt
    assert '"next_event_id": "place"' in request.prompt
    assert '"coarse_event"' not in request.prompt
    assert '"timestamp_ms": 12500' not in request.prompt
    assert '"evidence_timestamps_ms"' not in request.prompt
    assert "12345" not in request.prompt
    assert "14001" not in request.prompt
    assert "14002" not in request.prompt
    assert "窗口中央帧不是先验答案" in request.prompt
    assert "HEAD_RGB" in request.prompt
    assert "RIGHT_WRIST_RGB" in request.prompt
    assert request.response_schema == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_id": {"const": "transfer"},
            "event_frame_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": 19,
            },
            "evidence_frame_indices": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 19,
                },
            },
        },
        "required": [
            "event_id",
            "event_frame_index",
            "evidence_frame_indices",
        ],
    }
    assert FRAME_INDEX_REFINE_PROMPT_VERSION == (
        "ordered-step-multiview-frame-boundary-refine-v1"
    )
    assert FRAME_INDEX_REFINE_STAGE_VERSION == (
        "ordered-step-frame-boundary-refine-stage-v1"
    )


def test_event_refine_request_hides_previous_event_time_and_evidence() -> None:
    request = build_event_refine_frame_request(
        _item(),
        _refine_plan(),
        _event("place", 12_500),
        _contract(),
        steps=_steps(),
        previous_event=_event(
            "transfer",
            10_999,
            evidence_timestamps_ms=[10_998],
        ),
        next_event=None,
        view_labels=("HEAD_RGB", "RIGHT_WRIST_RGB"),
        primary_view_label="HEAD_RGB",
    )

    assert '"previous_event_id": "transfer"' in request.prompt
    assert "10999" not in request.prompt
    assert "10998" not in request.prompt


@pytest.mark.parametrize(
    "content",
    (
        {
            "event_id": "place",
            "event_frame_index": 3,
            "evidence_frame_indices": [],
        },
        {
            "event_id": "transfer",
            "event_frame_index": 20,
            "evidence_frame_indices": [],
        },
        {
            "event_id": "transfer",
            "event_frame_index": 3,
            "evidence_frame_indices": [20],
        },
        {
            "event_id": "transfer",
            "event_frame_index": 3,
            "evidence_frame_indices": [],
            "unexpected": True,
        },
    ),
)
def test_event_refine_request_schema_rejects_wrong_identity_or_indices(
    content: dict[str, object],
) -> None:
    request = build_event_refine_frame_request(
        _item(),
        _refine_plan(),
        _event("transfer", 12_500),
        _contract(),
        steps=_steps(),
        previous_event=None,
        next_event=_event("place", 14_000),
        view_labels=("HEAD_RGB", "RIGHT_WRIST_RGB"),
        primary_view_label="HEAD_RGB",
    )

    with pytest.raises(JSONSchemaValidationError):
        validate_response(request, content)


@pytest.mark.parametrize(
    "content",
    (
        {
            "event_id": "transfer",
            "event_frame_index": 1.0,
            "evidence_frame_indices": [],
        },
        {
            "event_id": "transfer",
            "event_frame_index": 1,
            "evidence_frame_indices": [2.0],
        },
    ),
)
def test_event_refinement_rejects_non_integer_indices_strictly(
    content: dict[str, object],
) -> None:
    with pytest.raises(PydanticValidationError):
        event_from_refinement_frame(
            coarse_event=_event("transfer", 11_600),
            plan=_refine_plan(),
            content=content,
        )


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (
            {
                "event_id": "place",
                "event_frame_index": 1,
                "evidence_frame_indices": [],
            },
            "event_id",
        ),
        (
            {
                "event_id": "transfer",
                "event_frame_index": 20,
                "evidence_frame_indices": [],
            },
            "event frame index",
        ),
        (
            {
                "event_id": "transfer",
                "event_frame_index": 1,
                "evidence_frame_indices": [20],
            },
            "evidence frame index",
        ),
        (
            {
                "event_id": "transfer",
                "event_frame_index": 1,
                "evidence_frame_indices": [2, 2],
            },
            "evidence frame indices must be unique",
        ),
    ),
)
def test_event_refinement_rejects_invalid_identity_index_and_evidence(
    content: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        event_from_refinement_frame(
            coarse_event=_event("transfer", 11_600),
            plan=_refine_plan(),
            content=content,
        )


def test_event_refinement_maps_local_indices_to_absolute_pts() -> None:
    coarse_event = _event("transfer", 11_600)
    plan = SamplePlan(
        chunk=VideoChunk(
            chunk_id="frame-index-refine-transfer",
            start_ms=11_500,
            end_ms=12_000,
        ),
        target_fps=10.0,
        points=(
            SamplePoint(source_timestamp_ms=11_510, chunk_timestamp_ms=10),
            SamplePoint(source_timestamp_ms=11_607, chunk_timestamp_ms=107),
            SamplePoint(source_timestamp_ms=11_709, chunk_timestamp_ms=209),
        ),
    )

    refined = event_from_refinement_frame(
        coarse_event=coarse_event,
        plan=plan,
        content={
            "event_id": "transfer",
            "event_frame_index": 1,
            "evidence_frame_indices": [2, 0],
        },
    )

    assert refined.timestamp_ms == 11_607
    assert refined.evidence_timestamps_ms == [11_510, 11_709]
    assert refined.values == coarse_event.values
    assert refined.sentence == coarse_event.sentence
    assert coarse_event.timestamp_ms == 11_600


def test_refined_events_rebuild_full_coverage_segments() -> None:
    annotation = annotation_from_events(
        task_summary="拾取并放置物体",
        events=[
            _event("transfer", 10_400, evidence_timestamps_ms=[10_400, 11_000]),
            _event("place", 12_800, evidence_timestamps_ms=[12_700, 12_800]),
        ],
        steps=_steps(),
        contract=_contract(),
        timeline=_plan().chunk,
    )

    assert [(event.event_id, event.timestamp_ms) for event in annotation.events] == [
        ("transfer", 10_400),
        ("place", 12_800),
    ]
    assert [
        (segment.segment_id, segment.start_ms, segment.end_ms)
        for segment in annotation.segments
    ] == [
        ("pick_up", 10_000, 10_400),
        ("transfer", 10_400, 12_800),
        ("place", 12_800, 13_000),
    ]
    assert [segment.values for segment in annotation.segments] == [
        step.values for step in _steps()
    ]
    assert [segment.sentence for segment in annotation.segments] == [
        step.sentence for step in _steps()
    ]
    assert [
        segment.evidence_timestamps_ms for segment in annotation.segments
    ] == [
        [],
        [10_400, 11_000, 12_700],
        [12_800],
    ]
    assert annotation.segments[0].start_ms == _plan().chunk.start_ms
    assert annotation.segments[-1].end_ms == _plan().chunk.end_ms
    assert all(
        left.end_ms == right.start_ms
        for left, right in zip(
            annotation.segments,
            annotation.segments[1:],
        )
    )


def test_refined_events_reject_reversed_point_event_timestamps() -> None:
    with pytest.raises(ValueError, match="increasing"):
        annotation_from_events(
            task_summary="拾取并放置物体",
            events=[
                _event("transfer", 12_500),
                _event("place", 12_000),
            ],
            steps=_steps(),
            contract=_contract(),
            timeline=_plan().chunk,
        )


@pytest.mark.parametrize(
    "events",
    (
        [_event("transfer", 10_000), _event("place", 12_000)],
        [_event("transfer", 10_500), _event("place", 13_000)],
    ),
)
def test_refined_events_must_be_strictly_inside_timeline(
    events: list[Event],
) -> None:
    with pytest.raises(ValueError, match="strictly inside .*timeline"):
        annotation_from_events(
            task_summary="拾取并放置物体",
            events=events,
            steps=_steps(),
            contract=_contract(),
            timeline=_plan().chunk,
        )


def test_annotation_from_events_supports_more_than_three_steps() -> None:
    contract = _contract(
        _entry("inspect", "interval_action"),
        _entry("approved", "point_event"),
    )
    steps = (
        *_steps(),
        TaskStep(
            step_id="inspect",
            values={"atomic_action": "inspect"},
            sentence="检查物体。",
        ),
        TaskStep(
            step_id="approved",
            values={"atomic_action": "approved"},
            sentence="确认检查完成。",
        ),
    )

    boundary_timestamps = (10_500, 11_000, 11_500, 12_000)
    events = [
        Event(
            event_id=step.step_id,
            timestamp_ms=timestamp_ms,
            values=step.values,
            sentence=step.sentence,
        )
        for step, timestamp_ms in zip(
            steps[1:], boundary_timestamps, strict=True
        )
    ]

    annotation = annotation_from_events(
        task_summary="拾取、放置并检查物体",
        events=events,
        steps=steps,
        contract=contract,
        timeline=_plan().chunk,
    )

    assert [segment.segment_id for segment in annotation.segments] == [
        "pick_up",
        "transfer",
        "place",
        "inspect",
        "approved",
    ]
    assert [segment.start_ms for segment in annotation.segments] == [
        10_000,
        *boundary_timestamps,
    ]
    assert annotation.segments[-1].end_ms == 13_000
