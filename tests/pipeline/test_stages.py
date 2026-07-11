import asyncio
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from auto_annotation.domain.models import ManifestItem, Segment, VideoChunk, VideoInfo
from auto_annotation.inference.base import GenerationRequest, GenerationResponse
from auto_annotation.inference.mock import ScriptedMockBackend
from auto_annotation.media.sampling import build_sample_plan
from auto_annotation.ontology.registry import load_contract
from auto_annotation.pipeline.coarse import run_coarse_stage
from auto_annotation.pipeline.refine import run_boundary_stage


FIXTURES = Path(__file__).parents[1] / "fixtures"


class UnvalidatedBackend:
    model_id = "unvalidated/video-annotator"
    model_revision = "test-v1"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = deque(deepcopy(responses))
        self.calls: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        if not self._responses:
            raise RuntimeError("unvalidated backend exhausted")
        content = self._responses.popleft()
        return GenerationResponse(content=content, raw=deepcopy(content))


def _item() -> ManifestItem:
    return ManifestItem(
        video_id="video-1",
        video_uri=Path("/tmp/video.mp4"),
        dataset_id="demo",
        schema_version="subtask-v1",
        ontology_id="kitchen-v1",
    )


def _info(
    item: ManifestItem,
    frame_timestamps_ms: tuple[int, ...] = tuple(range(0, 20_000, 250)),
) -> VideoInfo:
    return VideoInfo(
        path=item.video_uri,
        duration_ms=20_000,
        width=640,
        height=480,
        nominal_fps=30.0,
        frame_timestamps_ms=frame_timestamps_ms,
    )


def _coarse_content() -> dict[str, Any]:
    return {
        "task_summary": "拿起杯子",
        "segments": [
            {
                "segment_id": "seg-0001",
                "start_ms": 1000,
                "end_ms": 4000,
                "values": {
                    "atomic_action": "grasp",
                    "interaction_object": "cup",
                    "target": "杯柄",
                },
                "sentence": "抓住杯子的把手。",
                "evidence_timestamps_ms": [2000],
            }
        ],
    }


def _boundary_content(boundary_kind: str = "start") -> dict[str, Any]:
    return {
        "segment_id": "seg-0001",
        "boundary_kind": boundary_kind,
        "boundary_ms": 1250 if boundary_kind == "start" else 1750,
        "evidence_timestamps_ms": [1000, 1500],
    }


def _boundary_inputs() -> tuple[ManifestItem, VideoChunk, VideoInfo, list[Segment]]:
    item = _item()
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000)
    segment = _coarse_content()["segments"][0]
    segment["start_ms"] += chunk.start_ms
    segment["end_ms"] += chunk.start_ms
    segment["evidence_timestamps_ms"] = [
        timestamp + chunk.start_ms
        for timestamp in segment["evidence_timestamps_ms"]
    ]
    return item, chunk, _info(item), [Segment.model_validate(segment)]


def test_coarse_then_boundary_refinement() -> None:
    item = ManifestItem(
        video_id="video-1",
        video_uri=Path("/tmp/video.mp4"),
        dataset_id="demo",
        schema_version="subtask-v1",
        ontology_id="kitchen-v1",
    )
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )
    backend = ScriptedMockBackend(
        {
            "coarse:video-1": [
                {
                    "task_summary": "拿起杯子",
                    "segments": [
                        {
                            "segment_id": "seg-0001",
                            "start_ms": 1000,
                            "end_ms": 4000,
                            "values": {
                                "atomic_action": "grasp",
                                "interaction_object": "cup",
                                "target": "杯柄",
                            },
                            "sentence": "抓住杯子的把手。",
                            "evidence_timestamps_ms": [2000],
                        }
                    ],
                }
            ],
            "boundary-start:video-1:seg-0001": [
                {
                    "segment_id": "seg-0001",
                    "boundary_kind": "start",
                    "boundary_ms": 1250,
                    "evidence_timestamps_ms": [1000, 1500],
                }
            ],
            "boundary-end:video-1:seg-0001": [
                {
                    "segment_id": "seg-0001",
                    "boundary_kind": "end",
                    "boundary_ms": 1750,
                    "evidence_timestamps_ms": [1500, 2000],
                }
            ],
        }
    )
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000)
    info = VideoInfo(
        path=item.video_uri,
        duration_ms=20_000,
        width=640,
        height=480,
        nominal_fps=30.0,
        frame_timestamps_ms=tuple(range(0, 20_000, 250)),
    )
    coarse_plan = build_sample_plan(info, chunk, fps=2.0)

    coarse = asyncio.run(run_coarse_stage(item, coarse_plan, contract, backend))
    refined = asyncio.run(
        run_boundary_stage(
            item,
            chunk,
            info,
            coarse.segments,
            backend,
            refine_fps=6.0,
            window_ms=4000,
        )
    )

    assert coarse.task_summary == "拿起杯子"
    assert (coarse.segments[0].start_ms, coarse.segments[0].end_ms) == (
        11_000,
        14_000,
    )
    assert (refined[0].start_ms, refined[0].end_ms) == (11_250, 13_750)
    assert refined[0].values == coarse.segments[0].values
    boundary_calls = [call for call in backend.calls if call.stage.startswith("boundary-")]
    assert [call.stage for call in boundary_calls] == ["boundary-start", "boundary-end"]
    assert all(call.sample_timestamps_ms for call in boundary_calls)


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    [
        ("start_ms", 1000.0, PydanticValidationError),
        ("start_ms", "1000", JSONSchemaValidationError),
        ("evidence_timestamps_ms", [2000.0], PydanticValidationError),
        (
            "evidence_timestamps_ms",
            ["2000"],
            JSONSchemaValidationError,
        ),
    ],
)
def test_coarse_rejects_non_integer_timestamps(
    field: str,
    invalid_value: Any,
    expected_error: type[Exception],
) -> None:
    item = _item()
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000)
    plan = build_sample_plan(_info(item), chunk, fps=2.0)
    content = _coarse_content()
    content["segments"][0][field] = invalid_value
    backend = UnvalidatedBackend([content])
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )

    with pytest.raises(expected_error):
        asyncio.run(run_coarse_stage(item, plan, contract, backend))


@pytest.mark.parametrize("required_field", ["evidence_timestamps_ms", "sentence"])
def test_coarse_validates_untrusted_backend_response_schema(
    required_field: str,
) -> None:
    item = _item()
    chunk = VideoChunk(chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000)
    plan = build_sample_plan(_info(item), chunk, fps=2.0)
    content = _coarse_content()
    del content["segments"][0][required_field]
    backend = UnvalidatedBackend([content])
    contract = load_contract(
        FIXTURES / "schema/subtask-v1.yaml",
        FIXTURES / "ontology/kitchen-v1.yaml",
    )

    with pytest.raises(
        JSONSchemaValidationError,
        match=rf"'{required_field}' is a required property",
    ):
        asyncio.run(run_coarse_stage(item, plan, contract, backend))


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    [
        ("boundary_ms", 1250.0, PydanticValidationError),
        ("boundary_ms", "1250", JSONSchemaValidationError),
        ("evidence_timestamps_ms", [1000.0], PydanticValidationError),
        (
            "evidence_timestamps_ms",
            ["1000"],
            JSONSchemaValidationError,
        ),
    ],
)
def test_boundary_rejects_non_integer_timestamps(
    field: str,
    invalid_value: Any,
    expected_error: type[Exception],
) -> None:
    item, chunk, info, segments = _boundary_inputs()
    invalid = _boundary_content()
    invalid[field] = invalid_value
    backend = UnvalidatedBackend([invalid, _boundary_content("end")])

    with pytest.raises(expected_error):
        asyncio.run(
            run_boundary_stage(
                item,
                chunk,
                info,
                segments,
                backend,
                refine_fps=6.0,
                window_ms=4000,
            )
        )


def test_boundary_rejects_extra_response_fields() -> None:
    item, chunk, info, segments = _boundary_inputs()
    invalid = _boundary_content()
    invalid["unexpected"] = True
    backend = UnvalidatedBackend([invalid, _boundary_content("end")])

    with pytest.raises(JSONSchemaValidationError, match="Additional properties"):
        asyncio.run(
            run_boundary_stage(
                item,
                chunk,
                info,
                segments,
                backend,
                refine_fps=6.0,
                window_ms=4000,
            )
        )


def test_boundary_validates_untrusted_backend_response_schema() -> None:
    item, chunk, info, segments = _boundary_inputs()
    backend = UnvalidatedBackend([_boundary_content("end")])

    with pytest.raises(JSONSchemaValidationError, match=r"'start' was expected"):
        asyncio.run(
            run_boundary_stage(
                item,
                chunk,
                info,
                segments,
                backend,
                refine_fps=6.0,
                window_ms=4000,
            )
        )


def test_boundary_sorts_segments_and_passes_existing_neighbor_context() -> None:
    item = _item()
    chunk = VideoChunk(chunk_id="chunk-0000", start_ms=0, end_ms=20_000)
    info = _info(item)
    chronological = [
        Segment(
            segment_id="seg-0001",
            start_ms=2000,
            end_ms=4000,
            values={"atomic_action": "grasp", "target": "cup"},
            sentence="先拿起杯子",
            evidence_timestamps_ms=[2500],
        ),
        Segment(
            segment_id="seg-0002",
            start_ms=7000,
            end_ms=9000,
            values={"atomic_action": "place", "target": "tray"},
            sentence="再把杯子放到托盘",
            evidence_timestamps_ms=[7500],
        ),
        Segment(
            segment_id="seg-0003",
            start_ms=12_000,
            end_ms=14_000,
            values={"atomic_action": "grasp", "target": "tray"},
            sentence="最后拿起托盘",
            evidence_timestamps_ms=[12_500],
        ),
    ]
    responses = [
        {
            "segment_id": segment.segment_id,
            "boundary_kind": boundary_kind,
            "boundary_ms": 1000,
            "evidence_timestamps_ms": [1000],
        }
        for segment in chronological
        for boundary_kind in ("start", "end")
    ]
    backend = UnvalidatedBackend(responses)

    refined = asyncio.run(
        run_boundary_stage(
            item,
            chunk,
            info,
            list(reversed(chronological)),
            backend,
            refine_fps=6.0,
            window_ms=2000,
        )
    )

    assert [segment.segment_id for segment in refined] == [
        "seg-0001",
        "seg-0002",
        "seg-0003",
    ]
    prompts = {call.request_id: call.prompt for call in backend.calls}
    first_prompt = prompts["boundary-start:video-1:seg-0001"]
    middle_prompt = prompts["boundary-start:video-1:seg-0002"]
    last_prompt = prompts["boundary-start:video-1:seg-0003"]

    assert '"previous_segment"' not in first_prompt
    assert '"next_segment"' in first_prompt
    assert '"segment_id": "seg-0002"' in first_prompt
    assert '"previous_segment"' in middle_prompt
    assert '"next_segment"' in middle_prompt
    assert '"segment_id": "seg-0001"' in middle_prompt
    assert '"segment_id": "seg-0003"' in middle_prompt
    assert '"start_ms": 2000' in middle_prompt
    assert '"end_ms": 14000' in middle_prompt
    assert '"values"' in middle_prompt
    assert '"sentence": "先拿起杯子"' in middle_prompt
    assert '"sentence": "最后拿起托盘"' in middle_prompt
    assert '"previous_segment"' in last_prompt
    assert '"next_segment"' not in last_prompt
    assert '"segment_id": "seg-0002"' in last_prompt


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("segment_id", "seg-other", "'seg-0001' was expected"),
        ("boundary_kind", "end", "'start' was expected"),
    ],
)
def test_boundary_rejects_response_identity_mismatch(
    field: str, invalid_value: str, message: str
) -> None:
    item, chunk, info, segments = _boundary_inputs()
    invalid = _boundary_content()
    invalid[field] = invalid_value
    backend = UnvalidatedBackend([invalid])

    with pytest.raises(JSONSchemaValidationError, match=message):
        asyncio.run(
            run_boundary_stage(
                item,
                chunk,
                info,
                segments,
                backend,
                refine_fps=6.0,
                window_ms=4000,
            )
        )


def test_boundary_reports_window_without_pts_before_request() -> None:
    item, chunk, _, segments = _boundary_inputs()
    info = _info(item, frame_timestamps_ms=(19_500,))
    backend = UnvalidatedBackend([])

    with pytest.raises(ValueError, match=r"seg-0001.*start"):
        asyncio.run(
            run_boundary_stage(
                item,
                chunk,
                info,
                segments,
                backend,
                refine_fps=6.0,
                window_ms=4000,
            )
        )

    assert backend.calls == []


@pytest.mark.parametrize("window_ms", [0, -1])
def test_boundary_rejects_non_positive_window_before_processing(
    window_ms: int,
) -> None:
    item = _item()
    backend = UnvalidatedBackend([])

    with pytest.raises(ValueError, match="window_ms must be positive"):
        asyncio.run(
            run_boundary_stage(
                item,
                VideoChunk(
                    chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000
                ),
                _info(item),
                [],
                backend,
                refine_fps=6.0,
                window_ms=window_ms,
            )
        )

    assert backend.calls == []


@pytest.mark.parametrize("refine_fps", [0.0, -1.0, float("inf"), float("nan")])
def test_boundary_rejects_invalid_fps_before_processing(refine_fps: float) -> None:
    item = _item()
    backend = UnvalidatedBackend([])

    with pytest.raises(ValueError, match="refine_fps must be finite and positive"):
        asyncio.run(
            run_boundary_stage(
                item,
                VideoChunk(
                    chunk_id="chunk-0001", start_ms=10_000, end_ms=20_000
                ),
                _info(item),
                [],
                backend,
                refine_fps=refine_fps,
                window_ms=4000,
            )
        )

    assert backend.calls == []
