import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

import auto_annotation.artifacts.store as artifact_store_module
import auto_annotation.exporters.jsonl as jsonl_module
from auto_annotation.artifacts.store import ArtifactStore
from auto_annotation.exporters.jsonl import write_jsonl


class NestedValue(BaseModel):
    location: Path


class NestedPayload(BaseModel):
    name: str
    children: list[NestedValue]


class ExportRecord(BaseModel):
    video_id: str
    payload: NestedPayload


def test_artifact_write_is_addressed_by_run_video_and_stage(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = {"ok": True, "label": "中文"}

    reference = store.write_json("run-1", "video-1", "coarse", payload)

    assert reference.path == tmp_path / "artifacts/run-1/video-1/coarse.json"
    data = reference.path.read_bytes()
    assert reference.sha256 == f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert len(reference.sha256.removeprefix("sha256:")) == 64
    assert store.read_json(reference) == payload


def test_artifact_write_jsonizes_pydantic_models_at_any_nesting(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = {
        "model": NestedPayload(
            name="parent",
            children=[NestedValue(location=Path("clips/a.mp4"))],
        ),
        "items": [
            NestedValue(location=Path("clips/b.mp4")),
            {"deep": NestedValue(location=Path("clips/c.mp4"))},
        ],
    }

    reference = store.write_json("run-1", "video-1", "coarse", payload)

    assert store.read_json(reference) == {
        "model": {
            "name": "parent",
            "children": [{"location": "clips/a.mp4"}],
        },
        "items": [
            {"location": "clips/b.mp4"},
            {"deep": {"location": "clips/c.mp4"}},
        ],
    }


@pytest.mark.parametrize(
    "component",
    ["", ".", "..", "../video-1", "nested/../video-1", r"..\video-1"],
)
def test_artifact_components_cannot_escape_root(
    tmp_path: Path,
    component: str,
) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)

    for index in range(3):
        components = ["run-1", "video-1", "coarse"]
        components[index] = component
        with pytest.raises(ValueError, match="unsafe artifact component"):
            store.write_json(*components, {"ok": True})

    assert not root.exists()


@pytest.mark.parametrize("symlink_level", ["run", "video"])
def test_artifact_parent_symlink_cannot_escape_root(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_level == "run":
        (root / "run-1").symlink_to(outside, target_is_directory=True)
    else:
        (root / "run-1").mkdir()
        (root / "run-1" / "video-1").symlink_to(
            outside,
            target_is_directory=True,
        )
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="escapes artifact root"):
        store.write_json("run-1", "video-1", "coarse", {"ok": True})

    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("symlink_level", ["run", "video"])
def test_artifact_parent_symlink_is_rejected_even_when_target_stays_in_root(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    target = root / "symlink-target"
    target.mkdir()
    if symlink_level == "run":
        (root / "run-1").symlink_to(target, target_is_directory=True)
    else:
        (root / "run-1").mkdir()
        (root / "run-1" / "video-1").symlink_to(
            target,
            target_is_directory=True,
        )
    store = ArtifactStore(root)

    with pytest.raises(ValueError, match="symlink ancestor"):
        store.write_json("run-1", "video-1", "coarse", {"ok": True})

    assert list(target.iterdir()) == []


def test_artifact_root_may_be_a_user_supplied_symlink(tmp_path: Path) -> None:
    resolved_root = tmp_path / "resolved-artifacts"
    resolved_root.mkdir()
    root = tmp_path / "artifacts"
    root.symlink_to(resolved_root, target_is_directory=True)
    store = ArtifactStore(root)

    reference = store.write_json(
        "run-1",
        "video-1",
        "coarse",
        {"ok": True},
    )

    assert reference.path == root / "run-1/video-1/coarse.json"
    assert (resolved_root / "run-1/video-1/coarse.json").is_file()


def test_artifact_write_fsyncs_then_atomically_replaces_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    existing = store.write_json("run-1", "video-1", "coarse", {"value": "old"})
    real_fsync = artifact_store_module.os.fsync
    real_replace = artifact_store_module.os.replace
    fsync_calls: list[int] = []
    replace_calls: list[tuple[Path, Path]] = []

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == existing.path
        assert destination_path.read_text(encoding="utf-8") == '{"value":"old"}'
        assert source_path.parent == destination_path.parent
        assert source_path.read_text(encoding="utf-8") == '{"value":"new"}'
        replace_calls.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(artifact_store_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(artifact_store_module.os, "replace", recording_replace)

    store.write_json("run-1", "video-1", "coarse", {"value": "new"})

    assert len(fsync_calls) == 1
    assert len(replace_calls) == 1
    assert store.read_json(existing) == {"value": "new"}


def test_artifact_write_removes_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    existing = store.write_json("run-1", "video-1", "coarse", {"value": "old"})

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(artifact_store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.write_json("run-1", "video-1", "coarse", {"value": "new"})

    assert store.read_json(existing) == {"value": "old"}
    assert list(existing.path.parent.iterdir()) == [existing.path]


def test_jsonl_export_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"

    write_jsonl(
        output,
        [
            {"video_id": "b", "label": "second"},
            {"video_id": "a", "label": "first"},
        ],
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {"label": "first", "video_id": "a"},
        {"label": "second", "video_id": "b"},
    ]
    assert output.read_text(encoding="utf-8") == (
        '{"label":"first","video_id":"a"}\n'
        '{"label":"second","video_id":"b"}\n'
    )


def test_jsonl_export_jsonizes_nested_pydantic_records(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    records = (
        ExportRecord(
            video_id=video_id,
            payload=NestedPayload(
                name=f"payload-{video_id}",
                children=[NestedValue(location=Path(f"clips/{video_id}.mp4"))],
            ),
        )
        for video_id in ("b", "a")
    )

    write_jsonl(output, records)

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {
            "video_id": "a",
            "payload": {
                "name": "payload-a",
                "children": [{"location": "clips/a.mp4"}],
            },
        },
        {
            "video_id": "b",
            "payload": {
                "name": "payload-b",
                "children": [{"location": "clips/b.mp4"}],
            },
        },
    ]


def test_jsonl_export_fsyncs_then_atomically_replaces_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text('{"video_id":"old"}\n', encoding="utf-8")
    real_fsync = jsonl_module.os.fsync
    real_replace = jsonl_module.os.replace
    fsync_calls: list[int] = []
    replace_calls: list[tuple[Path, Path]] = []

    def recording_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == output
        assert destination_path.read_text(encoding="utf-8") == (
            '{"video_id":"old"}\n'
        )
        assert source_path.parent == output.parent
        assert source_path.read_text(encoding="utf-8") == (
            '{"video_id":"a"}\n{"video_id":"b"}\n'
        )
        replace_calls.append((source_path, destination_path))
        real_replace(source, destination)

    monkeypatch.setattr(jsonl_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(jsonl_module.os, "replace", recording_replace)

    write_jsonl(output, [{"video_id": "b"}, {"video_id": "a"}])

    assert len(fsync_calls) == 1
    assert len(replace_calls) == 1
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"video_id": "a"},
        {"video_id": "b"},
    ]


def test_jsonl_export_removes_temporary_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text('{"video_id":"old"}\n', encoding="utf-8")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(jsonl_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_jsonl(output, [{"video_id": "new"}])

    assert output.read_text(encoding="utf-8") == '{"video_id":"old"}\n'
    assert list(tmp_path.iterdir()) == [output]
