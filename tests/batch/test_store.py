import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import auto_annotation.batch.store as store_module
from auto_annotation.batch.store import (
    BatchRunLockedError,
    BatchStore,
)


FINGERPRINT_A = "sha256:" + "a" * 64
FINGERPRINT_B = "b" * 64


def test_run_lock_is_exclusive_and_released(tmp_path: Path) -> None:
    first = BatchStore(tmp_path / "batch", "processor-v1")
    second = BatchStore(tmp_path / "batch", "processor-v1")

    with first.acquire_run_lock() as lock_path:
        assert lock_path == tmp_path / "batch/processor-v1/run.lock"
        with pytest.raises(BatchRunLockedError, match="already locked"):
            with second.acquire_run_lock():
                pass

    with second.acquire_run_lock():
        pass


@pytest.mark.parametrize("level", ["run", "item", "fingerprint", "attempts"])
def test_descendant_symlinks_are_rejected(
    tmp_path: Path,
    level: str,
) -> None:
    root = tmp_path / "batch"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root = root / "processor-v1"
    item_root = run_root / "items/item-1"
    fingerprint_root = item_root / ("a" * 64)

    if level == "run":
        run_root.symlink_to(outside, target_is_directory=True)
    elif level == "item":
        (run_root / "items").mkdir(parents=True)
        item_root.symlink_to(outside, target_is_directory=True)
    elif level == "fingerprint":
        item_root.mkdir(parents=True)
        fingerprint_root.symlink_to(outside, target_is_directory=True)
    else:
        fingerprint_root.mkdir(parents=True)
        (fingerprint_root / "attempts").symlink_to(
            outside,
            target_is_directory=True,
        )

    store = BatchStore(root, "processor-v1")
    with pytest.raises(ValueError, match="symlink|escapes"):
        store.allocate_attempt("item-1", FINGERPRINT_A)

    assert list(outside.iterdir()) == []


def test_lock_file_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    run_root = root / "processor-v1"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside.lock"
    outside.write_text("do not touch", encoding="utf-8")
    (run_root / "run.lock").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        with BatchStore(root, "processor-v1").acquire_run_lock():
            pass

    assert outside.read_text(encoding="utf-8") == "do not touch"


def test_user_supplied_root_may_be_a_symlink(tmp_path: Path) -> None:
    resolved_root = tmp_path / "resolved-batch"
    resolved_root.mkdir()
    root = tmp_path / "batch"
    root.symlink_to(resolved_root, target_is_directory=True)
    store = BatchStore(root, "processor-v1")

    store.commit_failure(
        "item-1",
        FINGERPRINT_A,
        {"classification": "permanent"},
    )

    assert (
        resolved_root
        / "processor-v1/items/item-1"
        / ("a" * 64)
        / "failure.json"
    ).is_file()


@pytest.mark.parametrize(
    ("run_id", "item_id", "fingerprint"),
    [
        ("../run", "item-1", FINGERPRINT_A),
        ("processor-v1", "../item", FINGERPRINT_A),
        ("processor-v1", "item-1", "not-a-sha256"),
        ("processor/v1", "item-1", FINGERPRINT_A),
        ("processor-v1", r"..\item", FINGERPRINT_A),
    ],
)
def test_unsafe_identity_is_rejected_before_side_effects(
    tmp_path: Path,
    run_id: str,
    item_id: str,
    fingerprint: str,
) -> None:
    root = tmp_path / "batch"
    with pytest.raises(ValueError, match="unsafe|fingerprint"):
        store = BatchStore(root, run_id)
        store.commit_failure(item_id, fingerprint, {"failed": True})

    assert not root.exists()


def test_attempt_allocation_is_monotonic_and_scoped_by_fingerprint(
    tmp_path: Path,
) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")

    assert store.attempt_count("item-1", FINGERPRINT_A) == 0
    first_number, first_path = store.allocate_attempt(
        "item-1", FINGERPRINT_A
    )
    second_number, second_path = store.allocate_attempt(
        "item-1", FINGERPRINT_A
    )
    changed_number, _ = store.allocate_attempt("item-1", FINGERPRINT_B)

    assert (first_number, first_path.name) == (1, "000001")
    assert (second_number, second_path.name) == (2, "000002")
    assert changed_number == 1

    assert store.attempt_count("item-1", FINGERPRINT_A) == 2
    assert store.attempt_count("item-1", FINGERPRINT_B) == 1
    assert (
        tmp_path
        / "batch/processor-v1/items/item-1"
        / ("a" * 64)
        / "attempts/000002"
    ).is_dir()


def test_attempt_allocation_is_atomic_between_threads(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        allocated = list(
            pool.map(
                lambda _: store.allocate_attempt("item-1", FINGERPRINT_A),
                range(16),
            )
        )

    assert sorted(number for number, _ in allocated) == list(range(1, 17))
    assert store.attempt_count("item-1", FINGERPRINT_A) == 16


def test_attempt_json_requires_an_allocated_attempt(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")

    with pytest.raises(ValueError, match="not allocated"):
        store.write_attempt_json(
            "item-1",
            FINGERPRINT_A,
            1,
            "failure",
            {"message": "failed"},
        )

    attempt, _ = store.allocate_attempt("item-1", FINGERPRINT_A)
    reference = store.write_attempt_json(
        "item-1",
        FINGERPRINT_A,
        attempt,
        "failure",
        {"message": "失败"},
    )

    assert reference.path.name == "failure.json"
    assert store.read_attempt_json(
        "item-1", FINGERPRINT_A, attempt, "failure"
    ) == {"message": "失败"}


def test_result_and_failure_markers_are_identity_scoped(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")

    failure = store.commit_failure(
        "item-1",
        FINGERPRINT_A,
        {"attempt_count": 1, "classification": "retryable"},
    )
    assert failure.path.name == "failure.json"
    assert store.read_failure("item-1", FINGERPRINT_A) == {
        "attempt_count": 1,
        "classification": "retryable",
    }
    assert store.read_failure("item-1", FINGERPRINT_B) is None

    result = store.commit_result(
        "item-1",
        FINGERPRINT_A,
        {"record": {"video_id": "item-1"}},
    )

    assert result.path.name == "result.json"
    assert store.read_result("item-1", FINGERPRINT_A) == {
        "record": {"video_id": "item-1"}
    }
    assert store.read_failure("item-1", FINGERPRINT_A) is None
    with pytest.raises(ValueError, match="already has a committed result"):
        store.commit_failure(
            "item-1",
            FINGERPRINT_A,
            {"classification": "permanent"},
        )


def test_marker_write_is_deterministic_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")
    reference = store.commit_failure(
        "item-1",
        FINGERPRINT_A,
        {"value": "old"},
    )
    real_replace = store_module.os.replace

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        assert Path(destination) == reference.path
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {
            "a": 1,
            "value": "new",
        }
        raise OSError("replace failed")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.commit_failure(
            "item-1",
            FINGERPRINT_A,
            {"value": "new", "a": 1},
        )

    assert reference.path.read_text(encoding="utf-8") == '{"value":"old"}'
    assert list(reference.path.parent.iterdir()) == [reference.path]
    monkeypatch.setattr(store_module.os, "replace", real_replace)


def test_run_json_is_written_atomically_at_run_scope(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")

    reference = store.write_run_json(
        "summary",
        {"selected": 2, "succeeded": 1},
    )

    assert reference.path == tmp_path / "batch/processor-v1/summary.json"
    assert reference.path.read_text(encoding="utf-8") == (
        '{"selected":2,"succeeded":1}'
    )


def test_marker_destination_symlink_is_rejected(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")
    marker_parent = (
        tmp_path
        / "batch/processor-v1/items/item-1"
        / ("a" * 64)
    )
    marker_parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"keep":true}', encoding="utf-8")
    (marker_parent / "failure.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        store.commit_failure(
            "item-1",
            FINGERPRINT_A,
            {"keep": False},
        )

    assert outside.read_text(encoding="utf-8") == '{"keep":true}'


def test_reading_absent_markers_has_no_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    store = BatchStore(root, "processor-v1")

    assert store.read_result("item-1", FINGERPRINT_A) is None
    assert store.read_failure("item-1", FINGERPRINT_A) is None
    assert store.attempt_count("item-1", FINGERPRINT_A) == 0

    assert not root.exists()


def test_iter_markers_returns_identity_and_payload(tmp_path: Path) -> None:
    store = BatchStore(tmp_path / "batch", "processor-v1")
    store.commit_result("item-b", FINGERPRINT_B, {"record": "b"})
    store.commit_result("item-a", FINGERPRINT_A, {"record": "a"})
    store.commit_failure(
        "item-c",
        FINGERPRINT_A,
        {"classification": "permanent"},
    )

    assert [
        (marker.item_id, marker.item_fingerprint, marker.payload)
        for marker in store.iter_results()
    ] == [
        ("item-a", FINGERPRINT_A, {"record": "a"}),
        ("item-b", "sha256:" + "b" * 64, {"record": "b"}),
    ]
    assert [
        (marker.item_id, marker.payload)
        for marker in store.iter_failures()
    ] == [("item-c", {"classification": "permanent"})]
