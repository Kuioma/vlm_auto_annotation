import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import auto_annotation.batch.runner as runner_module
from auto_annotation.batch.models import (
    BatchConfig,
    BatchExecutionConfig,
    BatchItem,
    BatchOutputConfig,
    PluginSpec,
    ProcessorContext,
    ProcessorResult,
)
from auto_annotation.batch.plugins import BatchItemError
from auto_annotation.batch.runner import run_batch
from auto_annotation.batch.store import BatchStore


class FakeSource:
    def __init__(self, items: tuple[BatchItem, ...]) -> None:
        self.items = items

    def discover(self) -> tuple[BatchItem, ...]:
        return self.items


class FakeSession:
    def __init__(
        self,
        calls: list[str],
        failures_before_success: dict[str, int] | None = None,
        *,
        permanent_failures: set[str] | None = None,
    ) -> None:
        self.calls = calls
        self.failures_before_success = failures_before_success or {}
        self.permanent_failures = permanent_failures or set()
        self.call_count: dict[str, int] = {}

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult:
        self.calls.append(item.item_id)
        count = self.call_count.get(item.item_id, 0) + 1
        self.call_count[item.item_id] = count
        if item.item_id in self.permanent_failures:
            raise BatchItemError("permanent failure", retryable=False)
        if count <= self.failures_before_success.get(item.item_id, 0):
            raise BatchItemError("temporary failure", retryable=True)
        artifact = context.output_dir / "artifact.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(item.item_id, encoding="utf-8")
        return ProcessorResult(
            record={"item": item.item_id},
            artifacts=(artifact,),
        )


class FakeProcessorProvider:
    def __init__(self, identity: str, session: FakeSession) -> None:
        self._identity = identity
        self.session = session
        self.open_count = 0
        self.preflight_count = 0
        self.preflight_items: tuple[BatchItem, ...] = ()

    @property
    def identity(self) -> dict[str, str]:
        return {"fake_processor": self._identity}

    def preflight(self, items: tuple[BatchItem, ...]) -> None:
        self.preflight_count += 1
        self.preflight_items = items

    @asynccontextmanager
    async def open(self):
        self.open_count += 1
        yield self.session


class ConcurrencyTrackingSession:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.started: list[str] = []

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult:
        self.started.append(item.item_id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # Different durations force completion order to differ from item
            # order while leaving enough overlap to exercise the worker bound.
            item_number = int(item.item_id.rsplit("-", 1)[1])
            await asyncio.sleep(0.01 * (4 - (item_number % 3)))
            artifact = context.output_dir / "artifact.txt"
            artifact.write_text(item.item_id, encoding="utf-8")
            return ProcessorResult(
                record={"item": item.item_id},
                artifacts=(artifact,),
            )
        finally:
            self.active -= 1


class FailFastWindowSession:
    def __init__(self, *, window_size: int) -> None:
        self.window_size = window_size
        self.started: list[str] = []
        self.initial_window_started = asyncio.Event()

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult:
        self.started.append(item.item_id)
        if len(self.started) == self.window_size:
            self.initial_window_started.set()

        # This timeout keeps the RED test bounded if a serial implementation
        # accidentally reaches this session.
        await asyncio.wait_for(
            self.initial_window_started.wait(),
            timeout=0.5,
        )
        if item.item_id == "item-1":
            raise BatchItemError("stop the batch", retryable=False)

        # Already-started work is allowed to finish after item-1 fails.
        await asyncio.sleep(0.05)
        artifact = context.output_dir / "artifact.txt"
        artifact.write_text(item.item_id, encoding="utf-8")
        return ProcessorResult(
            record={"item": item.item_id},
            artifacts=(artifact,),
        )


class CancellationTrackingSession:
    def __init__(self, *, expected_active: int) -> None:
        self.expected_active = expected_active
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.cleaned_up: list[str] = []
        self.all_started = asyncio.Event()
        self.never_finishes = asyncio.Event()

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult:
        self.started.append(item.item_id)
        if len(self.started) == self.expected_active:
            self.all_started.set()
        try:
            await self.never_finishes.wait()
        except asyncio.CancelledError:
            self.cancelled.append(item.item_id)
            # Cancellation cleanup may itself need to await processor resources,
            # such as terminating and reaping a child process group.
            await asyncio.sleep(0)
            raise
        finally:
            self.cleaned_up.append(item.item_id)

        raise AssertionError("cancellation test processor unexpectedly finished")


class CancellationTrackingProvider:
    def __init__(self, session: CancellationTrackingSession) -> None:
        self.session = session
        self.open_count = 0
        self.exit_count = 0
        self.exit_after_cleanup = False

    @asynccontextmanager
    async def open(self):
        self.open_count += 1
        try:
            yield self.session
        finally:
            self.exit_count += 1
            self.exit_after_cleanup = (
                len(self.session.cleaned_up)
                == self.session.expected_active
            )


def _config(
    workspace: Path,
    *,
    resume: bool = True,
    retry_terminal_failures: bool = False,
    max_attempts: int = 2,
    continue_on_error: bool = True,
    concurrency: int = 1,
    start_index: int = 0,
    end_index: int | None = None,
    limit: int | None = None,
) -> BatchConfig:
    unused_plugin = PluginSpec(plugin="unused.module:create", config={})
    return BatchConfig(
        version=1,
        source=unused_plugin,
        processor=unused_plugin,
        execution=BatchExecutionConfig(
            resume=resume,
            retry_terminal_failures=retry_terminal_failures,
            max_attempts=max_attempts,
            retry_backoff_s=0,
            continue_on_error=continue_on_error,
            concurrency=concurrency,
            start_index=start_index,
            end_index=end_index,
            limit=limit,
        ),
        output=BatchOutputConfig(workspace=workspace),
    )


def _item(item_id: str, fingerprint_digit: str = "1") -> BatchItem:
    return BatchItem(
        item_id=item_id,
        dataset_id="demo",
        media={"HEAD_RGB": Path(f"/tmp/{item_id}.mp4")},
        primary_media="HEAD_RGB",
        context={},
        metadata={},
        input_fingerprint=f"sha256:{fingerprint_digit * 64}",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_batch_runner_isolates_failures_and_exports_processor_agnostic_records(
    tmp_path: Path,
) -> None:
    items = (_item("item-1"), _item("item-2"), _item("item-3"))
    calls: list[str] = []
    provider = FakeProcessorProvider(
        "v1",
        FakeSession(calls, permanent_failures={"item-2"}),
    )

    summary = run_batch(
        _config(tmp_path / "workspace", max_attempts=1),
        source_provider=FakeSource(items),
        processor_provider=provider,
    )

    assert calls == ["item-1", "item-2", "item-3"]
    assert provider.open_count == 1
    assert (
        summary.selected,
        summary.succeeded,
        summary.skipped,
        summary.failed,
        summary.attempted,
    ) == (3, 2, 0, 1, 3)
    assert _read_jsonl(summary.output_path) == [
        {"item": "item-1"},
        {"item": "item-3"},
    ]
    output_index = _read_jsonl(summary.output_index_path)
    assert [
        (entry["line_number"], entry["item_id"], entry["dataset_id"])
        for entry in output_index
    ] == [
        (1, "item-1", "demo"),
        (2, "item-3", "demo"),
    ]
    assert all(
        str(entry["record_sha256"]).startswith("sha256:")
        for entry in output_index
    )
    failures = _read_jsonl(summary.failures_path)
    assert [failure["item_id"] for failure in failures] == ["item-2"]
    assert failures[0]["retryable"] is False


def test_batch_runner_honors_concurrency_bound_and_exports_stably(
    tmp_path: Path,
) -> None:
    items = tuple(_item(f"item-{index}") for index in range(7, 0, -1))
    session = ConcurrencyTrackingSession()
    provider = FakeProcessorProvider("v1", session)

    summary = run_batch(
        _config(
            tmp_path / "workspace",
            max_attempts=1,
            concurrency=3,
        ),
        source_provider=FakeSource(items),
        processor_provider=provider,
    )

    assert session.max_active == 3
    assert sorted(session.started) == sorted(item.item_id for item in items)
    assert provider.open_count == 1
    assert (
        summary.selected,
        summary.succeeded,
        summary.failed,
        summary.pending,
        summary.attempted,
    ) == (7, 7, 0, 0, 7)
    assert _read_jsonl(summary.output_path) == [
        {"item": f"item-{index}"} for index in range(1, 8)
    ]


def test_fail_fast_stops_refilling_after_first_concurrent_failure(
    tmp_path: Path,
) -> None:
    items = tuple(_item(f"item-{index}") for index in range(1, 6))
    session = FailFastWindowSession(window_size=3)
    provider = FakeProcessorProvider("v1", session)

    summary = run_batch(
        _config(
            tmp_path / "workspace",
            max_attempts=1,
            continue_on_error=False,
            concurrency=3,
        ),
        source_provider=FakeSource(items),
        processor_provider=provider,
    )

    assert session.started == ["item-1", "item-2", "item-3"]
    assert provider.open_count == 1
    assert (
        summary.selected,
        summary.succeeded,
        summary.failed,
        summary.pending,
        summary.attempted,
    ) == (5, 2, 1, 2, 3)
    assert _read_jsonl(summary.output_path) == [
        {"item": "item-2"},
        {"item": "item-3"},
    ]


def test_external_cancellation_drains_workers_before_provider_exit(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        config = _config(
            tmp_path / "workspace",
            max_attempts=1,
            concurrency=2,
        )
        items = tuple(_item(f"item-{index}") for index in range(1, 4))
        store = BatchStore(config.output.workspace, "run-cancellation-test")
        actions, skipped = runner_module._prepare_actions(
            store,
            items,
            config,
            write_interrupted_failures=False,
        )
        session = CancellationTrackingSession(expected_active=2)
        provider = CancellationTrackingProvider(session)

        batch_task = asyncio.create_task(
            runner_module._process_actions(
                provider,
                store,
                actions,
                config,
                run_id=store.run_id,
                progress=None,
            )
        )
        await asyncio.wait_for(session.all_started.wait(), timeout=1)
        batch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(batch_task, timeout=1)

        assert skipped == 0
        assert session.started == ["item-1", "item-2"]
        assert sorted(session.cancelled) == ["item-1", "item-2"]
        assert sorted(session.cleaned_up) == ["item-1", "item-2"]
        assert provider.open_count == 1
        assert provider.exit_count == 1
        assert provider.exit_after_cleanup is True

    asyncio.run(exercise())


def test_batch_runner_resumes_successes_and_only_processes_new_items(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first_calls: list[str] = []
    first_provider = FakeProcessorProvider("v1", FakeSession(first_calls))
    first_items = (_item("item-1"), _item("item-2"))

    first = run_batch(
        _config(workspace),
        source_provider=FakeSource(first_items),
        processor_provider=first_provider,
    )

    second_calls: list[str] = []
    second_provider = FakeProcessorProvider("v1", FakeSession(second_calls))
    second = run_batch(
        _config(workspace, concurrency=3),
        source_provider=FakeSource((*first_items, _item("item-3"))),
        processor_provider=second_provider,
    )

    assert first_calls == ["item-1", "item-2"]
    assert second_calls == ["item-3"]
    assert first.run_root == second.run_root
    assert (first.succeeded, first.skipped) == (2, 0)
    assert (second.succeeded, second.skipped) == (3, 2)
    assert _read_jsonl(second.output_path) == [
        {"item": "item-1"},
        {"item": "item-2"},
        {"item": "item-3"},
    ]
    history = json.loads(
        second.invocations_path.read_text(encoding="utf-8")
    )
    assert [
        (entry["invocation_id"], entry["status"])
        for entry in history["invocations"]
    ] == [
        ("invocation-000001", "completed"),
        ("invocation-000002", "completed"),
    ]
    assert history["invocations"][1]["summary"]["skipped"] == 2
    assert second.invocation_id == "invocation-000002"
    catalog = json.loads(
        (second.run_root / "catalog.json").read_text(encoding="utf-8")
    )
    assert catalog["source_plugin"] == "unused.module:create"
    assert catalog["items"][0]["dataset_id"] == "demo"
    assert catalog["items"][0]["primary_media"] == "HEAD_RGB"


def test_batch_runner_retries_retryable_errors_within_item_budget(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    progress = []
    provider = FakeProcessorProvider(
        "v1",
        FakeSession(calls, failures_before_success={"item-1": 1}),
    )

    summary = run_batch(
        _config(tmp_path / "workspace", max_attempts=2),
        source_provider=FakeSource((_item("item-1"),)),
        processor_provider=provider,
        progress=progress.append,
    )

    assert calls == ["item-1", "item-1"]
    assert (summary.succeeded, summary.failed, summary.attempted) == (1, 0, 2)
    assert [event.event for event in progress] == [
        "attempt_started",
        "attempt_failed",
        "attempt_started",
        "succeeded",
    ]
    assert progress[1].retrying is True


def test_batch_runner_reprocesses_only_changed_item_fingerprint(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    items = (_item("item-1"), _item("item-2"))
    run_batch(
        _config(workspace),
        source_provider=FakeSource(items),
        processor_provider=FakeProcessorProvider("v1", FakeSession([])),
    )
    calls: list[str] = []

    summary = run_batch(
        _config(workspace),
        source_provider=FakeSource(
            (_item("item-1", "2"), _item("item-2"))
        ),
        processor_provider=FakeProcessorProvider("v1", FakeSession(calls)),
    )

    assert calls == ["item-1"]
    assert (summary.succeeded, summary.skipped) == (2, 1)


def test_batch_runner_processor_identity_creates_a_new_run(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    items = (_item("item-1"),)
    first = run_batch(
        _config(workspace),
        source_provider=FakeSource(items),
        processor_provider=FakeProcessorProvider("v1", FakeSession([])),
    )
    calls: list[str] = []

    second = run_batch(
        _config(workspace),
        source_provider=FakeSource(items),
        processor_provider=FakeProcessorProvider("v2", FakeSession(calls)),
    )

    assert calls == ["item-1"]
    assert first.run_root != second.run_root


def test_batch_runner_version_creates_a_new_run_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    items = (_item("item-1"),)
    provider = FakeProcessorProvider("v1", FakeSession([]))

    first = run_batch(
        _config(workspace),
        source_provider=FakeSource(items),
        processor_provider=provider,
        plan_only=True,
    )
    monkeypatch.setattr(
        runner_module,
        "_RUNNER_VERSION",
        "batch-runner-v-next",
    )
    second = run_batch(
        _config(workspace),
        source_provider=FakeSource(items),
        processor_provider=provider,
        plan_only=True,
    )

    assert first.processor_fingerprint == second.processor_fingerprint
    assert first.run_root != second.run_root


def test_batch_selection_and_plan_only_have_no_output_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    items = tuple(_item(f"item-{index}") for index in range(5))
    calls: list[str] = []
    provider = FakeProcessorProvider("v1", FakeSession(calls))

    summary = run_batch(
        _config(
            workspace,
            start_index=1,
            end_index=4,
            limit=2,
        ),
        source_provider=FakeSource(items),
        processor_provider=provider,
        plan_only=True,
    )

    assert summary.selected == 2
    assert summary.plan_only is True
    assert provider.preflight_count == 1
    assert provider.preflight_items == items
    assert provider.open_count == 0
    assert calls == []
    assert not workspace.exists()


def test_resume_false_refresh_failure_keeps_last_known_good_result(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    first = run_batch(
        _config(workspace),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider("v1", FakeSession([])),
    )
    refresh_calls: list[str] = []

    refresh = run_batch(
        _config(workspace, resume=False, max_attempts=1),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1",
            FakeSession(refresh_calls, permanent_failures={"item-1"}),
        ),
    )

    assert refresh_calls == ["item-1"]
    assert (refresh.succeeded, refresh.failed, refresh.attempted) == (0, 1, 1)
    assert _read_jsonl(refresh.output_path) == [{"item": "item-1"}]
    refresh_failures = _read_jsonl(refresh.failures_path)
    assert len(refresh_failures) == 1
    assert refresh_failures[0]["item_id"] == "item-1"
    assert refresh_failures[0]["preserved_result"] is True
    assert first.output_sha256 == refresh.output_sha256

    resumed_calls: list[str] = []
    resumed = run_batch(
        _config(workspace),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1", FakeSession(resumed_calls)
        ),
    )
    assert resumed_calls == []
    assert (resumed.succeeded, resumed.skipped, resumed.failed) == (1, 1, 0)


def test_increasing_attempt_budget_resumes_a_retryable_failure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    failed_calls: list[str] = []
    first = run_batch(
        _config(workspace, max_attempts=1),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1",
            FakeSession(
                failed_calls,
                failures_before_success={"item-1": 1},
            ),
        ),
    )

    retry_calls: list[str] = []
    second = run_batch(
        _config(workspace, max_attempts=2),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1", FakeSession(retry_calls)
        ),
    )

    assert failed_calls == ["item-1"]
    assert (first.failed, first.attempted) == (1, 1)
    assert retry_calls == ["item-1"]
    assert (second.succeeded, second.failed, second.attempted) == (1, 0, 1)


def test_config_can_reconsider_a_persisted_terminal_failure(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    first = run_batch(
        _config(workspace, max_attempts=1),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1",
            FakeSession([], permanent_failures={"item-1"}),
        ),
    )
    retry_calls: list[str] = []

    second = run_batch(
        _config(
            workspace,
            max_attempts=2,
            retry_terminal_failures=True,
        ),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1", FakeSession(retry_calls)
        ),
    )

    assert first.failed == 1
    assert retry_calls == ["item-1"]
    assert (second.succeeded, second.failed, second.attempted) == (1, 0, 1)


def test_continue_on_error_false_leaves_unvisited_items_pending(
    tmp_path: Path,
) -> None:
    items = (_item("item-1"), _item("item-2"), _item("item-3"))
    calls: list[str] = []

    summary = run_batch(
        _config(
            tmp_path / "workspace",
            max_attempts=1,
            continue_on_error=False,
        ),
        source_provider=FakeSource(items),
        processor_provider=FakeProcessorProvider(
            "v1",
            FakeSession(calls, permanent_failures={"item-1"}),
        ),
    )

    assert calls == ["item-1"]
    assert (
        summary.selected,
        summary.succeeded,
        summary.failed,
        summary.pending,
        summary.attempted,
    ) == (3, 0, 1, 2, 1)


def test_aggregate_projection_excludes_items_removed_from_current_source(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    item_1 = _item("item-1")
    item_2 = _item("item-2")
    run_batch(
        _config(workspace),
        source_provider=FakeSource((item_1, item_2)),
        processor_provider=FakeProcessorProvider("v1", FakeSession([])),
    )

    calls: list[str] = []
    summary = run_batch(
        _config(workspace),
        source_provider=FakeSource((item_2,)),
        processor_provider=FakeProcessorProvider("v1", FakeSession(calls)),
    )

    assert calls == []
    assert (summary.discovered, summary.selected, summary.skipped) == (1, 1, 1)
    assert _read_jsonl(summary.output_path) == [{"item": "item-2"}]


@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    [
        ("item_id", "other-item", "item_id mismatch"),
        (
            "input_fingerprint",
            "sha256:" + "2" * 64,
            "input_fingerprint mismatch",
        ),
        ("attempt_count", 999, "attempt_count mismatch"),
        (
            "artifacts",
            [
                {
                    "path": (
                        "items/other-item/"
                        + "1" * 64
                        + "/attempts/000001/output/artifact.txt"
                    ),
                    "sha256": "sha256:" + "0" * 64,
                    "size_bytes": 1,
                }
            ],
            "artifact identity mismatch",
        ),
    ],
)
def test_resume_rejects_committed_result_with_wrong_identity(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
    message: str,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    provider = FakeProcessorProvider("v1", FakeSession([]))
    first = run_batch(
        _config(workspace),
        source_provider=FakeSource((item,)),
        processor_provider=provider,
    )
    store = BatchStore(workspace, first.run_id)
    payload = store.read_result(item.item_id, item.input_fingerprint)
    assert payload is not None
    payload[changed_field] = changed_value
    store.commit_result(item.item_id, item.input_fingerprint, payload)

    with pytest.raises(ValueError, match=message):
        run_batch(
            _config(workspace),
            source_provider=FakeSource((item,)),
            processor_provider=FakeProcessorProvider(
                "v1", FakeSession([])
            ),
        )


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("attempt_count", "1"),
        ("record", []),
        (
            "artifacts",
            [
                {
                    "path": "../outside.json",
                    "sha256": "sha256:" + "0" * 64,
                    "size_bytes": 1,
                }
            ],
        ),
    ],
)
def test_resume_rejects_malformed_committed_result(
    tmp_path: Path,
    changed_field: str,
    changed_value: object,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    first = run_batch(
        _config(workspace),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider("v1", FakeSession([])),
    )
    store = BatchStore(workspace, first.run_id)
    payload = store.read_result(item.item_id, item.input_fingerprint)
    assert payload is not None
    payload[changed_field] = changed_value
    store.commit_result(item.item_id, item.input_fingerprint, payload)

    with pytest.raises(ValueError, match="invalid committed result"):
        run_batch(
            _config(workspace),
            source_provider=FakeSource((item,)),
            processor_provider=FakeProcessorProvider(
                "v1", FakeSession([])
            ),
        )


def test_increasing_budget_recovers_an_interrupted_attempt(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    item = _item("item-1")
    provider = FakeProcessorProvider("v1", FakeSession([]))
    plan = run_batch(
        _config(workspace, max_attempts=1),
        source_provider=FakeSource((item,)),
        processor_provider=provider,
        plan_only=True,
    )
    store = BatchStore(workspace, plan.run_id)
    store.allocate_attempt(item.item_id, item.input_fingerprint)

    exhausted = run_batch(
        _config(workspace, max_attempts=1),
        source_provider=FakeSource((item,)),
        processor_provider=provider,
    )
    failure = store.read_failure(item.item_id, item.input_fingerprint)
    assert failure is not None
    assert failure["error_type"] == "InterruptedAttempt"
    assert failure["retryable"] is True
    assert (exhausted.succeeded, exhausted.failed, exhausted.attempted) == (
        0,
        1,
        0,
    )

    calls: list[str] = []
    recovered = run_batch(
        _config(workspace, max_attempts=2),
        source_provider=FakeSource((item,)),
        processor_provider=FakeProcessorProvider(
            "v1", FakeSession(calls)
        ),
    )

    assert calls == ["item-1"]
    assert (recovered.succeeded, recovered.failed, recovered.attempted) == (
        1,
        0,
        1,
    )
