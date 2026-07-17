from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError, to_jsonable_python

from auto_annotation.batch.models import (
    BatchConfig,
    BatchFailure,
    BatchItem,
    ProcessorContext,
    ProcessorResult,
)
from auto_annotation.batch.plugins import (
    BatchItemError,
    ProcessorProvider,
    ProcessorSession,
    SourceProvider,
    load_processor_provider,
    load_source_provider,
)
from auto_annotation.batch.store import BatchStore
from auto_annotation.exporters.jsonl import write_jsonl


_RUNNER_VERSION = "batch-runner-v2"
_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class BatchRunSummary(BaseModel):
    """Durable counters and aggregate paths for one batch invocation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    invocation_id: str | None = None
    processor_fingerprint: str
    run_root: Path
    discovered: int = Field(ge=0)
    selected: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    skipped: int = Field(ge=0)
    failed: int = Field(ge=0)
    pending: int = Field(ge=0)
    attempted: int = Field(ge=0)
    output_path: Path
    output_index_path: Path
    failures_path: Path
    summary_path: Path
    invocations_path: Path
    output_sha256: str | None = None
    output_index_sha256: str | None = None
    failures_sha256: str | None = None
    plan_only: bool


class BatchProgress(BaseModel):
    """Processor-independent progress event emitted by an actual batch run."""

    model_config = ConfigDict(extra="forbid")

    event: Literal[
        "skipped",
        "attempt_started",
        "attempt_failed",
        "succeeded",
        "failed",
    ]
    item_id: str
    position: int = Field(ge=1)
    total: int = Field(ge=1)
    attempt: int | None = Field(default=None, ge=1)
    retrying: bool = False
    message: str | None = None


ProgressCallback = Callable[[BatchProgress], None]


class _InvocationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(pattern=r"^invocation-[0-9]{6}$")
    status: Literal["running", "completed"]
    started_at: str = Field(min_length=1)
    finished_at: str | None = None
    config_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    config_path: str | None = None
    config_file_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    source_plugin: str = Field(min_length=1)
    source_config_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    processor_plugin: str = Field(min_length=1)
    processor_config_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    execution: dict[str, Any]
    selected_item_ids: tuple[str, ...]
    summary: dict[str, Any] | None = None


class _CommittedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, strict=True)
    sha256: str = Field(pattern=_SHA256_PATTERN, strict=True)
    size_bytes: int = Field(ge=0, strict=True)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if (
            "\x00" in value
            or "\\" in value
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or posix_path in {PurePosixPath("."), PurePosixPath("..")}
            or ".." in posix_path.parts
            or posix_path.as_posix() != value
        ):
            raise ValueError(
                "committed artifact path must be a normalized relative path"
            )
        return value


class _CommittedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, strict=True)
    input_fingerprint: str = Field(pattern=_SHA256_PATTERN, strict=True)
    attempt_count: int = Field(ge=1, strict=True)
    record: dict[str, Any]
    artifacts: tuple[_CommittedArtifact, ...]

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> _CommittedResult:
        paths = [artifact.path for artifact in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("committed result contains duplicate artifacts")
        return self


@dataclass(frozen=True)
class _SelectedAction:
    item: BatchItem
    remaining_attempts: int
    terminal_failure: bool = False


def _emit_progress(
    callback: ProgressCallback | None,
    **values: Any,
) -> None:
    if callback is not None:
        callback(BatchProgress.model_validate(values))


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        to_jsonable_python(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invocation_config_fingerprint(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _load_invocations(store: BatchStore) -> list[_InvocationRecord]:
    payload = store.read_run_json("invocations")
    if payload is None:
        return []
    if set(payload) != {"version", "invocations"} or payload["version"] != 1:
        raise ValueError("invalid batch invocation history")
    raw_invocations = payload["invocations"]
    if not isinstance(raw_invocations, list):
        raise ValueError("invalid batch invocation history")
    try:
        invocations = [
            _InvocationRecord.model_validate(raw)
            for raw in raw_invocations
        ]
    except ValidationError as error:
        raise ValueError("invalid batch invocation history") from error
    expected_ids = [
        f"invocation-{index:06d}"
        for index in range(1, len(invocations) + 1)
    ]
    if [record.invocation_id for record in invocations] != expected_ids:
        raise ValueError("invalid batch invocation history sequence")
    return invocations


def _write_invocations(
    store: BatchStore,
    invocations: Sequence[_InvocationRecord],
) -> None:
    store.write_run_json(
        "invocations",
        {
            "version": 1,
            "invocations": [
                record.model_dump(mode="json") for record in invocations
            ],
        },
    )


def _start_invocation(
    store: BatchStore,
    config: BatchConfig,
    selected: tuple[BatchItem, ...],
) -> tuple[list[_InvocationRecord], _InvocationRecord]:
    invocations = _load_invocations(store)
    config_payload = config.model_dump(mode="json")
    source_payload = config.source.model_dump(mode="json")
    processor_payload = config.processor.model_dump(mode="json")
    config_path = config.source_path
    invocation = _InvocationRecord(
        invocation_id=f"invocation-{len(invocations) + 1:06d}",
        status="running",
        started_at=_utc_now(),
        config_fingerprint=_invocation_config_fingerprint(config_payload),
        config_path=None if config_path is None else str(config_path),
        config_file_sha256=(
            None if config_path is None else _sha256_file(config_path)
        ),
        source_plugin=config.source.plugin,
        source_config_fingerprint=_invocation_config_fingerprint(
            source_payload
        ),
        processor_plugin=config.processor.plugin,
        processor_config_fingerprint=_invocation_config_fingerprint(
            processor_payload
        ),
        execution=config.execution.model_dump(mode="json"),
        selected_item_ids=tuple(item.item_id for item in selected),
    )
    invocations.append(invocation)
    _write_invocations(store, invocations)
    return invocations, invocation


def _complete_invocation(
    store: BatchStore,
    invocations: list[_InvocationRecord],
    invocation: _InvocationRecord,
    summary: BatchRunSummary,
) -> None:
    completed = invocation.model_copy(
        update={
            "status": "completed",
            "finished_at": _utc_now(),
            "summary": summary.model_dump(mode="json"),
        }
    )
    invocations[-1] = completed
    _write_invocations(store, invocations)


def _processor_identity(
    processor_provider: ProcessorProvider,
) -> tuple[dict[str, str], str, str]:
    raw_identity = processor_provider.identity
    if not isinstance(raw_identity, Mapping):
        raise TypeError("processor identity must be a mapping of strings")

    identity: dict[str, str] = {}
    for key, value in raw_identity.items():
        if not isinstance(key, str) or not key:
            raise TypeError("processor identity keys must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise TypeError(
                "processor identity values must be non-empty strings"
            )
        identity[key] = value
    if not identity:
        raise ValueError("processor identity must not be empty")

    normalized_identity = dict(sorted(identity.items()))
    processor_fingerprint = _sha256_bytes(
        _canonical_json_bytes(normalized_identity)
    )
    run_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "runner_version": _RUNNER_VERSION,
                "processor_identity": normalized_identity,
            }
        )
    )
    run_id = f"run-{run_fingerprint.removeprefix('sha256:')}"
    return normalized_identity, processor_fingerprint, run_id


def _discover_items(source_provider: SourceProvider) -> tuple[BatchItem, ...]:
    discovered = source_provider.discover()
    if not isinstance(discovered, Sequence) or isinstance(
        discovered, (str, bytes, bytearray)
    ):
        raise TypeError("batch source discover() must return a sequence")

    items: list[BatchItem] = []
    item_ids: set[str] = set()
    for raw_item in discovered:
        item = BatchItem.model_validate(raw_item)
        if item.item_id in item_ids:
            raise ValueError(f"duplicate batch item_id: {item.item_id!r}")
        try:
            _canonical_json_bytes(item.model_dump(mode="json"))
        except (PydanticSerializationError, TypeError, ValueError) as error:
            raise ValueError(
                f"batch item is not JSON serializable: {item.item_id!r}"
            ) from error
        item_ids.add(item.item_id)
        items.append(item)
    return tuple(items)


def _preflight_processor(
    processor_provider: ProcessorProvider,
    items: tuple[BatchItem, ...],
) -> None:
    """Run an optional, side-effect-free source/processor compatibility check."""

    preflight = getattr(processor_provider, "preflight", None)
    if preflight is None:
        return
    if not callable(preflight):
        raise TypeError("processor preflight must be callable")
    result = preflight(items)
    if result is not None:
        raise TypeError("processor preflight must return None")


def _select_items(
    items: tuple[BatchItem, ...],
    config: BatchConfig,
) -> tuple[BatchItem, ...]:
    execution = config.execution
    selected = items[execution.start_index : execution.end_index]
    if execution.limit is not None:
        selected = selected[: execution.limit]
    return selected


def _failure_from_error(
    item: BatchItem,
    *,
    attempt_count: int,
    error: Exception,
) -> BatchFailure:
    retryable = isinstance(error, BatchItemError) and error.retryable
    error_type = type(error).__name__
    message = str(error).strip() or error_type
    return BatchFailure(
        item_id=item.item_id,
        attempt_count=attempt_count,
        error_type=error_type,
        message=message,
        retryable=retryable,
    )


def _stored_failure(
    store: BatchStore,
    item: BatchItem,
) -> BatchFailure | None:
    payload = store.read_failure(item.item_id, item.input_fingerprint)
    if payload is None:
        return None
    try:
        failure = BatchFailure.model_validate(payload)
    except ValidationError as error:
        raise ValueError(
            f"invalid committed failure for {item.item_id!r}"
        ) from error
    if failure.item_id != item.item_id:
        raise ValueError(
            f"committed failure item_id mismatch for {item.item_id!r}"
        )
    return failure


def _stored_result(
    store: BatchStore,
    item: BatchItem,
) -> _CommittedResult | None:
    payload = store.read_result(item.item_id, item.input_fingerprint)
    if payload is None:
        return None
    try:
        result = _CommittedResult.model_validate(payload)
        _canonical_json_bytes(result.record)
    except (TypeError, ValueError, ValidationError) as error:
        raise ValueError(
            f"invalid committed result for {item.item_id!r}"
        ) from error

    if result.item_id != item.item_id:
        raise ValueError(
            f"committed result item_id mismatch for {item.item_id!r}"
        )
    if result.input_fingerprint != item.input_fingerprint:
        raise ValueError(
            f"committed result input_fingerprint mismatch for "
            f"{item.item_id!r}"
        )

    allocated_attempts = store.attempt_count(
        item.item_id, item.input_fingerprint
    )
    if result.attempt_count > allocated_attempts:
        raise ValueError(
            f"committed result attempt_count mismatch for {item.item_id!r}"
        )

    expected_artifact_root = (
        PurePosixPath("items")
        / item.item_id
        / item.input_fingerprint.removeprefix("sha256:")
        / "attempts"
        / f"{result.attempt_count:06d}"
        / "output"
    )
    for artifact in result.artifacts:
        if not PurePosixPath(artifact.path).is_relative_to(
            expected_artifact_root
        ):
            raise ValueError(
                f"committed result artifact identity mismatch for "
                f"{item.item_id!r}"
            )
    return result


def _prepare_actions(
    store: BatchStore,
    selected: tuple[BatchItem, ...],
    config: BatchConfig,
    *,
    write_interrupted_failures: bool,
) -> tuple[tuple[_SelectedAction, ...], int]:
    actions: list[_SelectedAction] = []
    skipped = 0
    execution = config.execution

    for item in selected:
        if not execution.resume:
            actions.append(
                _SelectedAction(
                    item=item,
                    remaining_attempts=execution.max_attempts,
                )
            )
            continue

        result = _stored_result(store, item)
        if result is not None:
            skipped += 1
            actions.append(_SelectedAction(item=item, remaining_attempts=0))
            continue

        attempt_count = store.attempt_count(
            item.item_id, item.input_fingerprint
        )
        failure = _stored_failure(store, item)
        if failure is not None and failure.attempt_count > attempt_count:
            raise ValueError(
                f"committed failure attempt count mismatch for "
                f"{item.item_id!r}"
            )

        remaining = max(0, execution.max_attempts - attempt_count)
        if (
            failure is not None
            and not failure.retryable
            and failure.error_type != "InterruptedAttempt"
            and not execution.retry_terminal_failures
        ):
            actions.append(
                _SelectedAction(
                    item=item,
                    remaining_attempts=0,
                    terminal_failure=True,
                )
            )
            continue

        if remaining == 0:
            # An attempt directory without a same-number failure marker means
            # the process stopped before the per-item commit boundary.
            if failure is None or failure.attempt_count != attempt_count:
                interrupted = BatchFailure(
                    item_id=item.item_id,
                    attempt_count=attempt_count,
                    error_type="InterruptedAttempt",
                    message=(
                        "attempt budget was exhausted before result commit"
                    ),
                    retryable=True,
                )
                if write_interrupted_failures:
                    store.commit_failure(
                        item.item_id,
                        item.input_fingerprint,
                        interrupted.model_dump(mode="json"),
                    )
            actions.append(
                _SelectedAction(
                    item=item,
                    remaining_attempts=0,
                    terminal_failure=True,
                )
            )
            continue

        actions.append(
            _SelectedAction(item=item, remaining_attempts=remaining)
        )

    return tuple(actions), skipped


def _validate_output_file(path: Path, *, output_dir: Path) -> Path:
    if not path.is_absolute():
        raise BatchItemError(
            f"processor artifact path must be absolute: {path}",
            retryable=False,
        )

    absolute_path = Path(os.path.abspath(path))
    absolute_output = Path(os.path.abspath(output_dir))
    if absolute_output.is_symlink() or not absolute_output.is_dir():
        raise BatchItemError(
            f"processor output directory is not a safe directory: "
            f"{output_dir}",
            retryable=False,
        )
    try:
        relative = absolute_path.relative_to(absolute_output)
    except ValueError as error:
        raise BatchItemError(
            f"processor artifact escapes output directory: {path}",
            retryable=False,
        ) from error
    if not relative.parts:
        raise BatchItemError(
            f"processor artifact must be a file: {path}",
            retryable=False,
        )

    current = absolute_output
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise BatchItemError(
                f"processor artifact has symlink component: {current}",
                retryable=False,
            )

    try:
        file_stat = absolute_path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise BatchItemError(
            f"processor artifact does not exist: {path}",
            retryable=False,
        ) from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise BatchItemError(
            f"processor artifact is not a regular file: {path}",
            retryable=False,
        )

    resolved_output = absolute_output.resolve(strict=True)
    resolved_path = absolute_path.resolve(strict=True)
    if not resolved_path.is_relative_to(resolved_output):
        raise BatchItemError(
            f"processor artifact escapes output directory: {path}",
            retryable=False,
        )
    return absolute_path


def _normalize_result(
    raw_result: object,
    *,
    output_dir: Path,
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        result = ProcessorResult.model_validate(raw_result)
        record = to_jsonable_python(result.record)
        if not isinstance(record, dict):
            raise TypeError("processor record must be a JSON object")
        _canonical_json_bytes(record)
    except (TypeError, ValueError, ValidationError) as error:
        raise BatchItemError(
            f"invalid processor result: {error}",
            retryable=False,
        ) from error

    seen: set[Path] = set()
    artifacts: list[dict[str, Any]] = []
    for raw_path in result.artifacts:
        path = _validate_output_file(raw_path, output_dir=output_dir)
        if path in seen:
            raise BatchItemError(
                f"processor returned duplicate artifact path: {path}",
                retryable=False,
            )
        seen.add(path)
        artifacts.append(
            {
                "path": path.relative_to(run_root).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat(follow_symlinks=False).st_size,
            }
        )
    artifacts.sort(key=lambda artifact: str(artifact["path"]))
    return record, artifacts


async def _process_one(
    session: ProcessorSession,
    store: BatchStore,
    action: _SelectedAction,
    config: BatchConfig,
    *,
    run_id: str,
    position: int,
    total: int,
    progress: ProgressCallback | None,
) -> tuple[int, bool, BatchFailure | None]:
    item = action.item
    attempted = 0

    for local_attempt in range(action.remaining_attempts):
        attempt, attempt_dir = store.allocate_attempt(
            item.item_id, item.input_fingerprint
        )
        output_dir = attempt_dir / "output"
        output_dir.mkdir()
        attempted += 1
        context = ProcessorContext(
            run_id=run_id,
            workspace=store.workspace,
            attempt_dir=attempt_dir,
            output_dir=output_dir,
            attempt=attempt,
        )
        _emit_progress(
            progress,
            event="attempt_started",
            item_id=item.item_id,
            position=position,
            total=total,
            attempt=attempt,
        )

        try:
            raw_result = await session.process(item, context)
            record, artifacts = _normalize_result(
                raw_result,
                output_dir=output_dir,
                run_root=store.run_root,
            )
            committed_result = _CommittedResult(
                item_id=item.item_id,
                input_fingerprint=item.input_fingerprint,
                attempt_count=attempt,
                record=record,
                artifacts=tuple(artifacts),
            )
        except Exception as error:
            failure = _failure_from_error(
                item,
                attempt_count=attempt,
                error=error,
            )
            preserved_result = _stored_result(store, item) is not None
            if preserved_result:
                failure = failure.model_copy(
                    update={"preserved_result": True}
                )
            store.write_attempt_json(
                item.item_id,
                item.input_fingerprint,
                attempt,
                "failure",
                failure.model_dump(mode="json"),
            )
            will_retry = (
                failure.retryable
                and local_attempt + 1 < action.remaining_attempts
            )
            _emit_progress(
                progress,
                event="attempt_failed",
                item_id=item.item_id,
                position=position,
                total=total,
                attempt=attempt,
                retrying=will_retry,
                message=failure.message,
            )
            if will_retry:
                if config.execution.retry_backoff_s:
                    await asyncio.sleep(config.execution.retry_backoff_s)
                continue
            # A previous committed result is a last-known-good marker. A forced
            # refresh must not replace it with a failed, partial attempt.
            if not preserved_result:
                store.commit_failure(
                    item.item_id,
                    item.input_fingerprint,
                    failure.model_dump(mode="json"),
                )
            return attempted, False, failure

        result_payload = committed_result.model_dump(mode="json")
        store.write_attempt_json(
            item.item_id,
            item.input_fingerprint,
            attempt,
            "result",
            result_payload,
        )
        # The marker is deliberately the last per-item write. Resume only treats
        # an item as complete after this atomic commit succeeds.
        store.commit_result(
            item.item_id,
            item.input_fingerprint,
            result_payload,
        )
        _emit_progress(
            progress,
            event="succeeded",
            item_id=item.item_id,
            position=position,
            total=total,
            attempt=attempt,
        )
        return attempted, True, None

    return attempted, False, None


async def _process_actions(
    processor_provider: ProcessorProvider,
    store: BatchStore,
    actions: tuple[_SelectedAction, ...],
    config: BatchConfig,
    *,
    run_id: str,
    progress: ProgressCallback | None,
) -> tuple[int, dict[str, BatchFailure]]:
    attempted = 0
    invocation_failures: dict[str, BatchFailure] = {}

    scheduled_actions = actions
    if not config.execution.continue_on_error:
        terminal_index = next(
            (
                index
                for index, action in enumerate(actions)
                if action.terminal_failure
            ),
            len(actions),
        )
        # The first persisted terminal failure is itself observable, but items
        # after it must remain pending under fail-fast semantics.
        scheduled_actions = actions[: terminal_index + 1]

    has_processable = any(
        action.remaining_attempts > 0 and not action.terminal_failure
        for action in scheduled_actions
    )
    if not has_processable:
        for position, action in enumerate(scheduled_actions, start=1):
            if action.terminal_failure:
                failure = _stored_failure(store, action.item)
                _emit_progress(
                    progress,
                    event="failed",
                    item_id=action.item.item_id,
                    position=position,
                    total=len(actions),
                    attempt=(
                        None if failure is None else failure.attempt_count
                    ),
                    message=(
                        "terminal failure"
                        if failure is None
                        else failure.message
                    ),
                )
                if not config.execution.continue_on_error:
                    break
            elif action.remaining_attempts == 0:
                _emit_progress(
                    progress,
                    event="skipped",
                    item_id=action.item.item_id,
                    position=position,
                    total=len(actions),
                )
        return attempted, invocation_failures

    next_action_index = 0
    stop_assigning = False
    assignment_lock = asyncio.Lock()
    fatal_errors: list[Exception] = []

    async def claim_action() -> tuple[int, _SelectedAction] | None:
        nonlocal next_action_index, stop_assigning
        async with assignment_lock:
            if (
                stop_assigning
                or next_action_index >= len(scheduled_actions)
            ):
                return None
            index = next_action_index
            next_action_index += 1
            action = scheduled_actions[index]
            if (
                action.terminal_failure
                and not config.execution.continue_on_error
            ):
                # Set this while still holding the assignment lock so no worker
                # can claim an item after a persisted fail-fast boundary.
                stop_assigning = True
            return index + 1, action

    async def stop_after_failure() -> None:
        nonlocal stop_assigning
        async with assignment_lock:
            stop_assigning = True

    async with processor_provider.open() as session:
        async def worker() -> None:
            nonlocal attempted

            while True:
                claimed = await claim_action()
                if claimed is None:
                    return
                position, action = claimed

                if action.terminal_failure:
                    failure = _stored_failure(store, action.item)
                    _emit_progress(
                        progress,
                        event="failed",
                        item_id=action.item.item_id,
                        position=position,
                        total=len(actions),
                        attempt=(
                            None if failure is None else failure.attempt_count
                        ),
                        message=(
                            "terminal failure"
                            if failure is None
                            else failure.message
                        ),
                    )
                    if not config.execution.continue_on_error:
                        return
                    continue

                if action.remaining_attempts == 0:
                    _emit_progress(
                        progress,
                        event="skipped",
                        item_id=action.item.item_id,
                        position=position,
                        total=len(actions),
                    )
                    continue

                try:
                    item_attempts, succeeded, failure = await _process_one(
                        session,
                        store,
                        action,
                        config,
                        run_id=run_id,
                        position=position,
                        total=len(actions),
                        progress=progress,
                    )
                except Exception as error:
                    # Do not cancel other workers: a processor may be backed by
                    # a thread or child process that cannot be safely cancelled
                    # at an arbitrary await boundary. Stop refilling the pool,
                    # let in-flight items reach their commit boundary, then
                    # propagate the first fatal runner error.
                    fatal_errors.append(error)
                    await stop_after_failure()
                    return

                attempted += item_attempts
                if succeeded:
                    continue
                if failure is None:
                    fatal_errors.append(
                        RuntimeError(
                            "failed batch action did not return failure details"
                        )
                    )
                    await stop_after_failure()
                    return
                invocation_failures[action.item.item_id] = failure
                if not config.execution.continue_on_error:
                    await stop_after_failure()
                    return

        async def guarded_worker() -> None:
            try:
                await worker()
            except Exception as error:
                # Terminal-marker reads and progress callbacks happen outside
                # ``_process_one``. Treat failures there like any other fatal
                # runner error while still draining already-started workers.
                fatal_errors.append(error)
                await stop_after_failure()

        worker_count = min(
            config.execution.concurrency,
            sum(
                action.remaining_attempts > 0
                and not action.terminal_failure
                for action in scheduled_actions
            ),
        )
        workers = [
            asyncio.create_task(guarded_worker())
            for _ in range(worker_count)
        ]
        worker_group = asyncio.gather(*workers)
        try:
            await asyncio.shield(worker_group)
        except BaseException:
            # External cancellation (including Ctrl-C through asyncio.run) must
            # reach every in-flight processor call, then be fully drained before
            # the shared session and run lock are released.
            await stop_after_failure()
            for worker_task in workers:
                worker_task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await asyncio.gather(worker_group, return_exceptions=True)
            raise

    if fatal_errors:
        raise fatal_errors[0]
    return attempted, invocation_failures


def _current_aggregates(
    store: BatchStore,
    items: tuple[BatchItem, ...],
    *,
    invocation_failures: Mapping[str, BatchFailure],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    records: list[dict[str, Any]] = []
    output_index: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda candidate: candidate.item_id):
        result = _stored_result(store, item)
        if result is not None:
            records.append(result.record)
            output_index.append(
                {
                    "line_number": len(records),
                    "item_id": item.item_id,
                    "dataset_id": item.dataset_id,
                    "input_fingerprint": item.input_fingerprint,
                    "record_sha256": _sha256_bytes(
                        _canonical_json_bytes(result.record)
                    ),
                }
            )
        failure = invocation_failures.get(item.item_id)
        if failure is None:
            failure = _stored_failure(store, item)
        if failure is not None:
            failures.append(failure.model_dump(mode="json"))
    return records, output_index, failures


def _selected_counts(
    store: BatchStore,
    selected: tuple[BatchItem, ...],
    *,
    invocation_failures: Mapping[str, BatchFailure],
) -> tuple[int, int, int]:
    succeeded = 0
    failed = 0
    for item in selected:
        if item.item_id in invocation_failures:
            # ``resume=False`` is a refresh request. If that refresh fails, the
            # summary reports the failed invocation even though the aggregate
            # deliberately retains an older, committed last-known-good result.
            failed += 1
        elif _stored_result(store, item) is not None:
            succeeded += 1
        elif _stored_failure(store, item) is not None:
            failed += 1
    return succeeded, failed, len(selected) - succeeded - failed


def _summary(
    *,
    store: BatchStore,
    invocation_id: str | None,
    processor_fingerprint: str,
    discovered: int,
    selected: int,
    succeeded: int,
    skipped: int,
    failed: int,
    pending: int,
    attempted: int,
    output_sha256: str | None,
    output_index_sha256: str | None,
    failures_sha256: str | None,
    plan_only: bool,
) -> BatchRunSummary:
    return BatchRunSummary(
        run_id=store.run_id,
        invocation_id=invocation_id,
        processor_fingerprint=processor_fingerprint,
        run_root=store.run_root,
        discovered=discovered,
        selected=selected,
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        pending=pending,
        attempted=attempted,
        output_path=store.run_root / "output.jsonl",
        output_index_path=store.run_root / "output_index.jsonl",
        failures_path=store.run_root / "failures.jsonl",
        summary_path=store.run_root / "summary.json",
        invocations_path=store.run_root / "invocations.json",
        output_sha256=output_sha256,
        output_index_sha256=output_index_sha256,
        failures_sha256=failures_sha256,
        plan_only=plan_only,
    )


def run_batch(
    config: BatchConfig,
    *,
    source_provider: SourceProvider | None = None,
    processor_provider: ProcessorProvider | None = None,
    plan_only: bool = False,
    progress: ProgressCallback | None = None,
) -> BatchRunSummary:
    """Run a config-selected, bounded-concurrency batch against a processor.

    The processor identity and runner contract own the durable run namespace.
    Input fingerprints own per-item state within that namespace, so execution
    and source selection can change without teaching this runner anything about
    a dataset or processor.
    """

    effective_source = source_provider
    if effective_source is None:
        effective_source = load_source_provider(
            config.source,
            config_dir=config.config_dir,
        )
    effective_processor = processor_provider
    if effective_processor is None:
        effective_processor = load_processor_provider(
            config.processor,
            config_dir=config.config_dir,
        )
    identity, processor_fingerprint, run_id = _processor_identity(
        effective_processor
    )
    items = _discover_items(effective_source)
    _preflight_processor(effective_processor, items)
    selected = _select_items(items, config)
    store = BatchStore(config.output.workspace, run_id)

    if plan_only:
        actions, skipped = _prepare_actions(
            store,
            selected,
            config,
            write_interrupted_failures=False,
        )
        succeeded = skipped
        failed = sum(action.terminal_failure for action in actions)
        pending = len(selected) - succeeded - failed
        return _summary(
            store=store,
            invocation_id=None,
            processor_fingerprint=processor_fingerprint,
            discovered=len(items),
            selected=len(selected),
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            pending=pending,
            attempted=0,
            output_sha256=None,
            output_index_sha256=None,
            failures_sha256=None,
            plan_only=True,
        )

    with store.acquire_run_lock():
        store.write_run_json(
            "identity",
            {
                "runner_version": _RUNNER_VERSION,
                "run_id": run_id,
                "processor_fingerprint": processor_fingerprint,
                "processor_identity": identity,
            },
        )
        store.write_run_json(
            "catalog",
            {
                "source_plugin": config.source.plugin,
                "items": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        items, key=lambda candidate: candidate.item_id
                    )
                ]
            },
        )

        invocations, invocation = _start_invocation(
            store,
            config,
            selected,
        )

        actions, skipped = _prepare_actions(
            store,
            selected,
            config,
            write_interrupted_failures=True,
        )
        attempted, invocation_failures = asyncio.run(
            _process_actions(
                effective_processor,
                store,
                actions,
                config,
                run_id=run_id,
                progress=progress,
            )
        )

        records, output_index, failures = _current_aggregates(
            store,
            items,
            invocation_failures=invocation_failures,
        )
        output_path = store.run_root / "output.jsonl"
        output_index_path = store.run_root / "output_index.jsonl"
        failures_path = store.run_root / "failures.jsonl"
        write_jsonl(output_path, records, sort_key=None)
        write_jsonl(output_index_path, output_index, sort_key=None)
        write_jsonl(failures_path, failures, sort_key=None)
        output_sha256 = _sha256_file(output_path)
        output_index_sha256 = _sha256_file(output_index_path)
        failures_sha256 = _sha256_file(failures_path)

        succeeded, failed, pending = _selected_counts(
            store,
            selected,
            invocation_failures=invocation_failures,
        )
        summary = _summary(
            store=store,
            invocation_id=invocation.invocation_id,
            processor_fingerprint=processor_fingerprint,
            discovered=len(items),
            selected=len(selected),
            succeeded=succeeded,
            skipped=skipped,
            failed=failed,
            pending=pending,
            attempted=attempted,
            output_sha256=output_sha256,
            output_index_sha256=output_index_sha256,
            failures_sha256=failures_sha256,
            plan_only=False,
        )
        store.write_run_json("summary", summary.model_dump(mode="json"))
        _complete_invocation(store, invocations, invocation, summary)
        return summary
