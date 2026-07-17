from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pydantic_core import to_jsonable_python

from auto_annotation.artifacts.store import ArtifactReference


_FINGERPRINT_PATTERN = re.compile(
    r"^(?:sha256:)?(?P<digest>[0-9a-fA-F]{64})$"
)


class BatchRunLockedError(RuntimeError):
    """Raised when another process owns a batch run's exclusive lock."""


@dataclass(frozen=True)
class BatchMarker:
    item_id: str
    item_fingerprint: str
    payload: dict[str, Any]
    reference: ArtifactReference


def _validate_component(component: str, *, label: str) -> None:
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
    ):
        raise ValueError(f"unsafe {label}: {component!r}")


def _fingerprint_digest(item_fingerprint: str) -> str:
    match = _FINGERPRINT_PATTERN.fullmatch(item_fingerprint)
    if match is None:
        raise ValueError(
            "item_fingerprint must be a sha256 digest with 64 hex characters"
        )
    return match.group("digest").lower()


def _json_bytes(payload: Any) -> bytes:
    normalized = to_jsonable_python(payload)
    if not isinstance(normalized, dict):
        raise TypeError("batch JSON payload must be an object")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


class BatchStore:
    """Durable, identity-scoped state for one processor configuration.

    A committed ``result.json`` is the per-item completion marker. Attempts and
    failures live below the full item input fingerprint, so changed media never
    inherits an earlier input's retry budget.
    """

    def __init__(self, workspace: Path, run_id: str) -> None:
        _validate_component(run_id, label="run_id")
        self.workspace = Path(os.path.abspath(workspace.expanduser()))
        self.run_id = run_id

    @property
    def run_root(self) -> Path:
        return self.workspace / self.run_id

    @property
    def items_root(self) -> Path:
        return self.run_root / "items"

    def _validate_descendant(self, path: Path) -> None:
        try:
            relative = path.relative_to(self.workspace)
        except ValueError as error:
            raise ValueError(
                f"batch destination escapes workspace: {path}"
            ) from error

        current = self.workspace
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise ValueError(
                    f"batch destination has symlink ancestor: {current}"
                )

        resolved_workspace = self.workspace.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(resolved_workspace):
            raise ValueError(f"batch destination escapes workspace: {path}")

    def _ensure_directory(self, directory: Path) -> None:
        self._validate_descendant(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._validate_descendant(directory)
        if not directory.is_dir():
            raise ValueError(f"batch destination is not a directory: {directory}")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write_json(
        self,
        destination: Path,
        payload: Any,
    ) -> ArtifactReference:
        data = _json_bytes(payload)
        self._ensure_directory(destination.parent)
        self._validate_descendant(destination)
        if destination.is_symlink():
            raise ValueError(
                f"batch JSON destination must not be a symlink: {destination}"
            )

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        return ArtifactReference(
            path=destination,
            sha256=f"sha256:{hashlib.sha256(data).hexdigest()}",
        )

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        self._validate_descendant(path)
        if path.is_symlink():
            raise ValueError(f"batch JSON path must not be a symlink: {path}")
        if not path.exists():
            return None
        if not path.is_file():
            raise ValueError(f"batch JSON path is not a file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"batch JSON payload must be an object: {path}")
        return payload

    def _identity_root(self, item_id: str, item_fingerprint: str) -> Path:
        _validate_component(item_id, label="item_id")
        digest = _fingerprint_digest(item_fingerprint)
        return self.items_root / item_id / digest

    def _marker_path(
        self,
        item_id: str,
        item_fingerprint: str,
        marker: str,
    ) -> Path:
        _validate_component(marker, label="marker")
        return self._identity_root(item_id, item_fingerprint) / f"{marker}.json"

    @contextmanager
    def acquire_run_lock(self) -> Iterator[Path]:
        """Acquire a non-blocking, process-wide exclusive lock for this run."""

        self._ensure_directory(self.run_root)
        lock_path = self.run_root / "run.lock"
        self._validate_descendant(lock_path)
        if lock_path.is_symlink():
            raise ValueError(f"batch run lock must not be a symlink: {lock_path}")

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    f"batch run lock must not be a symlink: {lock_path}"
                ) from error
            raise

        locked = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(
                    f"batch run lock must be a regular file: {lock_path}"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError as error:
                raise BatchRunLockedError(
                    f"batch run is already locked: {self.run_id}"
                ) from error
            self._fsync_directory(self.run_root)
            yield lock_path
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def write_run_json(
        self,
        name: str,
        payload: Any,
    ) -> ArtifactReference:
        _validate_component(name, label="run JSON name")
        return self._atomic_write_json(self.run_root / f"{name}.json", payload)

    def read_run_json(self, name: str) -> dict[str, Any] | None:
        _validate_component(name, label="run JSON name")
        return self._read_json(self.run_root / f"{name}.json")

    def _attempts_root(
        self,
        item_id: str,
        item_fingerprint: str,
    ) -> Path:
        return self._identity_root(item_id, item_fingerprint) / "attempts"

    def _attempt_numbers(
        self,
        item_id: str,
        item_fingerprint: str,
    ) -> tuple[int, ...]:
        attempts_root = self._attempts_root(item_id, item_fingerprint)
        self._validate_descendant(attempts_root)
        if attempts_root.is_symlink():
            raise ValueError(
                f"batch attempts directory must not be a symlink: {attempts_root}"
            )
        if not attempts_root.exists():
            return ()
        if not attempts_root.is_dir():
            raise ValueError(
                f"batch attempts path is not a directory: {attempts_root}"
            )

        numbers: list[int] = []
        for child in attempts_root.iterdir():
            self._validate_descendant(child)
            if child.is_symlink():
                raise ValueError(
                    f"batch attempt path must not be a symlink: {child}"
                )
            if not child.is_dir() or not child.name.isdigit():
                raise ValueError(f"invalid batch attempt entry: {child}")
            number = int(child.name)
            if number <= 0 or child.name != f"{number:06d}":
                raise ValueError(f"invalid batch attempt entry: {child}")
            numbers.append(number)
        return tuple(sorted(numbers))

    def attempt_count(self, item_id: str, item_fingerprint: str) -> int:
        numbers = self._attempt_numbers(item_id, item_fingerprint)
        return numbers[-1] if numbers else 0

    def allocate_attempt(
        self,
        item_id: str,
        item_fingerprint: str,
    ) -> tuple[int, Path]:
        attempts_root = self._attempts_root(item_id, item_fingerprint)
        self._ensure_directory(attempts_root)
        while True:
            numbers = self._attempt_numbers(item_id, item_fingerprint)
            number = (numbers[-1] if numbers else 0) + 1
            attempt_directory = attempts_root / f"{number:06d}"
            self._validate_descendant(attempt_directory)
            try:
                attempt_directory.mkdir()
            except FileExistsError:
                continue
            self._validate_descendant(attempt_directory)
            if not attempt_directory.is_dir():
                raise ValueError(
                    f"batch attempt path is not a directory: {attempt_directory}"
                )
            self._fsync_directory(attempts_root)
            return number, attempt_directory

    @staticmethod
    def _validate_attempt_number(attempt: int) -> None:
        if type(attempt) is not int or attempt <= 0:
            raise ValueError("attempt must be a positive integer")

    def _attempt_directory(
        self,
        item_id: str,
        item_fingerprint: str,
        attempt: int,
    ) -> Path:
        self._validate_attempt_number(attempt)
        return self._attempts_root(item_id, item_fingerprint) / f"{attempt:06d}"

    def write_attempt_json(
        self,
        item_id: str,
        item_fingerprint: str,
        attempt: int,
        name: str,
        payload: Any,
    ) -> ArtifactReference:
        _validate_component(name, label="attempt JSON name")
        attempt_directory = self._attempt_directory(
            item_id,
            item_fingerprint,
            attempt,
        )
        self._validate_descendant(attempt_directory)
        if attempt_directory.is_symlink():
            raise ValueError(
                f"batch attempt path must not be a symlink: {attempt_directory}"
            )
        if not attempt_directory.is_dir():
            raise ValueError(
                f"batch attempt {attempt} was not allocated for {item_id!r}"
            )
        return self._atomic_write_json(
            attempt_directory / f"{name}.json",
            payload,
        )

    def read_attempt_json(
        self,
        item_id: str,
        item_fingerprint: str,
        attempt: int,
        name: str,
    ) -> dict[str, Any] | None:
        _validate_component(name, label="attempt JSON name")
        attempt_directory = self._attempt_directory(
            item_id,
            item_fingerprint,
            attempt,
        )
        return self._read_json(attempt_directory / f"{name}.json")

    def commit_result(
        self,
        item_id: str,
        item_fingerprint: str,
        payload: Any,
    ) -> ArtifactReference:
        result_path = self._marker_path(
            item_id,
            item_fingerprint,
            "result",
        )
        failure_path = self._marker_path(
            item_id,
            item_fingerprint,
            "failure",
        )
        self._validate_descendant(failure_path)
        if failure_path.is_symlink():
            raise ValueError(
                f"batch failure marker must not be a symlink: {failure_path}"
            )
        reference = self._atomic_write_json(result_path, payload)
        if failure_path.exists():
            if not failure_path.is_file():
                raise ValueError(
                    f"batch failure marker is not a file: {failure_path}"
                )
            failure_path.unlink()
            self._fsync_directory(failure_path.parent)
        return reference

    def read_result(
        self,
        item_id: str,
        item_fingerprint: str,
    ) -> dict[str, Any] | None:
        return self._read_json(
            self._marker_path(item_id, item_fingerprint, "result")
        )

    def commit_failure(
        self,
        item_id: str,
        item_fingerprint: str,
        payload: Any,
    ) -> ArtifactReference:
        if self.read_result(item_id, item_fingerprint) is not None:
            raise ValueError(
                f"item {item_id!r} already has a committed result"
            )
        return self._atomic_write_json(
            self._marker_path(item_id, item_fingerprint, "failure"),
            payload,
        )

    def read_failure(
        self,
        item_id: str,
        item_fingerprint: str,
    ) -> dict[str, Any] | None:
        return self._read_json(
            self._marker_path(item_id, item_fingerprint, "failure")
        )

    def _iter_markers(self, marker: str) -> Iterator[BatchMarker]:
        self._validate_descendant(self.items_root)
        if self.items_root.is_symlink():
            raise ValueError(
                f"batch items directory must not be a symlink: {self.items_root}"
            )
        if not self.items_root.exists():
            return
        if not self.items_root.is_dir():
            raise ValueError(
                f"batch items path is not a directory: {self.items_root}"
            )

        for item_path in sorted(self.items_root.iterdir(), key=lambda path: path.name):
            self._validate_descendant(item_path)
            _validate_component(item_path.name, label="stored item_id")
            if item_path.is_symlink() or not item_path.is_dir():
                raise ValueError(f"invalid batch item path: {item_path}")
            for fingerprint_path in sorted(
                item_path.iterdir(), key=lambda path: path.name
            ):
                self._validate_descendant(fingerprint_path)
                if fingerprint_path.is_symlink() or not fingerprint_path.is_dir():
                    raise ValueError(
                        f"invalid batch fingerprint path: {fingerprint_path}"
                    )
                digest = _fingerprint_digest(fingerprint_path.name)
                marker_path = fingerprint_path / f"{marker}.json"
                payload = self._read_json(marker_path)
                if payload is None:
                    continue
                data = marker_path.read_bytes()
                yield BatchMarker(
                    item_id=item_path.name,
                    item_fingerprint=f"sha256:{digest}",
                    payload=payload,
                    reference=ArtifactReference(
                        path=marker_path,
                        sha256=(
                            f"sha256:{hashlib.sha256(data).hexdigest()}"
                        ),
                    ),
                )

    def iter_results(self) -> Iterator[BatchMarker]:
        return self._iter_markers("result")

    def iter_failures(self) -> Iterator[BatchMarker]:
        return self._iter_markers("failure")
