"""Safe, fsync-backed, immutable evaluation directory publication."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import tempfile
from pathlib import Path

from auto_annotation.evaluation.config import EvaluationConfig
from auto_annotation.evaluation.errors import EvaluationError


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def _assert_no_symlink_ancestor(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in [candidate, *candidate.parents]:
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as error:
            raise EvaluationError(f"cannot inspect output path ancestor: {ancestor}") from error
        if stat.S_ISLNK(mode):
            raise EvaluationError(f"output path has a symlink ancestor: {ancestor}")


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError as error:
        raise EvaluationError(f"cannot compare output path aliases: {left}, {right}") from error


def validate_output_topology(config: EvaluationConfig, evaluation_id: str | None = None) -> None:
    root = config.output.root
    _assert_no_symlink_ancestor(root)
    input_paths = [
        config.inputs.gold_set_path,
        config.inputs.annotations_dir,
        config.inputs.prediction_path,
    ]
    if config.source_path is not None:
        input_paths.append(config.source_path)
    try:
        root_normalized = root.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise EvaluationError(f"cannot normalize evaluation output root: {root}") from error
    target = None if evaluation_id is None else root / evaluation_id
    for input_path in input_paths:
        try:
            normalized_input = input_path.resolve(strict=False)
        except (OSError, RuntimeError) as error:
            raise EvaluationError(f"cannot normalize evaluator input: {input_path}") from error
        if root_normalized == normalized_input or _same_existing_file(root, input_path):
            raise EvaluationError(f"output root aliases an evaluator input: {input_path}")
        if input_path.is_dir() and root_normalized.is_relative_to(normalized_input):
            raise EvaluationError(f"output root is inside an evaluator input directory: {input_path}")
        if target is not None and (
            target.resolve(strict=False) == normalized_input or _same_existing_file(target, input_path)
        ):
            raise EvaluationError(f"evaluation output aliases an evaluator input: {input_path}")
    if target is not None:
        if target.parent.resolve(strict=False) != root_normalized:
            raise EvaluationError("evaluation output path escapes output root")
        _assert_no_symlink_ancestor(target)


def _verify_existing(target: Path, files: dict[str, bytes]) -> None:
    if not target.is_dir() or target.is_symlink():
        raise EvaluationError(f"existing evaluation target is not a safe directory: {target}")
    actual = {path.name for path in target.iterdir()}
    expected = set(files)
    if actual != expected:
        raise EvaluationError(
            f"existing evaluation directory file set differs: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for name, expected_bytes in files.items():
        path = target / name
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise EvaluationError(f"existing evaluation member is not a regular file: {path}")
        try:
            actual_bytes = path.read_bytes()
        except OSError as error:
            raise EvaluationError(f"cannot read existing evaluation member: {path}") from error
        if actual_bytes != expected_bytes:
            raise EvaluationError(f"existing evaluation member differs: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing any target inode."""
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise EvaluationError(
            "atomic no-replace directory publication requires Linux renameat2"
        ) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == 0:
            error_number = errno.EIO
        raise OSError(
            error_number,
            os.strerror(error_number),
            str(target),
        )


def publish_evaluation(config: EvaluationConfig, evaluation_id: str, files: dict[str, bytes]) -> Path:
    validate_output_topology(config, evaluation_id)
    root = config.output.root
    target = root / evaluation_id
    if target.exists():
        _verify_existing(target, files)
        return target
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise EvaluationError(f"cannot create evaluation output root: {root}") from error
    validate_output_topology(config, evaluation_id)
    staging = Path(tempfile.mkdtemp(prefix=f".{evaluation_id}.", dir=root))
    published = False
    try:
        for name, content in sorted(files.items()):
            path = staging / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                raise
        _fsync_directory(staging)
        if target.exists():
            _verify_existing(target, files)
            return target
        try:
            _rename_directory_no_replace(staging, target)
            published = True
        except FileExistsError:
            _verify_existing(target, files)
            return target
        _fsync_directory(root)
        return target
    except EvaluationError:
        raise
    except OSError as error:
        raise EvaluationError(f"cannot publish evaluation directory: {target}") from error
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
