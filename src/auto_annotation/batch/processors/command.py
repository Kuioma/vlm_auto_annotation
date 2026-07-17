from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import stat
import string
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any, AsyncIterator, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from auto_annotation.batch.models import (
    BatchItem,
    ProcessorContext,
    ProcessorResult,
)
from auto_annotation.batch.plugins import BatchItemError, ProcessorPlugin


COMMAND_PROCESSOR_VERSION = "command-processor-v3"
_PROCESS_GROUP_TERM_GRACE_S = 0.25
_MEDIA_PLACEHOLDER = re.compile(r"media_[A-Za-z0-9_]+")
_FIXED_PLACEHOLDERS = frozenset(
    {"item_id", "input_path", "item_json", "output_dir", "workspace"}
)
_ITEM_SNAPSHOT_FIELDS = frozenset(
    {
        "item_id",
        "dataset_id",
        "media",
        "primary_media",
        "context",
        "metadata",
        "input_fingerprint",
    }
)
_FORMATTER = string.Formatter()


def _template_fields(template: str) -> tuple[str, ...]:
    try:
        parsed = tuple(_FORMATTER.parse(template))
    except ValueError as error:
        raise ValueError(f"invalid command template: {template!r}") from error

    fields: list[str] = []
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if format_spec or conversion is not None:
            raise ValueError(
                "command placeholders do not support conversion or format "
                f"specifiers: {template!r}"
            )
        if (
            field_name not in _FIXED_PLACEHOLDERS
            and _MEDIA_PLACEHOLDER.fullmatch(field_name) is None
        ):
            raise ValueError(
                f"unsupported command placeholder {field_name!r}: {template!r}"
            )
        fields.append(field_name)
    return tuple(fields)


def _validate_template(template: str, *, label: str) -> str:
    if not template or "\x00" in template:
        raise ValueError(f"{label} must be non-empty and contain no NUL")
    _template_fields(template)
    return template


def _validate_relative_result_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    raw = str(path)
    windows_path = PureWindowsPath(raw)
    if (
        not raw
        or "\x00" in raw
        or path.is_absolute()
        or windows_path.is_absolute()
        or path in {Path("."), Path("..")}
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(
            "result_path must be a safe relative path below processor output_dir"
        )
    return path


class CommandProcessorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    semantic_version: str = Field(min_length=1)
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str | Path | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(default=600.0, gt=0, allow_inf_nan=False)
    retryable_exit_codes: tuple[int, ...] = ()
    result_format: Literal["single_jsonl", "json"] = "single_jsonl"
    result_path: Path | None = Path("output.jsonl")
    identity_files: tuple[Path, ...] = ()

    @field_validator("semantic_version")
    @classmethod
    def validate_semantic_version(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError(
                "semantic_version must be non-blank and contain no NUL"
            )
        return value

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for index, argument in enumerate(value):
            _validate_template(argument, label=f"argv[{index}]")
        return value

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str | Path | None) -> str | None:
        if value is None:
            return None
        return _validate_template(str(value), label="cwd")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for key, template in value.items():
            if not key or "=" in key or "\x00" in key:
                raise ValueError(f"invalid environment variable name: {key!r}")
            _validate_template(
                template,
                label=f"environment[{key!r}]",
            )
        return value

    @field_validator("retryable_exit_codes")
    @classmethod
    def validate_retryable_exit_codes(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if 0 in value:
            raise ValueError("retryable_exit_codes must not contain zero")
        if len(set(value)) != len(value):
            raise ValueError("retryable_exit_codes must be unique")
        return value

    @field_validator("result_path")
    @classmethod
    def validate_result_path(cls, value: Path | None) -> Path | None:
        return _validate_relative_result_path(value)

    @model_validator(mode="after")
    def validate_identity_files(self) -> "CommandProcessorConfig":
        normalized = [str(path) for path in self.identity_files]
        if len(set(normalized)) != len(normalized):
            raise ValueError("identity_files must be unique")
        return self


class CommandProcessorError(BatchItemError):
    def __init__(
        self,
        message: str,
        *,
        kind: Literal[
            "template",
            "invalid_item",
            "spawn",
            "timeout",
            "exit_code",
            "invalid_result",
            "unsafe_artifact",
        ],
        retryable: bool,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.kind = kind
        self.returncode = returncode


class _CommandCancelled(asyncio.CancelledError):
    def __init__(self, *, stdout: bytes, stderr: bytes) -> None:
        super().__init__("command processing was cancelled")
        self.stdout = stdout
        self.stderr = stderr


def _absolute_path(path: Path, *, base: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(candidate))


def _identity_file(path: Path, *, config_dir: Path) -> Path:
    absolute = _absolute_path(path, base=config_dir)
    try:
        resolved = absolute.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"identity file does not exist: {absolute}") from error
    if absolute != resolved:
        raise ValueError(f"identity file must not use symlinks: {absolute}")
    if not resolved.is_file():
        raise ValueError(f"identity file is not a regular file: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _semantic_cwd(config: CommandProcessorConfig, config_dir: Path) -> str | None:
    if config.cwd is None:
        return None
    cwd = str(config.cwd)
    if _template_fields(cwd):
        return cwd
    return str(_absolute_path(Path(cwd), base=config_dir))


def _semantic_payload(
    config: CommandProcessorConfig,
    *,
    config_dir: Path,
) -> dict[str, Any]:
    return {
        "argv": list(config.argv),
        "cwd": _semantic_cwd(config, config_dir),
        "environment": dict(sorted(config.environment.items())),
        "result_format": config.result_format,
        "result_path": (
            None if config.result_path is None else config.result_path.as_posix()
        ),
        "semantic_version": config.semantic_version,
    }


def _semantic_sha256(
    config: CommandProcessorConfig,
    *,
    config_dir: Path,
) -> str:
    data = json.dumps(
        _semantic_payload(config, config_dir=config_dir),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _render_template(template: str, values: Mapping[str, str]) -> str:
    try:
        return template.format_map(values)
    except KeyError as error:
        missing = str(error.args[0])
        raise CommandProcessorError(
            f"command template requires unavailable value: {missing}",
            kind="template",
            retryable=False,
        ) from error
    except ValueError as error:
        raise CommandProcessorError(
            f"cannot render command template: {template!r}",
            kind="template",
            retryable=False,
        ) from error


def _item_snapshot_bytes(item: BatchItem) -> bytes:
    try:
        payload = item.model_dump(
            mode="json",
            include=_ITEM_SNAPSHOT_FIELDS,
        )
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (PydanticSerializationError, TypeError, ValueError) as error:
        raise CommandProcessorError(
            f"cannot serialize batch item snapshot: {item.item_id}: {error}",
            kind="invalid_item",
            retryable=False,
        ) from error


def _write_item_snapshot(item: BatchItem, attempt_dir: Path) -> Path:
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise CommandProcessorError(
            f"processor attempt_dir is not a safe directory: {attempt_dir}",
            kind="unsafe_artifact",
            retryable=False,
        )
    destination = attempt_dir / "item.json"
    if destination.is_symlink():
        raise CommandProcessorError(
            f"batch item snapshot path must not be a symlink: {destination}",
            kind="unsafe_artifact",
            retryable=False,
        )

    data = _item_snapshot_bytes(item)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=attempt_dir,
            prefix=".item.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(attempt_dir, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise CommandProcessorError(
            f"cannot write batch item snapshot: {destination}: {error}",
            kind="unsafe_artifact",
            retryable=False,
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _template_values(
    item: BatchItem,
    context: ProcessorContext,
    *,
    item_json: Path,
) -> dict[str, str]:
    values = {
        "item_id": item.item_id,
        "input_path": str(item.input_path),
        "item_json": str(item_json),
        "output_dir": str(context.output_dir),
        "workspace": str(context.workspace),
    }
    values.update(
        {
            f"media_{alias}": str(path)
            for alias, path in item.media.items()
        }
    )
    return values


def _command_environment(
    configured: Mapping[str, str],
    values: Mapping[str, str],
) -> dict[str, str] | None:
    if not configured:
        return None
    environment = os.environ.copy()
    environment.update(
        {
            key: _render_template(template, values)
            for key, template in configured.items()
        }
    )
    return environment


def _command_cwd(
    template: str | None,
    values: Mapping[str, str],
    *,
    config_dir: Path,
) -> Path | None:
    if template is None:
        return None
    rendered = _render_template(template, values)
    cwd = _absolute_path(Path(rendered), base=config_dir)
    try:
        resolved = cwd.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CommandProcessorError(
            f"command cwd does not exist: {cwd}",
            kind="template",
            retryable=False,
        ) from error
    if not resolved.is_dir():
        raise CommandProcessorError(
            f"command cwd is not a directory: {resolved}",
            kind="template",
            retryable=False,
        )
    return resolved


def _as_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    return value if isinstance(value, bytes) else value.encode("utf-8")


def _write_log(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CommandProcessorError(
            f"command log path must not be a symlink: {path}",
            kind="unsafe_artifact",
            retryable=False,
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def _run_subprocess(
    *,
    argv: tuple[str, ...],
    cwd: Path | None,
    environment: dict[str, str] | None,
    timeout_s: float,
) -> subprocess.CompletedProcess[bytes]:
    spawn = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    )
    try:
        process = await asyncio.shield(spawn)
    except asyncio.CancelledError as error:
        # Shield subprocess creation so cancellation cannot lose a child in the
        # narrow interval between fork/exec and receiving its process handle.
        try:
            process = await asyncio.shield(spawn)
        except BaseException:
            raise _CommandCancelled(stdout=b"", stderr=b"") from error
        communication = asyncio.create_task(process.communicate())
        stdout, stderr = await _terminate_process_group(
            process,
            communication,
        )
        raise _CommandCancelled(stdout=stdout, stderr=stderr) from error

    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communication),
            timeout=timeout_s,
        )
    except TimeoutError as error:
        stdout, stderr = await _terminate_process_group(
            process,
            communication,
        )
        raise subprocess.TimeoutExpired(
            cmd=argv,
            timeout=timeout_s,
            output=stdout,
            stderr=stderr,
        ) from error
    except asyncio.CancelledError as error:
        stdout, stderr = await _terminate_process_group(
            process,
            communication,
        )
        raise _CommandCancelled(stdout=stdout, stderr=stderr) from error
    return subprocess.CompletedProcess(
        args=argv,
        returncode=process.returncode,
        stdout=_as_bytes(stdout),
        stderr=_as_bytes(stderr),
    )


def _signal_process_group(
    process: asyncio.subprocess.Process,
    signal_number: signal.Signals,
) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def _process_group_exists(process: asyncio.subprocess.Process) -> bool:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    _signal_process_group(process, signal.SIGTERM)
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communication),
            timeout=_PROCESS_GROUP_TERM_GRACE_S,
        )
    except TimeoutError:
        _signal_process_group(process, signal.SIGKILL)
        stdout, stderr = await asyncio.shield(communication)
        return _as_bytes(stdout), _as_bytes(stderr)

    # The direct child may exit and close its pipes while a descendant that
    # ignored SIGTERM remains in the session. Kill the surviving group before
    # returning so a retry cannot overlap with an orphaned annotator/FFmpeg.
    if _process_group_exists(process):
        _signal_process_group(process, signal.SIGKILL)
    return _as_bytes(stdout), _as_bytes(stderr)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def _parse_result(
    data: bytes,
    *,
    result_format: Literal["single_jsonl", "json"],
    source: str,
) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CommandProcessorError(
            f"command result is not UTF-8: {source}",
            kind="invalid_result",
            retryable=False,
        ) from error

    if result_format == "single_jsonl":
        lines = text.splitlines()
        if len(lines) != 1 or not lines[0].strip():
            raise CommandProcessorError(
                f"command single_jsonl result must contain exactly one line: {source}",
                kind="invalid_result",
                retryable=False,
            )
        text = lines[0]

    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_strict_json_object_pairs,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise CommandProcessorError(
            f"command result is not strict JSON: {source}: {error}",
            kind="invalid_result",
            retryable=False,
        ) from error
    if not isinstance(payload, dict):
        raise CommandProcessorError(
            f"command result must be a JSON object: {source}",
            kind="invalid_result",
            retryable=False,
        )
    return payload


def _prepare_output_directory(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise CommandProcessorError(
            f"processor output_dir must not be a symlink: {output_dir}",
            kind="unsafe_artifact",
            retryable=False,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise CommandProcessorError(
            f"processor output_dir is not a safe directory: {output_dir}",
            kind="unsafe_artifact",
            retryable=False,
        )


def _scan_artifacts(output_dir: Path) -> tuple[Path, ...]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise CommandProcessorError(
            f"processor output_dir is not a safe directory: {output_dir}",
            kind="unsafe_artifact",
            retryable=False,
        )

    root = output_dir.resolve(strict=True)
    artifacts: list[Path] = []
    for current, directory_names, file_names in os.walk(
        output_dir,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in (*directory_names, *file_names):
            candidate = current_path / name
            try:
                mode = candidate.lstat().st_mode
            except OSError as error:
                raise CommandProcessorError(
                    f"cannot inspect command artifact: {candidate}",
                    kind="unsafe_artifact",
                    retryable=False,
                ) from error
            if stat.S_ISLNK(mode):
                raise CommandProcessorError(
                    f"command artifact must not be a symlink: {candidate}",
                    kind="unsafe_artifact",
                    retryable=False,
                )
            if name in file_names:
                if not stat.S_ISREG(mode):
                    raise CommandProcessorError(
                        f"command artifact must be a regular file: {candidate}",
                        kind="unsafe_artifact",
                        retryable=False,
                    )
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise CommandProcessorError(
                        f"command artifact escapes output_dir: {candidate}",
                        kind="unsafe_artifact",
                        retryable=False,
                    )
                artifacts.append(candidate)
    return tuple(sorted(artifacts, key=lambda path: path.as_posix()))


class CommandProcessorSession:
    def __init__(
        self,
        config: CommandProcessorConfig,
        *,
        config_dir: Path,
    ) -> None:
        self._config = config
        self._config_dir = config_dir

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult:
        item_json = _write_item_snapshot(item, context.attempt_dir)
        values = _template_values(item, context, item_json=item_json)
        argv = tuple(
            _render_template(argument, values)
            for argument in self._config.argv
        )
        cwd = _command_cwd(
            None if self._config.cwd is None else str(self._config.cwd),
            values,
            config_dir=self._config_dir,
        )
        environment = _command_environment(self._config.environment, values)
        _prepare_output_directory(context.output_dir)

        stdout_path = context.attempt_dir / "stdout.log"
        stderr_path = context.attempt_dir / "stderr.log"
        try:
            completed = await _run_subprocess(
                argv=argv,
                cwd=cwd,
                environment=environment,
                timeout_s=self._config.timeout_s,
            )
        except asyncio.CancelledError as error:
            _write_log(
                stdout_path,
                _as_bytes(getattr(error, "stdout", None)),
            )
            _write_log(
                stderr_path,
                _as_bytes(getattr(error, "stderr", None)),
            )
            raise
        except subprocess.TimeoutExpired as error:
            _write_log(stdout_path, _as_bytes(error.stdout))
            _write_log(stderr_path, _as_bytes(error.stderr))
            raise CommandProcessorError(
                f"command timed out after {self._config.timeout_s:g}s",
                kind="timeout",
                retryable=True,
            ) from error
        except OSError as error:
            _write_log(stdout_path, b"")
            _write_log(stderr_path, str(error).encode("utf-8"))
            raise CommandProcessorError(
                f"cannot start command: {error}",
                kind="spawn",
                retryable=False,
            ) from error

        _write_log(stdout_path, completed.stdout)
        _write_log(stderr_path, completed.stderr)
        if completed.returncode != 0:
            raise CommandProcessorError(
                f"command exited with status {completed.returncode}",
                kind="exit_code",
                retryable=(
                    completed.returncode
                    in self._config.retryable_exit_codes
                ),
                returncode=completed.returncode,
            )

        artifacts = _scan_artifacts(context.output_dir)
        if self._config.result_path is None:
            result_bytes = completed.stdout
            result_source = "stdout"
        else:
            result_path = context.output_dir / self._config.result_path
            if result_path not in artifacts:
                raise CommandProcessorError(
                    f"command result file does not exist: {result_path}",
                    kind="invalid_result",
                    retryable=False,
                )
            try:
                result_bytes = result_path.read_bytes()
            except OSError as error:
                raise CommandProcessorError(
                    f"cannot read command result file: {result_path}",
                    kind="invalid_result",
                    retryable=False,
                ) from error
            result_source = str(result_path)
        record = _parse_result(
            result_bytes,
            result_format=self._config.result_format,
            source=result_source,
        )
        return ProcessorResult(record=record, artifacts=artifacts)


class CommandProcessorProvider:
    def __init__(
        self,
        config: CommandProcessorConfig,
        config_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._config_dir = _absolute_path(
            config_dir or Path.cwd(),
            base=Path.cwd(),
        )

    @property
    def identity(self) -> Mapping[str, str]:
        identity: dict[str, str] = {
            "command_processor_version": COMMAND_PROCESSOR_VERSION,
            "command_semantic_version": self._config.semantic_version,
            "command_semantic_sha256": _semantic_sha256(
                self._config,
                config_dir=self._config_dir,
            ),
        }
        for index, configured_path in enumerate(self._config.identity_files):
            path = _identity_file(
                configured_path,
                config_dir=self._config_dir,
            )
            identity[f"identity_file:{index}:path"] = str(path)
            identity[f"identity_file:{index}:sha256"] = _sha256_file(path)
        return identity

    def preflight(self, items: Sequence[BatchItem]) -> None:
        templates = [*self._config.argv, *self._config.environment.values()]
        if self._config.cwd is not None:
            templates.append(str(self._config.cwd))
        required_aliases = {
            field.removeprefix("media_")
            for template in templates
            for field in _template_fields(template)
            if field.startswith("media_")
        }
        if not required_aliases:
            return

        missing_by_item = {
            item.item_id: sorted(required_aliases - item.media.keys())
            for item in items
            if required_aliases - item.media.keys()
        }
        if not missing_by_item:
            return
        details = "; ".join(
            f"{item_id}: {', '.join(aliases)}"
            for item_id, aliases in sorted(missing_by_item.items())
        )
        raise CommandProcessorError(
            "command templates require unavailable media aliases for "
            f"batch items: {details}",
            kind="template",
            retryable=False,
        )

    @asynccontextmanager
    async def open(self) -> AsyncIterator[CommandProcessorSession]:
        yield CommandProcessorSession(
            self._config,
            config_dir=self._config_dir,
        )


def _create_provider(
    validated_config: BaseModel,
    config_dir: Path,
) -> CommandProcessorProvider:
    if not isinstance(validated_config, CommandProcessorConfig):
        raise TypeError(
            "command processor requires CommandProcessorConfig, got "
            f"{type(validated_config)!r}"
        )
    return CommandProcessorProvider(validated_config, config_dir)


PLUGIN = ProcessorPlugin(
    config_model=CommandProcessorConfig,
    factory=_create_provider,
)
