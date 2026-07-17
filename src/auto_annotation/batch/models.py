from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


class BatchModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_safe_component(value: str, *, field_name: str) -> str:
    if (
        value in {"", ".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe {field_name}: {value!r}")
    return value


class PluginSpec(BatchModel):
    plugin: str = Field(min_length=3)
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("plugin")
    @classmethod
    def validate_import_path(cls, value: str) -> str:
        module_name, separator, attribute_path = value.partition(":")
        if (
            separator != ":"
            or not module_name
            or not attribute_path
            or any(
                not component.isidentifier()
                for component in (*module_name.split("."), *attribute_path.split("."))
            )
        ):
            raise ValueError(
                "plugin must use the import path form "
                "'package.module:attribute'"
            )
        return value


class BatchItem(BatchModel):
    item_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    media: dict[str, Path] = Field(min_length=1)
    primary_media: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_fingerprint: str

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _validate_safe_component(value, field_name="item_id")

    @field_validator("media")
    @classmethod
    def validate_media(cls, value: dict[str, Path]) -> dict[str, Path]:
        seen_paths: set[Path] = set()
        for alias, path in value.items():
            _validate_safe_component(alias, field_name="media alias")
            if not path.is_absolute():
                raise ValueError(f"batch item media path must be absolute: {path}")
            normalized = path.resolve(strict=False)
            if normalized in seen_paths:
                raise ValueError(f"duplicate batch item media path: {path}")
            seen_paths.add(normalized)
        return value

    @field_validator("input_fingerprint")
    @classmethod
    def normalize_fingerprint(cls, value: str) -> str:
        digest = value.removeprefix("sha256:").lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "input_fingerprint must be a 64-character SHA-256 digest"
            )
        return f"sha256:{digest}"

    @model_validator(mode="after")
    def validate_primary_media(self) -> BatchItem:
        if self.primary_media not in self.media:
            raise ValueError("primary_media must name an entry in media")
        return self

    @property
    def input_path(self) -> Path:
        return self.media[self.primary_media]


@dataclass(frozen=True)
class ProcessorContext:
    run_id: str
    workspace: Path
    attempt_dir: Path
    output_dir: Path
    attempt: int


class ProcessorResult(BatchModel):
    record: dict[str, Any]
    artifacts: tuple[Path, ...] = ()


class BatchFailure(BatchModel):
    item_id: str
    attempt_count: int = Field(ge=1)
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool
    preserved_result: bool = False


class BatchExecutionConfig(BatchModel):
    resume: bool = True
    retry_terminal_failures: bool = False
    max_attempts: int = Field(default=1, ge=1)
    retry_backoff_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    continue_on_error: bool = True
    concurrency: int = Field(default=1, ge=1)
    start_index: int = Field(default=0, ge=0)
    end_index: int | None = Field(default=None, ge=1)
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selection(self) -> BatchExecutionConfig:
        if self.end_index is not None and self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        return self


class BatchOutputConfig(BatchModel):
    workspace: Path


class BatchConfig(BatchModel):
    _source_path: Path | None = PrivateAttr(default=None)

    version: Literal[1] = 1
    source: PluginSpec
    processor: PluginSpec
    execution: BatchExecutionConfig = Field(default_factory=BatchExecutionConfig)
    output: BatchOutputConfig

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def config_dir(self) -> Path:
        if self._source_path is None:
            return Path.cwd().resolve()
        return self._source_path.parent
