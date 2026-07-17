from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

from auto_annotation.batch.models import (
    BatchItem,
    PluginSpec,
    ProcessorContext,
    ProcessorResult,
)


class BatchItemError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class SourceProvider(Protocol):
    def discover(self) -> tuple[BatchItem, ...]: ...


class ProcessorSession(Protocol):
    """One processor session shared by the bounded batch worker pool.

    ``process`` may be awaited concurrently up to the configured execution
    concurrency. Implementations must either be concurrency-safe or serialize
    access to any state that cannot be shared across items. On cancellation,
    implementations must stop and reap owned resources before re-raising
    ``CancelledError``.
    """

    async def process(
        self,
        item: BatchItem,
        context: ProcessorContext,
    ) -> ProcessorResult: ...


class ProcessorProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, str]: ...

    def open(
        self,
    ) -> AbstractAsyncContextManager[ProcessorSession]: ...


ProviderT = TypeVar("ProviderT")
PluginFactory = Callable[[BaseModel, Path], ProviderT]


@dataclass(frozen=True)
class ProcessorPlugin:
    config_model: type[BaseModel]
    factory: PluginFactory[ProcessorProvider]


@dataclass(frozen=True)
class SourcePlugin:
    config_model: type[BaseModel]
    factory: PluginFactory[SourceProvider]


def _import_attribute(import_path: str) -> object:
    module_name, _, attribute_path = import_path.partition(":")
    value: object = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        try:
            value = getattr(value, component)
        except AttributeError as error:
            raise ImportError(
                f"plugin attribute does not exist: {import_path}"
            ) from error
    return value


def _validate_plugin_model(config_model: type[BaseModel]) -> None:
    if not isinstance(config_model, type) or not issubclass(config_model, BaseModel):
        raise TypeError("plugin config_model must be a Pydantic BaseModel class")
    if config_model.model_config.get("extra") != "forbid":
        raise TypeError("plugin config_model must set extra='forbid'")


def _validated_plugin_config(
    spec: PluginSpec,
    plugin: ProcessorPlugin | SourcePlugin,
    *,
    config_dir: Path,
) -> BaseModel:
    _validate_plugin_model(plugin.config_model)
    resolved_config_dir = config_dir.expanduser().resolve(strict=True)
    if not resolved_config_dir.is_dir():
        raise ValueError(
            f"batch config directory is not a directory: {resolved_config_dir}"
        )
    return plugin.config_model.model_validate(
        spec.config,
        context={"config_dir": resolved_config_dir},
    )


def load_processor_provider(
    spec: PluginSpec,
    *,
    config_dir: Path,
) -> ProcessorProvider:
    imported = _import_attribute(spec.plugin)
    if not isinstance(imported, ProcessorPlugin):
        raise TypeError(
            f"processor plugin must export ProcessorPlugin: {spec.plugin}"
        )
    config = _validated_plugin_config(spec, imported, config_dir=config_dir)
    provider = imported.factory(config, config_dir.resolve(strict=True))
    return cast(ProcessorProvider, provider)


def load_source_provider(
    spec: PluginSpec,
    *,
    config_dir: Path,
) -> SourceProvider:
    imported = _import_attribute(spec.plugin)
    if not isinstance(imported, SourcePlugin):
        raise TypeError(f"source plugin must export SourcePlugin: {spec.plugin}")
    config = _validated_plugin_config(spec, imported, config_dir=config_dir)
    provider = imported.factory(config, config_dir.resolve(strict=True))
    return cast(SourceProvider, provider)
