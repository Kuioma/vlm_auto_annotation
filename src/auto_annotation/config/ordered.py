"""Structured run configuration for the ordered segmentation CLI."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from auto_annotation.config.models import ConfigModel


class OrderedSamplingConfig(ConfigModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    coarse_fps: float = Field(default=2.0, gt=0)
    refine_fps: float = Field(default=10.0, gt=0)
    refine_window_ms: int = Field(default=2000, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_rates(self) -> "OrderedSamplingConfig":
        if self.refine_fps <= self.coarse_fps:
            raise ValueError("refine_fps must be greater than coarse_fps")
        return self


class OrderedBackendConfig(ConfigModel):
    kind: Literal["vllm", "openai_compatible"] = "vllm"
    base_url: str | None = None
    base_url_env: str | None = None
    model_id: str = Field(default="Qwen/Qwen3.6-27B", min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    vllm_version: str = Field(default="0.24.0", min_length=1)
    timeout_s: int = Field(default=600, gt=0, strict=True)
    media_staging_root: Path = Path("/home/cz-sjj/ws/vllm/media")
    api_key_env: str | None = None


class OrderedRunConfig(ConfigModel):
    input_mode: Literal["direct", "wa2"] = "direct"
    video: Path
    left_wrist_video: Path | None = None
    right_wrist_video: Path | None = None
    output_dir: Path
    schema_path: Path
    ontology_path: Path
    task_path: Path
    backend: OrderedBackendConfig = Field(default_factory=OrderedBackendConfig)
    sampling: OrderedSamplingConfig = Field(default_factory=OrderedSamplingConfig)
    columns: int = Field(default=3, gt=0, strict=True)
    prepare_only: bool = Field(default=False, strict=True)


def load_ordered_defaults(path: Path) -> dict[str, object]:
    """Validate YAML and translate it into existing CLI argument defaults."""
    path = path.expanduser().resolve(strict=True)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid ordered YAML: {path}") from exc
    config = OrderedRunConfig.model_validate(payload)
    defaults = config.model_dump(exclude={"backend", "sampling"})
    defaults.update(config.backend.model_dump(exclude={"kind"}))
    defaults["backend_kind"] = config.backend.kind
    defaults.update(config.sampling.model_dump(exclude={"coarse_fps"}))
    defaults["fps"] = config.sampling.coarse_fps
    for key, value in defaults.items():
        if isinstance(value, Path):
            expanded = value.expanduser()
            defaults[key] = (path.parent / expanded).resolve()
    return defaults
