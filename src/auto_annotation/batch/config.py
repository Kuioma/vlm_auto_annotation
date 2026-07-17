from pathlib import Path

import yaml

from auto_annotation.batch.models import BatchConfig


def load_batch_config(path: Path) -> BatchConfig:
    source_path = path.expanduser().resolve(strict=True)
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("batch config must contain a YAML object")

    config = BatchConfig.model_validate(raw)
    workspace = config.output.workspace.expanduser()
    if not workspace.is_absolute():
        workspace = source_path.parent / workspace
    resolved = config.model_copy(
        update={
            "output": config.output.model_copy(
                update={"workspace": workspace.resolve(strict=False)}
            )
        }
    )
    resolved._source_path = source_path
    return resolved
