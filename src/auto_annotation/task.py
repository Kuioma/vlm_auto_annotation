from pathlib import Path

import yaml

from auto_annotation.domain.models import OrderedTask


def load_ordered_task(path: Path) -> OrderedTask:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ordered task must be a YAML mapping: {path}")
    return OrderedTask.model_validate(payload)
