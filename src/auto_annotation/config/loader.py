import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from auto_annotation.config.models import RunConfig


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def load_run_config(path: Path) -> RunConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RunConfig.model_validate(raw)
    base = path.resolve().parent
    resolved = config.model_copy(
        update={
            "schema_path": _resolve(base, config.schema_path),
            "ontology_path": _resolve(base, config.ontology_path),
            "manifest_path": _resolve(base, config.manifest_path),
            "artifact_root": _resolve(base, config.artifact_root),
            "output_path": _resolve(base, config.output_path),
            "backend": config.backend.model_copy(
                update={"fixture_path": _resolve(base, config.backend.fixture_path)}
            ),
        }
    )
    resolved._source_path = path.resolve()
    return resolved


def config_hash(
    config: RunConfig, identity: Mapping[str, str] | None = None
) -> str:
    payload = json.dumps(
        {
            "config": config.model_dump(mode="json"),
            "identity": dict(sorted((identity or {}).items())),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
