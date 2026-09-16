"""Content identity for reproducible evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from auto_annotation.evaluation.config import EvaluationConfig, canonical_config_bytes


def sha256_label(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def aggregate_annotation_bytes(records: Mapping[int, bytes]) -> bytes:
    payload = bytearray()
    for episode_index, raw in sorted(records.items()):
        index_bytes = str(episode_index).encode("ascii")
        payload.extend(len(index_bytes).to_bytes(8, "big"))
        payload.extend(index_bytes)
        payload.extend(len(raw).to_bytes(8, "big"))
        payload.extend(raw)
    return bytes(payload)


def build_evaluation_identity(
    *,
    config: EvaluationConfig,
    versions: Mapping[str, object],
    gold_manifest_bytes: bytes,
    selected_annotation_bytes: Mapping[int, bytes],
    prediction_bytes: bytes,
) -> tuple[str, dict[str, str]]:
    annotation_aggregate = aggregate_annotation_bytes(selected_annotation_bytes)
    components = {
        "config_sha256": sha256_label(canonical_config_bytes(config)),
        "gold_manifest_sha256": sha256_label(gold_manifest_bytes),
        "included_annotations_sha256": sha256_label(annotation_aggregate),
        "prediction_sha256": sha256_label(prediction_bytes),
    }
    digest = hashlib.sha256()
    for label, raw in (
        (
            "versions",
            json.dumps(
                versions,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8"),
        ),
        ("config", canonical_config_bytes(config)),
        ("manifest", gold_manifest_bytes),
        ("annotations", annotation_aggregate),
        ("predictions", prediction_bytes),
    ):
        label_bytes = label.encode("ascii")
        digest.update(len(label_bytes).to_bytes(8, "big"))
        digest.update(label_bytes)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return f"eval-{digest.hexdigest()}", components

