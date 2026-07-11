import json
from pathlib import Path

import pytest

from auto_annotation.manifest import read_manifest


def test_manifest_resolves_local_video_and_rejects_duplicates(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    record = {
        "video_id": "clip-1",
        "video_uri": "clip.mp4",
        "dataset_id": "demo",
        "schema_version": "subtask-v1",
        "ontology_id": "kitchen-v1",
    }
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        "\n".join([json.dumps(record), json.dumps(record)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate video_id"):
        read_manifest(manifest)


def test_manifest_rejects_remote_uri(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "clip-1",
                "video_uri": "https://example.invalid/clip.mp4",
                "dataset_id": "demo",
                "schema_version": "subtask-v1",
                "ontology_id": "kitchen-v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="local video"):
        read_manifest(manifest)
