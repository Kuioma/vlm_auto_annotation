import json
from pathlib import Path

from auto_annotation.domain.models import ManifestItem


def read_manifest(path: Path) -> list[ManifestItem]:
    base = path.resolve().parent
    items: list[ManifestItem] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        uri = str(raw["video_uri"])
        if "://" in uri:
            raise ValueError(f"line {line_number}: video_uri must be a local video")
        video_path = Path(uri)
        raw["video_uri"] = (
            video_path if video_path.is_absolute() else (base / video_path).resolve()
        )
        item = ManifestItem.model_validate(raw)
        key = (item.dataset_id, item.video_id)
        if key in seen:
            raise ValueError(f"duplicate video_id in dataset: {item.video_id}")
        if not item.video_uri.is_file():
            raise ValueError(f"video does not exist: {item.video_uri}")
        seen.add(key)
        items.append(item)
    return items
