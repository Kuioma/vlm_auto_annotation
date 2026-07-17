from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from auto_annotation.batch.models import BatchItem
from auto_annotation.batch.plugins import SourcePlugin, SourceProvider


WA2_SOURCE_VERSION = "wa2-episodes-source-v1"


class Wa2SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: Path
    dataset_id: str | None = Field(default=None, min_length=1)
    episodes_path: Path = Path("meta/episodes.jsonl")
    media: dict[str, str] = Field(min_length=1)
    primary_media: str | None = Field(default=None, min_length=1)
    fingerprint_mode: Literal["stat", "sha256"] = "sha256"

    @field_validator("media")
    @classmethod
    def validate_media_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        for alias, video_key in value.items():
            if alias in {"", ".", ".."} or "/" in alias or "\\" in alias:
                raise ValueError(f"unsafe media alias: {alias!r}")
            if not video_key.strip():
                raise ValueError(f"empty episodes video key for alias {alias!r}")
        if len(set(value.values())) != len(value):
            raise ValueError("duplicate selected media video keys")
        return value

    @model_validator(mode="after")
    def validate_primary_media(self) -> Wa2SourceConfig:
        if self.primary_media is not None and self.primary_media not in self.media:
            raise ValueError("primary_media must name a configured media alias")
        return self


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


class Wa2Source:
    def __init__(self, config: Wa2SourceConfig, *, config_dir: Path) -> None:
        self.config = config
        self.config_dir = config_dir.expanduser().resolve(strict=True)

    def _dataset_root(self) -> Path:
        root = self.config.dataset_root.expanduser()
        if not root.is_absolute():
            root = self.config_dir / root
        try:
            root = root.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"WA2 dataset root does not exist: {root}"
            ) from error
        if not root.is_dir():
            raise ValueError(f"WA2 dataset root is not a directory: {root}")
        return root

    def _episodes_path(self, dataset_root: Path) -> Path:
        path = self.config.episodes_path.expanduser()
        if not path.is_absolute():
            path = dataset_root / path
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(dataset_root):
            raise ValueError(
                f"episodes path escapes dataset root {dataset_root}: {path}"
            )
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"episodes metadata does not exist: {path}") from error
        if not resolved.is_file():
            raise ValueError(f"episodes metadata is not a file: {resolved}")
        return resolved

    @staticmethod
    def _read_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read episodes metadata: {path}") from error
        rows: list[tuple[int, dict[str, Any]]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid episodes JSON on line {line_number}: {path}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"episode row {line_number} must be a JSON object: {path}"
                )
            rows.append((line_number, cast(dict[str, Any], value)))
        if not rows:
            raise ValueError(f"episodes metadata contains no rows: {path}")
        return rows

    @staticmethod
    def _resolve_media_path(
        dataset_root: Path,
        raw_path: object,
        *,
        alias: str,
        line_number: int,
    ) -> Path:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"episode row {line_number} has an invalid path for {alias}"
            )
        path = Path(raw_path).expanduser()
        candidate = path if path.is_absolute() else dataset_root / path
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(dataset_root):
            raise ValueError(
                f"episode media path escapes dataset root {dataset_root}: "
                f"{raw_path}"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"episode media does not exist for {alias}: {candidate}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"episode media is not a file for {alias}: {resolved}")
        return resolved

    def _media_identity(self, path: Path) -> dict[str, Any]:
        if self.config.fingerprint_mode == "sha256":
            return {"sha256": _sha256_file(path)}
        stat = path.stat()
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _item_fingerprint(
        self,
        *,
        dataset_root: Path,
        dataset_id: str,
        primary_media: str,
        row: dict[str, Any],
        media: dict[str, Path],
    ) -> str:
        payload = {
            "source_version": WA2_SOURCE_VERSION,
            "fingerprint_mode": self.config.fingerprint_mode,
            "dataset_id": dataset_id,
            "primary_media": primary_media,
            "video_keys": self.config.media,
            "episode": row,
            "media": {
                alias: {
                    "path": path.relative_to(dataset_root).as_posix(),
                    "identity": self._media_identity(path),
                }
                for alias, path in media.items()
            },
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def discover(self) -> tuple[BatchItem, ...]:
        dataset_root = self._dataset_root()
        episodes_path = self._episodes_path(dataset_root)
        parsed: list[tuple[int, int, dict[str, Any]]] = []
        seen_indices: set[int] = set()
        for line_number, row in self._read_rows(episodes_path):
            episode_index = row.get("episode_index")
            if type(episode_index) is not int or episode_index < 0:
                raise ValueError(
                    f"episode row {line_number} must contain a non-negative "
                    f"integer episode_index: {episodes_path}"
                )
            if episode_index in seen_indices:
                raise ValueError(f"duplicate episode_index: {episode_index}")
            seen_indices.add(episode_index)
            parsed.append((episode_index, line_number, row))

        dataset_id = self.config.dataset_id or dataset_root.name
        primary_media = self.config.primary_media or next(iter(self.config.media))
        seen_media_paths: dict[Path, str] = {}
        items: list[BatchItem] = []
        for episode_index, line_number, row in sorted(parsed):
            videos = row.get("videos")
            if not isinstance(videos, dict):
                raise ValueError(
                    f"episode row {line_number} must contain a videos object"
                )
            media: dict[str, Path] = {}
            for alias, video_key in self.config.media.items():
                if video_key not in videos:
                    raise ValueError(
                        f"episode row {line_number} is missing configured "
                        f"video key {video_key!r} for {alias}"
                    )
                media[alias] = self._resolve_media_path(
                    dataset_root,
                    videos[video_key],
                    alias=alias,
                    line_number=line_number,
                )
            if len(set(media.values())) != len(media):
                raise ValueError(
                    f"duplicate selected media paths in episode {episode_index}"
                )
            item_id = f"episode_{episode_index:06d}"
            for path in media.values():
                previous_item = seen_media_paths.get(path)
                if previous_item is not None:
                    raise ValueError(
                        f"reused media path across items {previous_item} and "
                        f"{item_id}: {path}"
                    )
                seen_media_paths[path] = item_id

            task = row.get("tasks")
            context = {"task": task} if isinstance(task, str) else {}
            metadata: dict[str, Any] = {"episode_index": episode_index}
            if "length" in row:
                metadata["length"] = row["length"]
            metadata["episodes_path"] = episodes_path.relative_to(
                dataset_root
            ).as_posix()
            items.append(
                BatchItem(
                    item_id=item_id,
                    dataset_id=dataset_id,
                    media=media,
                    primary_media=primary_media,
                    context=context,
                    metadata=metadata,
                    input_fingerprint=self._item_fingerprint(
                        dataset_root=dataset_root,
                        dataset_id=dataset_id,
                        primary_media=primary_media,
                        row=row,
                        media=media,
                    ),
                )
            )
        return tuple(items)


def _build_source(config: BaseModel, config_dir: Path) -> SourceProvider:
    typed = Wa2SourceConfig.model_validate(config)
    return Wa2Source(typed, config_dir=config_dir)


PLUGIN = SourcePlugin(
    config_model=Wa2SourceConfig,
    factory=_build_source,
)
