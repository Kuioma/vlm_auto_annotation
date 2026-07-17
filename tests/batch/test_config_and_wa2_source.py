import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from auto_annotation.batch.config import load_batch_config
from auto_annotation.batch.models import (
    BatchConfig,
    BatchExecutionConfig,
    BatchItem,
    BatchOutputConfig,
    PluginSpec,
)
from auto_annotation.batch.plugins import (
    ProcessorPlugin,
    load_processor_provider,
    load_source_provider,
)
from auto_annotation.batch.sources.wa2 import (
    Wa2Source,
    Wa2SourceConfig,
)


def _write_batch_config(path: Path) -> None:
    path.write_text(
        """
version: 1
source:
  plugin: auto_annotation.batch.sources.wa2:PLUGIN
  config:
    dataset_root: dataset
    media:
      HEAD_RGB: observation.images.head_rgb
processor:
  plugin: example.processor:PLUGIN
  config:
    relative_file: processor.yaml
execution:
  resume: true
  max_attempts: 2
  retry_backoff_s: 0
  continue_on_error: true
  concurrency: 1
output:
  workspace: batch-output
""".lstrip(),
        encoding="utf-8",
    )


def test_load_batch_config_resolves_only_batch_owned_paths(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs/batch.yaml"
    config_path.parent.mkdir()
    _write_batch_config(config_path)

    config = load_batch_config(config_path)

    assert config.source_path == config_path.resolve()
    assert config.config_dir == config_path.parent.resolve()
    assert config.output.workspace == (config_path.parent / "batch-output").resolve()
    assert config.source.config["dataset_root"] == "dataset"
    assert config.processor.config["relative_file"] == "processor.yaml"


def test_batch_config_forbids_unknown_fields() -> None:
    plugin = PluginSpec(plugin="example.plugin:PLUGIN", config={})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        BatchConfig.model_validate(
            {
                "version": 1,
                "source": plugin,
                "processor": plugin,
                "execution": {},
                "output": {"workspace": "/tmp/batch"},
                "surprise": True,
            }
        )


def test_execution_config_keeps_operational_selection_separate() -> None:
    config = BatchExecutionConfig(
        start_index=10,
        end_index=30,
        limit=5,
        max_attempts=3,
        retry_backoff_s=0.25,
        continue_on_error=False,
        concurrency=7,
        resume=False,
    )

    assert (config.start_index, config.end_index, config.limit) == (10, 30, 5)
    assert config.concurrency == 7
    with pytest.raises(ValidationError, match="end_index"):
        BatchExecutionConfig(start_index=10, end_index=10)
    with pytest.raises(ValidationError, match="concurrency"):
        BatchExecutionConfig(concurrency=0)


@pytest.mark.parametrize("unsafe_item_id", ["../item", "item/name", "item\x00id"])
def test_batch_item_rejects_unsafe_storage_identity(
    unsafe_item_id: str,
) -> None:
    with pytest.raises(ValidationError, match="unsafe item_id"):
        BatchItem(
            item_id=unsafe_item_id,
            dataset_id="demo",
            media={"HEAD_RGB": Path("/tmp/video.mp4")},
            primary_media="HEAD_RGB",
            input_fingerprint="sha256:" + "a" * 64,
        )


class _StrictPluginConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class _Provider:
    def __init__(self, name: str, config_dir: Path) -> None:
        self.name = name
        self.config_dir = config_dir


def test_plugin_config_is_strictly_validated_before_factory_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("batch_test_processor_plugin")
    factory_calls: list[tuple[str, Path]] = []

    def factory(config: BaseModel, config_dir: Path) -> _Provider:
        typed = _StrictPluginConfig.model_validate(config)
        factory_calls.append((typed.name, config_dir))
        return _Provider(typed.name, config_dir)

    module.PLUGIN = ProcessorPlugin(
        config_model=_StrictPluginConfig,
        factory=factory,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    invalid = PluginSpec(
        plugin=f"{module.__name__}:PLUGIN",
        config={"name": "processor", "unknown": True},
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_processor_provider(invalid, config_dir=tmp_path)
    assert factory_calls == []

    provider = load_processor_provider(
        PluginSpec(
            plugin=f"{module.__name__}:PLUGIN",
            config={"name": "processor"},
        ),
        config_dir=tmp_path,
    )
    assert isinstance(provider, _Provider)
    assert (provider.name, provider.config_dir) == ("processor", tmp_path)
    assert factory_calls == [("processor", tmp_path)]


def test_plugin_rejects_non_strict_config_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LooseConfig(BaseModel):
        name: str

    module = ModuleType("batch_test_loose_plugin")
    module.PLUGIN = ProcessorPlugin(
        config_model=LooseConfig,
        factory=lambda config, config_dir: object(),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="extra='forbid'"):
        load_processor_provider(
            PluginSpec(
                plugin=f"{module.__name__}:PLUGIN",
                config={"name": "processor"},
            ),
            config_dir=tmp_path,
        )


def _write_media(root: Path, relative_path: str, content: bytes) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_episodes(root: Path, rows: list[dict[str, Any]]) -> Path:
    path = root / "meta/episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


def _episode(index: int, *, head: str, wrist: str) -> dict[str, Any]:
    return {
        "episode_index": index,
        "tasks": "pick and place",
        "length": 100 + index,
        "videos": {
            "observation.images.head_rgb": head,
            "observation.images.right_wrist_rgb": wrist,
        },
    }


def _wa2_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _write_media(
        root,
        "videos/chunk-000/head/episode_000002.mp4",
        b"head-2",
    )
    _write_media(
        root,
        "videos/chunk-000/wrist/episode_000002.mp4",
        b"wrist-2",
    )
    _write_media(
        root,
        "videos/chunk-000/head/episode_000001.mp4",
        b"head-1",
    )
    _write_media(
        root,
        "videos/chunk-000/wrist/episode_000001.mp4",
        b"wrist-1",
    )
    _write_episodes(
        root,
        [
            _episode(
                2,
                head="videos/chunk-000/head/episode_000002.mp4",
                wrist="videos/chunk-000/wrist/episode_000002.mp4",
            ),
            _episode(
                1,
                head="videos/chunk-000/head/episode_000001.mp4",
                wrist="videos/chunk-000/wrist/episode_000001.mp4",
            ),
        ],
    )
    return root


@pytest.mark.parametrize("fingerprint_mode", ["stat", "sha256"])
def test_wa2_source_discovers_stably_sorted_multiview_items(
    tmp_path: Path,
    fingerprint_mode: str,
) -> None:
    root = _wa2_dataset(tmp_path)
    source = Wa2Source(
        Wa2SourceConfig(
            dataset_root=root,
            dataset_id="wa2-demo",
            media={
                "HEAD_RGB": "observation.images.head_rgb",
                "RIGHT_WRIST_RGB": "observation.images.right_wrist_rgb",
            },
            fingerprint_mode=fingerprint_mode,
        ),
        config_dir=tmp_path,
    )

    items = source.discover()

    assert [item.item_id for item in items] == [
        "episode_000001",
        "episode_000002",
    ]
    assert all(item.dataset_id == "wa2-demo" for item in items)
    assert list(items[0].media) == ["HEAD_RGB", "RIGHT_WRIST_RGB"]
    assert items[0].primary_media == "HEAD_RGB"
    assert items[0].input_path == items[0].media["HEAD_RGB"]
    assert items[0].media["HEAD_RGB"] == (
        root / "videos/chunk-000/head/episode_000001.mp4"
    ).resolve()
    assert items[0].context == {"task": "pick and place"}
    assert items[0].metadata["episode_index"] == 1
    assert items[0].metadata["length"] == 101
    assert items[0].input_fingerprint.startswith("sha256:")
    assert len(items[0].input_fingerprint) == 71
    assert items[0].input_fingerprint != items[1].input_fingerprint
    assert source.discover() == items


def test_wa2_source_plugin_resolves_dataset_root_from_batch_config_dir(
    tmp_path: Path,
) -> None:
    root = _wa2_dataset(tmp_path)

    provider = load_source_provider(
        PluginSpec(
            plugin="auto_annotation.batch.sources.wa2:PLUGIN",
            config={
                "dataset_root": root.name,
                "media": {
                    "HEAD_RGB": "observation.images.head_rgb",
                },
                "primary_media": "HEAD_RGB",
            },
        ),
        config_dir=tmp_path,
    )

    assert [item.item_id for item in provider.discover()] == [
        "episode_000001",
        "episode_000002",
    ]


def test_wa2_sha256_fingerprint_tracks_media_content(tmp_path: Path) -> None:
    root = _wa2_dataset(tmp_path)
    config = Wa2SourceConfig(
        dataset_root=root,
        media={"HEAD_RGB": "observation.images.head_rgb"},
        fingerprint_mode="sha256",
    )
    source = Wa2Source(config, config_dir=tmp_path)
    before = source.discover()[0].input_fingerprint

    (root / "videos/chunk-000/head/episode_000001.mp4").write_bytes(
        b"changed-head-1"
    )

    after = source.discover()[0].input_fingerprint
    assert after != before


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.mp4", "/tmp/outside.mp4"],
)
def test_wa2_source_rejects_media_paths_outside_dataset(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_episodes(
        root,
        [_episode(0, head=unsafe_path, wrist=unsafe_path)],
    )

    with pytest.raises(ValueError, match="escapes dataset root"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={"HEAD_RGB": "observation.images.head_rgb"},
            ),
            config_dir=tmp_path,
        ).discover()


def test_wa2_source_rejects_symlink_media_escape(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = root / "videos/episode_000000.mp4"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    _write_episodes(
        root,
        [
            _episode(
                0,
                head="videos/episode_000000.mp4",
                wrist="videos/episode_000000.mp4",
            )
        ],
    )

    with pytest.raises(ValueError, match="escapes dataset root"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={"HEAD_RGB": "observation.images.head_rgb"},
            ),
            config_dir=tmp_path,
        ).discover()


def test_wa2_source_rejects_duplicate_episode_indices(tmp_path: Path) -> None:
    root = _wa2_dataset(tmp_path)
    row = _episode(
        1,
        head="videos/chunk-000/head/episode_000001.mp4",
        wrist="videos/chunk-000/wrist/episode_000001.mp4",
    )
    _write_episodes(root, [row, row])

    with pytest.raises(ValueError, match="duplicate episode_index"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={"HEAD_RGB": "observation.images.head_rgb"},
            ),
            config_dir=tmp_path,
        ).discover()


def test_wa2_source_rejects_reused_media_across_items(tmp_path: Path) -> None:
    root = _wa2_dataset(tmp_path)
    repeated = "videos/chunk-000/head/episode_000001.mp4"
    _write_episodes(
        root,
        [
            _episode(1, head=repeated, wrist=repeated),
            _episode(2, head=repeated, wrist=repeated),
        ],
    )

    with pytest.raises(ValueError, match="reused media path"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={"HEAD_RGB": "observation.images.head_rgb"},
            ),
            config_dir=tmp_path,
        ).discover()


def test_wa2_source_rejects_duplicate_selected_media_within_item(
    tmp_path: Path,
) -> None:
    root = _wa2_dataset(tmp_path)

    with pytest.raises(ValueError, match="duplicate selected media"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={
                    "HEAD_RGB": "observation.images.head_rgb",
                    "ALSO_HEAD": "observation.images.head_rgb",
                },
            ),
            config_dir=tmp_path,
        ).discover()


def test_wa2_source_rejects_missing_configured_media_key(
    tmp_path: Path,
) -> None:
    root = _wa2_dataset(tmp_path)

    with pytest.raises(ValueError, match="missing configured video key"):
        Wa2Source(
            Wa2SourceConfig(
                dataset_root=root,
                media={"SIDE_RGB": "observation.images.side_rgb"},
            ),
            config_dir=tmp_path,
        ).discover()
