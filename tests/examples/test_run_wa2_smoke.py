import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.vllm_smoke.run_wa2_smoke import (
    default_dataset_workspace,
    discover_head_rgb_videos,
)


def _write_video(dataset_root: Path, chunk: str, name: str) -> Path:
    video = (
        dataset_root
        / "videos"
        / chunk
        / "observation.images.head_rgb"
        / name
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    return video


def test_discovers_head_rgb_episodes_in_stable_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    expected = [
        _write_video(dataset_root, "chunk-000", "episode_000000.mp4"),
        _write_video(dataset_root, "chunk-001", "episode_000001.mp4"),
        _write_video(dataset_root, "chunk-000", "episode_000002.mp4"),
    ]
    wrist_video = (
        dataset_root
        / "videos/chunk-000/observation.images.left_wrist_rgb"
        / "episode_000000.mp4"
    )
    wrist_video.parent.mkdir(parents=True)
    wrist_video.write_bytes(b"video")

    discovered = discover_head_rgb_videos(dataset_root)

    assert discovered == [path.resolve() for path in expected]
    assert discover_head_rgb_videos(dataset_root, limit=2) == discovered[:2]


def test_dataset_workspace_is_namespaced_below_meta(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)

    assert default_dataset_workspace(dataset_root) == (
        dataset_root / "meta/auto_annotation"
    ).resolve(strict=False)


def test_dataset_discovery_rejects_invalid_or_empty_input(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        discover_head_rgb_videos(dataset_root, limit=0)
    with pytest.raises(ValueError, match="no head_rgb episode videos"):
        discover_head_rgb_videos(dataset_root)
