#!/usr/bin/env python3
"""Run one or all WA2 head-camera videos through local vLLM."""

import argparse
import json
from pathlib import Path

import yaml

from auto_annotation.cli import main as cli_main


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path(
    "/home/cz-sjj/ws/dataset/WA2_industrial_sorting_3cam_gripper_cam/"
    "0622_333_20260624_093035_converted_online_clean_20260624_194551"
)
DEFAULT_VIDEO = Path(
    DEFAULT_DATASET_ROOT
    / "videos/chunk-000/observation.images.head_rgb/episode_000000.mp4"
)
HEAD_RGB_GLOB = "videos/chunk-*/observation.images.head_rgb/episode_*.mp4"


def discover_head_rgb_videos(
    dataset_root: Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")
    root = dataset_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    videos = sorted(
        (path.resolve(strict=True) for path in root.glob(HEAD_RGB_GLOB)),
        key=lambda path: (path.stem, path.as_posix()),
    )
    if not videos:
        raise ValueError(f"no head_rgb episode videos found under: {root}")
    if limit is not None:
        videos = videos[:limit]
    video_ids = [f"{video.stem}_head_rgb" for video in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("duplicate head_rgb episode IDs across video chunks")
    return videos


def default_dataset_workspace(dataset_root: Path) -> Path:
    meta_root = dataset_root.expanduser().resolve(strict=True) / "meta"
    meta_root = meta_root.resolve(strict=True)
    if not meta_root.is_dir():
        raise ValueError(f"dataset meta path is not a directory: {meta_root}")
    return meta_root / "auto_annotation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--video",
        type=Path,
        help="annotate one video; defaults to episode_000000 head_rgb",
    )
    source.add_argument(
        "--dataset-root",
        type=Path,
        help="annotate every head_rgb episode below this dataset root",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help=(
            "generated config and output directory; dataset runs default to "
            "<dataset-root>/meta/auto_annotation"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N episodes from --dataset-root",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--model-revision",
        default="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    )
    parser.add_argument("--vllm-version", default="0.24.0")
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--coarse-fps", type=float, default=2.0)
    parser.add_argument("--refine-fps", type=float, default=2.0)
    parser.add_argument(
        "--max-overlap-resolution-ms",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--boundary-max-completion-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--media-staging-root",
        type=Path,
        default=Path("/home/cz-sjj/ws/vllm/media"),
    )
    parser.add_argument("--api-key-env")
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    dataset_root: Path | None = None
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser().resolve(strict=True)
        videos = discover_head_rgb_videos(dataset_root, limit=args.limit)
        default_workspace = default_dataset_workspace(dataset_root)
    else:
        if args.limit is not None:
            raise ValueError("--limit requires --dataset-root")
        video = (args.video or DEFAULT_VIDEO).expanduser().resolve(strict=True)
        if not video.is_file():
            raise ValueError(f"video is not a file: {video}")
        videos = [video]
        default_workspace = Path("/tmp/auto-annotation-vllm-smoke")

    staging_root = args.media_staging_root.expanduser().resolve(strict=True)
    if not staging_root.is_dir():
        raise ValueError(f"media staging root is not a directory: {staging_root}")

    workspace = (args.workspace or default_workspace).expanduser()
    if workspace.is_symlink():
        raise ValueError(f"workspace must not be a symlink: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve(strict=True)

    manifest_path = workspace / "manifest.jsonl"
    config_path = workspace / "run.yaml"
    output_path = workspace / "output.jsonl"
    artifact_root = workspace / "artifacts"

    dataset_id = (
        dataset_root.name
        if dataset_root is not None
        else "wa2-industrial-sorting-smoke"
    )
    manifest_records = []
    for video in videos:
        metadata = {"camera": "head_rgb"}
        if dataset_root is not None:
            metadata["dataset_relative_video_path"] = video.relative_to(
                dataset_root
            ).as_posix()
        manifest_records.append(
            {
                "video_id": f"{video.stem}_head_rgb",
                "video_uri": str(video),
                "dataset_id": dataset_id,
                "schema_version": "industrial-sorting-smoke-v1",
                "ontology_id": "industrial-sorting-smoke-v1",
                "metadata": metadata,
            }
        )
    manifest_path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in manifest_records
        ),
        encoding="utf-8",
    )

    backend = {
        "kind": "vllm",
        "base_url": args.base_url,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "vllm_version": args.vllm_version,
        "timeout_s": args.timeout_s,
        "max_completion_tokens": 4096,
        "boundary_max_completion_tokens": (
            args.boundary_max_completion_tokens
        ),
        "temperature": 0,
        "seed": 0,
        "enable_thinking": False,
        "media_staging_root": str(staging_root),
    }
    if args.api_key_env is not None:
        backend["api_key_env"] = args.api_key_env

    config = {
        "schema_path": str(HERE / "schema.yaml"),
        "ontology_path": str(HERE / "ontology.yaml"),
        "manifest_path": str(manifest_path),
        "artifact_root": str(artifact_root),
        "output_path": str(output_path),
        "backend": backend,
        "sampling": {
            "coarse_fps": args.coarse_fps,
            "refine_fps": args.refine_fps,
            "refine_window_ms": 4000,
        },
        "timeline": {
            "merge_gap_ms": 250,
            "min_segment_ms": 300,
            "max_overlap_resolution_ms": args.max_overlap_resolution_ms,
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    if dataset_root is not None:
        print(f"dataset_root={dataset_root}")
    print(f"episode_count={len(videos)}")
    print(f"first_video={videos[0]}")
    print(f"last_video={videos[-1]}")
    print(f"config={config_path}")
    print(f"artifacts={artifact_root}")
    print(f"output={output_path}")
    print(f"timeout_s={args.timeout_s}")
    print(f"coarse_fps={args.coarse_fps}")
    print(f"refine_fps={args.refine_fps}")
    print(f"max_overlap_resolution_ms={args.max_overlap_resolution_ms}")
    print(
        "boundary_max_completion_tokens="
        f"{args.boundary_max_completion_tokens}"
    )
    try:
        exit_code = cli_main(["run", "--config", str(config_path)])
    except Exception:
        if output_path.is_file():
            completed = sum(
                bool(line.strip())
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
            print(
                f"\nPartial annotations preserved: {output_path} "
                f"({completed} episodes)"
            )
        raise
    if exit_code != 0:
        return exit_code

    output_lines = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"\nCompleted annotations={len(output_lines)}")
    print(f"Final output={output_path}")
    if len(output_lines) == 1:
        print("\nFinal annotation:")
        print(
            json.dumps(
                json.loads(output_lines[0]),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
