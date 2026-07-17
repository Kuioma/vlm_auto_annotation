import json
import sys
from pathlib import Path

import yaml

from auto_annotation.cli import build_parser, main


def _write_wa2_dataset(
    tmp_path: Path,
    *,
    episode_count: int = 1,
) -> Path:
    dataset_root = tmp_path / "dataset"
    episodes = dataset_root / "meta/episodes.jsonl"
    episodes.parent.mkdir(parents=True)
    rows: list[str] = []
    for episode_index in range(episode_count):
        video = (
            dataset_root
            / "videos/chunk-000/observation.images.head_rgb"
            / f"episode_{episode_index:06d}.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{episode_index}".encode())
        rows.append(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": "demo",
                    "length": 1,
                    "videos": {
                        "observation.images.head_rgb": video.relative_to(
                            dataset_root
                        ).as_posix()
                    },
                }
            )
            + "\n"
        )
    episodes.write_text("".join(rows), encoding="utf-8")
    return dataset_root


def test_cli_exposes_config_driven_batch_entry() -> None:
    args = build_parser().parse_args(
        ["batch", "--config", "batch.yaml", "--plan-only"]
    )

    assert args.command == "batch"
    assert args.config == Path("batch.yaml")
    assert args.plan_only is True
    assert args.quiet is False


def test_cli_plan_loads_source_and_processor_plugins_from_config(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_root = _write_wa2_dataset(tmp_path)
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "source": {
                    "plugin": "auto_annotation.batch.sources.wa2:PLUGIN",
                    "config": {
                        "dataset_root": str(dataset_root),
                        "media": {
                            "HEAD_RGB": "observation.images.head_rgb"
                        },
                        "fingerprint_mode": "stat",
                    },
                },
                "processor": {
                    "plugin": (
                        "auto_annotation.batch.processors.command:PLUGIN"
                    ),
                    "config": {
                        "semantic_version": "test-v1",
                        "argv": ["unused-plan-command"],
                    },
                },
                "output": {"workspace": "workspace"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        ["batch", "--config", str(config_path), "--plan-only"]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["discovered"] == 1
    assert summary["selected"] == 1
    assert summary["pending"] == 1
    assert summary["plan_only"] is True
    assert not (tmp_path / "workspace").exists()


def test_cli_executes_configured_plugins_and_writes_audited_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_root = _write_wa2_dataset(tmp_path)
    processor_script = tmp_path / "processor.py"
    processor_script.write_text(
        """
import json
import pathlib
import sys

item = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
output_dir = pathlib.Path(sys.argv[2])
(output_dir / "output.jsonl").write_text(
    json.dumps({"item_id": item["item_id"], "task": item["context"]["task"]})
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "source": {
                    "plugin": "auto_annotation.batch.sources.wa2:PLUGIN",
                    "config": {
                        "dataset_root": str(dataset_root),
                        "media": {
                            "HEAD_RGB": "observation.images.head_rgb"
                        },
                        "fingerprint_mode": "stat",
                    },
                },
                "processor": {
                    "plugin": (
                        "auto_annotation.batch.processors.command:PLUGIN"
                    ),
                    "config": {
                        "semantic_version": "cli-e2e-v1",
                        "argv": [
                            sys.executable,
                            str(processor_script),
                            "{item_json}",
                            "{output_dir}",
                        ],
                        "identity_files": [str(processor_script)],
                    },
                },
                "output": {"workspace": "workspace"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["batch", "--config", str(config_path)])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["succeeded"] == 1
    assert summary["invocation_id"] == "invocation-000001"
    assert "[1/1] episode_000000: attempt 1 started" in captured.err
    assert "[1/1] episode_000000: succeeded" in captured.err
    assert Path(summary["output_path"]).read_text(encoding="utf-8") == (
        '{"item_id":"episode_000000","task":"demo"}\n'
    )
    output_index = json.loads(
        Path(summary["output_index_path"])
        .read_text(encoding="utf-8")
        .strip()
    )
    assert output_index["item_id"] == "episode_000000"
    assert Path(summary["invocations_path"]).is_file()


def test_cli_runs_command_processor_items_concurrently(
    tmp_path: Path,
    capsys,
) -> None:
    dataset_root = _write_wa2_dataset(tmp_path, episode_count=2)
    barrier_dir = tmp_path / "barrier"
    barrier_dir.mkdir()
    processor_script = tmp_path / "concurrent_processor.py"
    processor_script.write_text(
        """
import json
import pathlib
import sys
import time

item = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
output_dir = pathlib.Path(sys.argv[2])
barrier_dir = pathlib.Path(sys.argv[3])
(barrier_dir / f"{item['item_id']}.started").write_text(
    "started\\n",
    encoding="utf-8",
)
deadline = time.monotonic() + 5.0
while len(tuple(barrier_dir.glob("*.started"))) < 2:
    if time.monotonic() >= deadline:
        print("timed out waiting for concurrent item", file=sys.stderr)
        raise SystemExit(23)
    time.sleep(0.02)
(output_dir / "output.jsonl").write_text(
    json.dumps({"item_id": item["item_id"], "task": item["context"]["task"]})
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "source": {
                    "plugin": "auto_annotation.batch.sources.wa2:PLUGIN",
                    "config": {
                        "dataset_root": str(dataset_root),
                        "media": {
                            "HEAD_RGB": "observation.images.head_rgb"
                        },
                        "fingerprint_mode": "stat",
                    },
                },
                "processor": {
                    "plugin": (
                        "auto_annotation.batch.processors.command:PLUGIN"
                    ),
                    "config": {
                        "semantic_version": "cli-concurrency-e2e-v1",
                        "argv": [
                            sys.executable,
                            str(processor_script),
                            "{item_json}",
                            "{output_dir}",
                            str(barrier_dir),
                        ],
                        "timeout_s": 10,
                        "identity_files": [str(processor_script)],
                    },
                },
                "execution": {
                    "concurrency": 2,
                    "max_attempts": 1,
                    "continue_on_error": True,
                },
                "output": {"workspace": "workspace"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["batch", "--config", str(config_path)])

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert exit_code == 0
    assert summary["selected"] == 2
    assert summary["attempted"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert set(path.name for path in barrier_dir.glob("*.started")) == {
        "episode_000000.started",
        "episode_000001.started",
    }
    for position, item_id in enumerate(
        ("episode_000000", "episode_000001"),
        start=1,
    ):
        assert (
            f"[{position}/2] {item_id}: attempt 1 started" in captured.err
        )
        assert (
            f"[{position}/2] {item_id}: succeeded on attempt 1"
            in captured.err
        )

    output = [
        json.loads(line)
        for line in Path(summary["output_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert output == [
        {"item_id": "episode_000000", "task": "demo"},
        {"item_id": "episode_000001", "task": "demo"},
    ]
    output_index = [
        json.loads(line)
        for line in Path(summary["output_index_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["line_number"] for row in output_index] == [1, 2]
    assert [row["item_id"] for row in output_index] == [
        "episode_000000",
        "episode_000001",
    ]

    history = json.loads(
        Path(summary["invocations_path"]).read_text(encoding="utf-8")
    )
    assert history["version"] == 1
    assert len(history["invocations"]) == 1
    invocation = history["invocations"][0]
    assert invocation["status"] == "completed"
    assert invocation["execution"]["concurrency"] == 2
    assert invocation["selected_item_ids"] == [
        "episode_000000",
        "episode_000001",
    ]
    assert invocation["summary"]["attempted"] == 2
    assert invocation["summary"]["succeeded"] == 2
