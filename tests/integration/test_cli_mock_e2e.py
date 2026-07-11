import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from auto_annotation.cli import main
from auto_annotation.pipeline import runner as runner_module


FIXTURES = Path(__file__).parents[1] / "fixtures"
MISSING_MEDIA_TOOLS = [
    tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None
]
requires_media_tools = pytest.mark.skipif(
    bool(MISSING_MEDIA_TOOLS),
    reason=f"required media tools are missing: {', '.join(MISSING_MEDIA_TOOLS)}",
)


def _make_video(tmp_path: Path, color: str = "black") -> Path:
    video = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x240:d=1:r=30",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


def _make_placeholder_video(tmp_path: Path) -> Path:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder video bytes")
    return video


def _write_config(
    tmp_path: Path,
    manifest: Path,
    fixture_path: Path = FIXTURES / "mock/annotation.json",
    *,
    schema_path: Path = FIXTURES / "schema/subtask-v1.yaml",
    ontology_path: Path = FIXTURES / "ontology/kitchen-v1.yaml",
    artifact_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    artifact_root = artifact_root or tmp_path / "artifacts"
    output_path = output_path or tmp_path / "output.jsonl"
    config = tmp_path / "run.yaml"
    config.write_text(
        f"""
schema_path: {schema_path}
ontology_path: {ontology_path}
manifest_path: {manifest}
artifact_root: {artifact_root}
output_path: {output_path}
backend:
  kind: mock
  fixture_path: {fixture_path}
""".strip(),
        encoding="utf-8",
    )
    return config


def _manifest_record(
    video: Path,
    *,
    video_id: str = "video-1",
    dataset_id: str = "demo",
) -> dict[str, str]:
    return {
        "video_id": video_id,
        "video_uri": str(video),
        "dataset_id": dataset_id,
        "schema_version": "subtask-v1",
        "ontology_id": "kitchen-v1",
    }


def _write_manifest(path: Path, records: list[dict[str, str]]) -> bytes:
    data = (
        "\n".join(json.dumps(record) for record in records) + "\n"
    ).encode()
    path.write_bytes(data)
    return data


def _output_identity(tmp_path: Path) -> tuple[str, bytes]:
    output = (tmp_path / "output.jsonl").read_bytes()
    document = json.loads(output)
    digest = document["provenance"]["config_hash"]
    run_id = f"run-{digest.removeprefix('sha256:')}"
    assert len(run_id) == len("run-") + 64
    assert (tmp_path / "artifacts" / run_id).is_dir()
    return digest, output


@requires_media_tools
def test_cli_runs_mock_pipeline_on_real_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = _make_video(tmp_path)
    manifest = tmp_path / "input.jsonl"
    manifest_record = _manifest_record(video)
    _write_manifest(manifest, [manifest_record])
    config = _write_config(tmp_path, manifest)

    real_asyncio_run = asyncio.run
    event_loop_runs = 0

    def counting_asyncio_run(coro: object) -> object:
        nonlocal event_loop_runs
        event_loop_runs += 1
        return real_asyncio_run(coro)

    monkeypatch.setattr(asyncio, "run", counting_asyncio_run)

    exit_code = main(["run", "--config", str(config)])

    assert exit_code == 0
    assert event_loop_runs == 1
    output = tmp_path / "output.jsonl"
    first_output = output.read_bytes()
    result = json.loads(first_output)
    assert result["task_summary"] == "将杯子放到托盘"
    assert result["segments"][0]["start_ms"] == 100
    assert result["segments"][0]["end_ms"] == 800
    assert result["segments"][0]["values"]["atomic_action"] == "grasp"
    assert result["provenance"]["model_id"] == "mock/video-annotator"

    run_directories = list((tmp_path / "artifacts").iterdir())
    assert len(run_directories) == 1
    video_artifacts = run_directories[0] / "video-1"
    sample_plan_path = video_artifacts / "coarse-sample-plan.json"
    coarse_path = video_artifacts / "coarse.json"
    refined_path = video_artifacts / "refined.json"
    final_path = video_artifacts / "final.json"
    assert sample_plan_path.is_file()
    assert coarse_path.is_file()
    assert refined_path.is_file()
    assert final_path.is_file()

    sample_plan = json.loads(sample_plan_path.read_bytes())
    coarse = json.loads(coarse_path.read_bytes())
    refined = json.loads(refined_path.read_bytes())
    final = json.loads(final_path.read_bytes())
    assert sample_plan["chunk"] == {
        "chunk_id": "chunk-0000",
        "start_ms": 0,
        "end_ms": 1000,
    }
    assert (coarse["segments"][0]["start_ms"], coarse["segments"][0]["end_ms"]) == (
        0,
        900,
    )
    assert (refined[0]["start_ms"], refined[0]["end_ms"]) == (100, 800)
    assert refined[0]["evidence_timestamps_ms"] == [100, 200, 700, 800]
    assert final == result

    assert main(["run", "--config", str(config)]) == 0
    assert event_loop_runs == 2
    assert output.read_bytes() == first_output
    assert len(list((tmp_path / "artifacts").iterdir())) == 1

    manifest_record["schema_version"] = "wrong-schema"
    manifest.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest schema mismatch"):
        main(["run", "--config", str(config)])

    manifest_record["schema_version"] = "subtask-v1"
    manifest_record["ontology_id"] = "wrong-ontology"
    manifest.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest ontology mismatch"):
        main(["run", "--config", str(config)])


@requires_media_tools
def test_run_identity_tracks_content_and_every_stage_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _make_video(tmp_path)
    fixture = tmp_path / "annotation.json"
    fixture_bytes = (FIXTURES / "mock/annotation.json").read_bytes()
    fixture.write_bytes(fixture_bytes)
    manifest = tmp_path / "input.jsonl"
    manifest_bytes = _write_manifest(manifest, [_manifest_record(video)])
    config = _write_config(tmp_path, manifest, fixture)

    assert main(["run", "--config", str(config)]) == 0
    baseline_digest, baseline_output = _output_identity(tmp_path)
    seen_digests = {baseline_digest}

    assert main(["run", "--config", str(config)]) == 0
    repeated_digest, repeated_output = _output_identity(tmp_path)
    assert repeated_digest == baseline_digest
    assert repeated_output == baseline_output
    assert len(list((tmp_path / "artifacts").iterdir())) == 1

    fixture.write_bytes(fixture_bytes + b"\n")
    assert main(["run", "--config", str(config)]) == 0
    fixture_digest, _ = _output_identity(tmp_path)
    assert fixture_digest not in seen_digests
    seen_digests.add(fixture_digest)

    fixture.write_bytes(fixture_bytes)
    manifest.write_bytes(manifest_bytes + b"\n")
    assert main(["run", "--config", str(config)]) == 0
    manifest_digest, _ = _output_identity(tmp_path)
    assert manifest_digest not in seen_digests
    seen_digests.add(manifest_digest)

    manifest.write_bytes(manifest_bytes)
    _make_video(tmp_path, color="white")
    assert main(["run", "--config", str(config)]) == 0
    media_digest, _ = _output_identity(tmp_path)
    assert media_digest not in seen_digests
    seen_digests.add(media_digest)

    for version_name in (
        "COARSE_STAGE_VERSION",
        "REFINE_STAGE_VERSION",
        "FINALIZE_STAGE_VERSION",
        "RUNNER_STAGE_VERSION",
    ):
        current_version = getattr(runner_module, version_name)
        monkeypatch.setattr(
            runner_module,
            version_name,
            f"{current_version}-changed",
        )
        assert main(["run", "--config", str(config)]) == 0
        changed_digest, _ = _output_identity(tmp_path)
        assert changed_digest not in seen_digests
        seen_digests.add(changed_digest)

    assert len(list((tmp_path / "artifacts").iterdir())) == len(seen_digests)


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "message"),
    [
        ("schema_version", "wrong-schema", "manifest schema mismatch"),
        ("ontology_id", "wrong-ontology", "manifest ontology mismatch"),
    ],
)
def test_runner_prevalidates_the_entire_manifest_before_side_effects(
    tmp_path: Path,
    invalid_field: str,
    invalid_value: str,
    message: str,
) -> None:
    video = _make_placeholder_video(tmp_path)
    valid = _manifest_record(video)
    invalid = {**valid, "video_id": "video-2", invalid_field: invalid_value}
    manifest = tmp_path / "input.jsonl"
    _write_manifest(manifest, [valid, invalid])
    config = _write_config(tmp_path, manifest, tmp_path / "missing-script.json")

    with pytest.raises(ValueError, match=message):
        main(["run", "--config", str(config)])

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "output.jsonl").exists()


def test_runner_rejects_mixed_dataset_ids_before_side_effects(
    tmp_path: Path,
) -> None:
    video = _make_placeholder_video(tmp_path)
    record = _manifest_record(video, dataset_id="demo-a")
    manifest = tmp_path / "input.jsonl"
    _write_manifest(
        manifest,
        [record, {**record, "dataset_id": "demo-b"}],
    )
    config = _write_config(tmp_path, manifest, tmp_path / "missing-script.json")

    with pytest.raises(ValueError, match="mixed dataset_id"):
        main(["run", "--config", str(config)])

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "output.jsonl").exists()


def test_runner_rejects_empty_manifest_before_side_effects(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("\n", encoding="utf-8")
    config = _write_config(tmp_path, manifest, tmp_path / "missing-script.json")

    with pytest.raises(ValueError, match="manifest must contain at least one item"):
        main(["run", "--config", str(config)])

    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "output.jsonl").exists()


@pytest.mark.parametrize(
    "unsafe_video_id",
    [".", "..", "nested/video", "nested\\video"],
)
def test_runner_rejects_unsafe_video_id_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_video_id: str,
) -> None:
    video = _make_placeholder_video(tmp_path)
    manifest = tmp_path / "input.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_record(video),
            _manifest_record(video, video_id=unsafe_video_id),
        ],
    )
    fixture = tmp_path / "annotation.json"
    fixture.write_bytes((FIXTURES / "mock/annotation.json").read_bytes())
    config = _write_config(tmp_path, manifest, fixture)

    fixture_reads = 0
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def tracking_read_bytes(path: Path) -> bytes:
        nonlocal fixture_reads
        if path == fixture:
            fixture_reads += 1
        return real_read_bytes(path)

    def tracking_read_text(path: Path, *args: object, **kwargs: object) -> str:
        nonlocal fixture_reads
        if path == fixture:
            fixture_reads += 1
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
    monkeypatch.setattr(Path, "read_text", tracking_read_text)

    with pytest.raises(ValueError, match="unsafe video_id"):
        main(["run", "--config", str(config)])

    assert fixture_reads == 0
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "output.jsonl").exists()


@pytest.mark.parametrize(
    ("input_name", "alias_kind"),
    [
        ("manifest", "exact"),
        ("schema", "exact"),
        ("ontology", "exact"),
        ("fixture", "exact"),
        ("video", "exact"),
        ("manifest", "normalized"),
        ("fixture", "hardlink"),
    ],
)
def test_runner_rejects_output_path_aliases_before_side_effects(
    tmp_path: Path,
    input_name: str,
    alias_kind: str,
) -> None:
    schema = tmp_path / "schema.yaml"
    schema.write_bytes((FIXTURES / "schema/subtask-v1.yaml").read_bytes())
    ontology = tmp_path / "ontology.yaml"
    ontology.write_bytes((FIXTURES / "ontology/kitchen-v1.yaml").read_bytes())
    fixture = tmp_path / "annotation.json"
    fixture.write_bytes((FIXTURES / "mock/annotation.json").read_bytes())
    video = _make_placeholder_video(tmp_path)
    manifest = tmp_path / "input.jsonl"
    _write_manifest(manifest, [_manifest_record(video)])
    inputs = {
        "manifest": manifest,
        "schema": schema,
        "ontology": ontology,
        "fixture": fixture,
        "video": video,
    }
    target = inputs[input_name]
    original_bytes = target.read_bytes()

    if alias_kind == "exact":
        output = target
    elif alias_kind == "normalized":
        alias_parent = tmp_path / "alias-parent"
        alias_parent.mkdir()
        output = alias_parent / ".." / target.name
    else:
        output = tmp_path / "output-hardlink.jsonl"
        os.link(target, output)

    config = _write_config(
        tmp_path,
        manifest,
        fixture,
        schema_path=schema,
        ontology_path=ontology,
        output_path=output,
    )

    with pytest.raises(ValueError, match="output_path aliases"):
        main(["run", "--config", str(config)])

    assert target.read_bytes() == original_bytes
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    "alias_kind", ["exact", "normalized", "symlink", "hardlink"]
)
def test_runner_rejects_config_output_alias_before_downstream_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    config_path = tmp_path / "run.yaml"
    if alias_kind == "exact":
        output = config_path
    elif alias_kind == "normalized":
        alias_parent = tmp_path / "alias-parent"
        alias_parent.mkdir()
        output = alias_parent / ".." / config_path.name
    elif alias_kind == "symlink":
        output = tmp_path / "output-symlink.yaml"
    else:
        output = tmp_path / "output-hardlink.jsonl"

    manifest = tmp_path / "input.jsonl"
    config = _write_config(
        tmp_path,
        manifest,
        tmp_path / "missing-script.json",
        output_path=output,
    )
    if alias_kind == "symlink":
        output.symlink_to(config)
    elif alias_kind == "hardlink":
        os.link(config, output)
    original_bytes = config.read_bytes()
    downstream_calls: list[str] = []

    def unexpected_contract_load(*args: object, **kwargs: object) -> object:
        downstream_calls.append("load_contract")
        raise AssertionError("contract loading must not run")

    monkeypatch.setattr(runner_module, "load_contract", unexpected_contract_load)

    with pytest.raises(ValueError, match="output_path aliases config source"):
        main(["run", "--config", str(config)])

    assert downstream_calls == []
    assert config.read_bytes() == original_bytes
    assert output.read_bytes() == original_bytes
    assert not (tmp_path / "artifacts").exists()
