---
name: auto-annotation-run-and-test
description: Use when asked to “set up the project,” “run auto-annotate,” “write a run config,” “select pytest coverage,” “inspect artifacts,” or diagnose FFmpeg, mock-backend, timeline, path-safety, and deterministic-run failures.
---

# Auto Annotation Run and Test

## Overview

Use this skill for the executable mock-backend core slice. It covers local setup, CLI inputs, generated files, focused tests, and failure routing. It does not describe a real vLLM deployment because the root package has no real vLLM adapter yet.

## Path Convention

This skill lives at `.agents/skills/auto-annotation-run-and-test/SKILL.md`; the repository root is `../../../` relative to this directory. Run commands from the repository root.

## Where to Find Answers

| Need | Source |
| --- | --- |
| Python/dependency/entry-point declarations | `pyproject.toml`, `uv.lock` |
| CLI arguments | `src/auto_annotation/cli.py` |
| Run config fields/defaults | `src/auto_annotation/config/models.py` |
| Complete run behavior and artifact stage names | `src/auto_annotation/pipeline/runner.py` |
| Minimal Schema/Ontology/mock shapes | `tests/fixtures/schema/subtask-v1.yaml`, `tests/fixtures/ontology/kitchen-v1.yaml`, `tests/fixtures/mock/annotation.json` |
| Executable config/manifest construction | `tests/integration/test_cli_mock_e2e.py` |
| FFprobe command and parsing | `src/auto_annotation/media/probe.py` |
| Atomic artifact/export behavior | `src/auto_annotation/artifacts/store.py`, `src/auto_annotation/exporters/jsonl.py` |

## Prerequisites and Setup

- Python `>=3.12,<3.14`.
- `uv` for the locked environment.
- `ffprobe` for every real CLI run.
- Both `ffmpeg` and `ffprobe` for the real-video integration test because the test generates a temporary video.

Install the package and test extra:

```bash
uv sync --extra test
```

No lint, formatter, type checker, build verification, or Markdown checker is configured in this repository.

## Run the CLI

Create a YAML config using the currently supported fields:

```yaml
schema_path: schema.yaml
ontology_path: ontology.yaml
manifest_path: input.jsonl
artifact_root: artifacts
output_path: output.jsonl
backend:
  kind: mock
  fixture_path: mock-responses.json
sampling:
  coarse_fps: 2.0
  refine_fps: 6.0
  refine_window_ms: 4000
timeline:
  merge_gap_ms: 250
  min_segment_ms: 300
```

Paths above resolve from the config directory. Sampling and timeline sections may be omitted to use those defaults.

The manifest is JSONL. A minimal record is:

```json
{"video_id":"video-1","video_uri":"video.mp4","dataset_id":"demo","schema_version":"subtask-v1","ontology_id":"kitchen-v1"}
```

`video_uri` resolves from the manifest directory. It must be a local existing file. The loaded Schema/Ontology IDs must match each manifest record.

The mock fixture maps generated request IDs to response queues. Expect one `coarse:<video_id>` response followed by `boundary-start:<video_id>:<segment_id>` and `boundary-end:<video_id>:<segment_id>` responses for every coarse segment. Use `tests/fixtures/mock/annotation.json` only as an executable shape example.

Run:

```bash
uv run auto-annotate run --config path/to/run.yaml
```

The command returns exit code 0 on success. It raises validation/runtime errors directly; there is no CLI error translation or retry layer yet.

## Inspect Outputs

- Final JSONL is written to `output_path`, sorted by `video_id` with sorted JSON keys.
- Intermediate JSON is written under `<artifact_root>/<run_id>/<video_id>/`.
- Current stage files are `coarse-sample-plan.json`, `coarse.json`, `refined.json`, and `final.json`.
- `run_id` is the deterministic `run-` prefix plus the 64-hex content/config digest stored as `provenance.config_hash`.
- Identical inputs produce identical output bytes and reuse the same run directory; content or behavior-version changes produce a different identity.

## Test Selection

| Scope | Command |
| --- | --- |
| Full suite | `uv run pytest -q` |
| Repository layout | `uv run pytest tests/test_repository_layout.py -q` |
| Config | `uv run pytest tests/config/test_loader.py -q` |
| Domain and manifest | `uv run pytest tests/domain -q` |
| Media probe/chunk/sample | `uv run pytest tests/media/test_probe_sampling.py -q` |
| Schema/Ontology | `uv run pytest tests/ontology/test_registry.py -q` |
| Backend protocol/mock | `uv run pytest tests/inference/test_mock_backend.py -q` |
| Prompts/response schemas | `uv run pytest tests/prompts/test_builders.py -q` |
| Coarse/refine stages | `uv run pytest tests/pipeline/test_stages.py -q` |
| Finalization/quality | `uv run pytest tests/pipeline/test_finalize.py -q` |
| Artifacts/export | `uv run pytest tests/artifacts/test_store_export.py -q` |
| Full CLI with real media | `uv run pytest tests/integration/test_cli_mock_e2e.py -q` |

Run the narrowest applicable module first, then the integration module when the change crosses package boundaries. The integration module skips real-media cases only when `ffmpeg` or `ffprobe` is missing.

## Failure Routing

| Symptom | Check first |
| --- | --- |
| Unknown/missing YAML keys or invalid positive values | `config/models.py`, then `tests/config/test_loader.py` |
| Missing/remote video, duplicate manifest entry | `manifest.py`, then `tests/domain/test_manifest.py` |
| Mixed dataset, unsafe `video_id`, Schema/Ontology mismatch | Runner prevalidation and CLI integration tests |
| Output aliases an input, or any artifact path escapes root | `pipeline/runner.py`, `artifacts/store.py`, artifact/integration tests |
| `ffprobe failed`, invalid duration/FPS/PTS | Run the recorded `ffprobe` path manually, then inspect `media/probe.py` tests |
| Sampling rate/count or empty-chunk error | `media/sampling.py`; maximum FPS is 240 and maximum targets are 100,000 |
| `core slice accepts exactly one short-video chunk` | Input exceeded the runner's 120,000 ms single-chunk boundary; long-video merge is not implemented |
| `mock script exhausted` | Fixture request IDs/order do not match coarse and boundary calls |
| JSON Schema or strict Pydantic response error | Fixture/backend output violates the generated request contract |
| Segment-ID mismatch or timeline conflict | Compare coarse/refined IDs, values, sentence, and boundaries in stage artifacts |
| Unexpected new run directory or digest | Compare config, raw manifest/fixture bytes, video content, Schema/Ontology, prompt/stage versions, and backend identity |

## Safety and Reproducibility Checks

- Keep output separate from the config, manifest, Schema, Ontology, fixture, and videos; aliases through symlinks are rejected.
- Artifact path components cannot be empty, dot paths, or contain separators. Symlink ancestors below the artifact root are rejected.
- Artifact and final-output writes use fsync plus atomic replacement and remove temporary files on failure.
- Do not compare only JSON meaning when debugging run identity: manifest and mock-fixture raw bytes are hashed.

## Related Skills

- `.agents/skills/auto-annotation-codebase-nav/SKILL.md`
- `.agents/skills/auto-annotation-contracts/SKILL.md`
