# Auto Annotation

Read this first, then open the repository skill that matches the task before changing behavior.

## Overview

This Python package implements a deterministic core slice for video subtask annotation. The current CLI reads a YAML run config and JSONL manifest, probes local video with `ffprobe`, builds PTS-aware samples, runs coarse annotation and boundary refinement through a scripted mock backend, finalizes the timeline, writes per-stage artifacts, and exports JSONL.

The broader real-vLLM, orchestration, retry, long-video merge, Parquet, and evaluation architecture is documented but not implemented. `thirdparty/vllm-main/` is a vendored source snapshot with its own `AGENTS.md`; do not modify it for root-package work.

## Commands

| Task | Command |
| --- | --- |
| Install project and test dependencies | `uv sync --extra test` |
| Run all tests | `uv run pytest -q` |
| Run one subsystem | `uv run pytest tests/pipeline/test_stages.py -q` |
| Run one test | `uv run pytest tests/integration/test_cli_mock_e2e.py::test_cli_runs_mock_pipeline_on_real_video -q` |
| Run the CLI | `uv run auto-annotate run --config path/to/run.yaml` |

No repository lint, format, static-type, or Markdown-check command is currently configured.

## Repository Rules

- Support Python `>=3.12,<3.14`; keep runtime dependencies aligned with `pyproject.toml`.
- Preserve the typed boundaries between config, domain, media, ontology, inference, prompts, pipeline, artifacts, and exporters.
- Use absolute integer milliseconds internally and half-open intervals `[start_ms, end_ms)`. Final segments may have gaps but must be sorted and non-overlapping.
- Keep segment values driven by Schema and Ontology. Do not hardcode dataset vocabulary IDs in pipeline logic.
- Treat backend output as untrusted: retain JSON Schema validation and strict Pydantic validation at stage boundaries.
- Preserve prevalidation before output side effects, deterministic serialization, atomic replacement, and artifact path/symlink protections.
- When stage or prompt behavior changes, review the corresponding version constant and run-identity tests; provenance must identify behavior and content inputs.
- Keep root-package changes out of `thirdparty/vllm-main/` unless the task explicitly targets the vendored project, then follow its instructions.

## Key Locations

| Area | Paths |
| --- | --- |
| CLI and orchestration | `src/auto_annotation/cli.py`, `src/auto_annotation/pipeline/runner.py` |
| Run configuration | `src/auto_annotation/config/` |
| Manifest and domain models | `src/auto_annotation/manifest.py`, `src/auto_annotation/domain/` |
| Video probing and sampling | `src/auto_annotation/media/` |
| Schema and Ontology contracts | `src/auto_annotation/ontology/` |
| Inference protocol and mock | `src/auto_annotation/inference/` |
| Prompt and response-schema builders | `src/auto_annotation/prompts/` |
| Pipeline stages | `src/auto_annotation/pipeline/` |
| Atomic artifacts and JSONL | `src/auto_annotation/artifacts/`, `src/auto_annotation/exporters/` |
| Tests and executable examples | `tests/`, especially `tests/integration/test_cli_mock_e2e.py` |

## Documentation

| Document | Use |
| --- | --- |
| `docs/superpowers/specs/2026-07-10-vllm-subtask-auto-annotation-design.md` | Approved architecture and future scope; verify claims against code before treating them as implemented |
| `docs/superpowers/plans/2026-07-10-vllm-subtask-core-inference.md` | Historical implementation plan for the current core slice |

## Common Tasks and Skills

- Use `.agents/skills/auto-annotation-codebase-nav/SKILL.md` to trace ownership and the end-to-end data flow.
- Use `.agents/skills/auto-annotation-contracts/SKILL.md` for run config, manifest, Schema, Ontology, output, or provenance changes.
- Use `.agents/skills/auto-annotation-run-and-test/SKILL.md` to set up, run the CLI, select tests, inspect artifacts, or diagnose failures.

## Gotchas

- The only configured backend is `kind: mock`; no real vLLM transport exists in the root package.
- `build_chunks` can describe multiple chunks, but the current runner rejects videos that produce anything other than one chunk; its limit is 120,000 ms.
- Config-relative paths resolve from the run-config directory. Manifest `video_uri` paths resolve from the manifest directory and must name existing local files.
- A run accepts one non-empty dataset, safe unique `video_id` values, and manifest Schema/Ontology IDs matching the loaded contract.
- `ffmpeg` and `ffprobe` are required by the real-video integration test; that test is skipped when either executable is unavailable.
- Runtime `/artifacts/` is ignored, while `src/auto_annotation/artifacts/` and `tests/artifacts/` are source and must remain visible.
