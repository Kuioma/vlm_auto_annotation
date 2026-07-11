---
name: auto-annotation-codebase-nav
description: Use when asked to “find the implementation,” “trace the annotation pipeline,” “locate the owner of a behavior,” “map tests to source,” or distinguish current auto-annotation code from planned architecture or vendored vLLM code.
---

# Auto Annotation Codebase Navigation

## Overview

Use this skill to find the smallest authoritative source and its paired tests before making a change. The root package is a layered Python core slice; design documents contain additional future architecture that may not exist in code.

## Path Convention

This skill lives at `.agents/skills/auto-annotation-codebase-nav/SKILL.md`; the repository root is `../../../` relative to this directory. All paths below are repository-relative.

## Where to Find Answers

| Question | Source of truth | Tests |
| --- | --- | --- |
| How is the command parsed? | `src/auto_annotation/cli.py` | `tests/integration/test_cli_mock_e2e.py` |
| How are config paths and hashes handled? | `src/auto_annotation/config/models.py`, `src/auto_annotation/config/loader.py` | `tests/config/test_loader.py` |
| What does a manifest record contain? | `src/auto_annotation/manifest.py`, `src/auto_annotation/domain/models.py` | `tests/domain/test_manifest.py`, `tests/domain/test_models.py` |
| How are video timestamps obtained? | `src/auto_annotation/media/probe.py` | `tests/media/test_probe_sampling.py` |
| How are chunks and samples built? | `src/auto_annotation/media/sampling.py` | `tests/media/test_probe_sampling.py` |
| How do Schema and Ontology become response constraints? | `src/auto_annotation/ontology/models.py`, `src/auto_annotation/ontology/registry.py` | `tests/ontology/test_registry.py` |
| What is the backend boundary? | `src/auto_annotation/inference/base.py`, `src/auto_annotation/inference/mock.py` | `tests/inference/test_mock_backend.py` |
| How are prompts and response schemas assembled? | `src/auto_annotation/prompts/builders.py` | `tests/prompts/test_builders.py` |
| How do coarse and boundary stages work? | `src/auto_annotation/pipeline/coarse.py`, `src/auto_annotation/pipeline/refine.py` | `tests/pipeline/test_stages.py` |
| How are timelines normalized and scored? | `src/auto_annotation/pipeline/finalize.py` | `tests/pipeline/test_finalize.py` |
| How is a complete run coordinated? | `src/auto_annotation/pipeline/runner.py` | `tests/integration/test_cli_mock_e2e.py` |
| How are artifacts and JSONL made deterministic? | `src/auto_annotation/artifacts/store.py`, `src/auto_annotation/exporters/jsonl.py` | `tests/artifacts/test_store_export.py` |
| Why is `/artifacts/` ignored but source packages retained? | `.gitignore` | `tests/test_repository_layout.py` |
| What is planned beyond the core slice? | `docs/superpowers/specs/2026-07-10-vllm-subtask-auto-annotation-design.md` | Verify against current source before relying on it |

## End-to-End Data Flow

1. `cli.main` loads a YAML file with `load_run_config`, then calls `run_annotation`.
2. The runner resolves Schema/Ontology, reads and prevalidates the full manifest, checks output aliases, and builds a content-derived run ID.
3. `probe_video` uses frame PTS from `ffprobe`; `build_chunks` and `build_sample_plan` create timestamp-aware media plans.
4. `run_coarse_stage` builds a dynamic prompt/JSON Schema request, validates untrusted mock output, and converts chunk-local times to absolute video times.
5. `run_boundary_stage` samples local windows around each start/end, validates each decision, and converts window-local boundaries/evidence to absolute times.
6. `finalize_annotation` checks contract values and segment identity, merges compatible nearby segments, rejects timeline conflicts, and emits quality signals.
7. `ArtifactStore` writes stage JSON under `<artifact_root>/<run_id>/<video_id>/`; `write_jsonl` atomically writes sorted final records.

## Navigation Workflow

1. Start at the row matching the observed behavior, not at the repository-wide design document.
2. Read the owning source file and its focused test file together.
3. Trace imports only across the relevant boundary; typed models and request/response schemas usually expose the contract without reading every implementation.
4. For end-to-end or provenance behavior, include `pipeline/runner.py` and `tests/integration/test_cli_mock_e2e.py`.
5. Check whether a proposed path exists before describing it as current behavior.

## Gotchas

- `thirdparty/vllm-main/` is not the root application. It is a large vendored snapshot with separate `AGENTS.md` instructions.
- The approved design mentions `orchestration/`, a real vLLM adapter, directory scanning, Parquet, gold evaluation, and multi-chunk merging. Those root-package implementations do not currently exist.
- Tests mirror source domains closely; use the narrow test module first, then the integration test when behavior crosses boundaries.
- Prompt and stage version constants contribute to reproducibility. Search both `prompts/builders.py` and `pipeline/runner.py` when behavior changes.

## Related Skills

- `.agents/skills/auto-annotation-contracts/SKILL.md`
- `.agents/skills/auto-annotation-run-and-test/SKILL.md`
