---
name: auto-annotation-contracts
description: Use when asked to “change run config,” “change the manifest,” “add a Schema field,” “update an Ontology vocabulary,” “change annotation output,” “debug provenance,” or modify timestamp and validation invariants across the auto-annotation pipeline.
---

# Auto Annotation Contracts

## Overview

The core slice has several linked contracts: YAML run configuration, JSONL manifest records, Schema/Ontology YAML, backend request/response JSON, finalized annotation models, and provenance. Change the narrow owner, then verify every downstream consumer whose contract actually changes.

## Path Convention

This skill lives at `.agents/skills/auto-annotation-contracts/SKILL.md`; the repository root is `../../../` relative to this directory. All paths below are repository-relative.

## Where to Find Answers

| Contract | Definition | Primary verification |
| --- | --- | --- |
| Run config fields/defaults | `src/auto_annotation/config/models.py` | `tests/config/test_loader.py` |
| Config path resolution and canonical hash | `src/auto_annotation/config/loader.py` | `tests/config/test_loader.py` |
| Manifest parsing and local-path rules | `src/auto_annotation/manifest.py` | `tests/domain/test_manifest.py` |
| Shared input/output models | `src/auto_annotation/domain/models.py` | `tests/domain/test_models.py` |
| Schema and Ontology YAML models | `src/auto_annotation/ontology/models.py` | `tests/ontology/test_registry.py` |
| Resolved dynamic response schema | `src/auto_annotation/ontology/registry.py` | `tests/ontology/test_registry.py` |
| Prompt/backend request contracts | `src/auto_annotation/prompts/builders.py`, `src/auto_annotation/inference/base.py` | `tests/prompts/test_builders.py`, `tests/inference/test_mock_backend.py` |
| Stage time conversion and strict response validation | `src/auto_annotation/pipeline/coarse.py`, `src/auto_annotation/pipeline/refine.py` | `tests/pipeline/test_stages.py` |
| Final timeline and quality contract | `src/auto_annotation/pipeline/finalize.py` | `tests/pipeline/test_finalize.py` |
| Run-wide validation, identity, and provenance | `src/auto_annotation/pipeline/runner.py` | `tests/integration/test_cli_mock_e2e.py` |
| Executable contract examples | `tests/fixtures/schema/subtask-v1.yaml`, `tests/fixtures/ontology/kitchen-v1.yaml`, `tests/fixtures/mock/annotation.json` | Their consuming test modules |

## Contract Map

### Run configuration

`RunConfig` forbids unknown keys and requires `schema_path`, `ontology_path`, `manifest_path`, `artifact_root`, `output_path`, and `backend`. The only backend shape is `{kind: mock, fixture_path: ...}`. Sampling defaults are coarse `2.0` FPS, refine `6.0` FPS, and a `4000` ms refine window. Timeline defaults are a `250` ms merge gap and `300` ms minimum segment.

Relative config paths resolve from the run-config file's directory. `source_path` is private runtime context and is intentionally excluded from serialization and the canonical config hash.

### Manifest

Each nonblank JSONL line validates as a `ManifestItem`: `video_id`, `video_uri`, `dataset_id`, `schema_version`, and `ontology_id` are required; `context` and `metadata` default to mappings. `video_uri` must be an existing local file and resolves relative to the manifest directory.

The runner requires a non-empty manifest, exactly one `dataset_id` per run, safe globally unique `video_id` values, and Schema/Ontology IDs matching the loaded contract. Preserve whole-manifest prevalidation before media probing, artifact creation, or final output.

### Schema and Ontology

Schema `segment_values` are data-driven. A field is currently either `enum` (with `source: ontology.<vocabulary>`) or `string`, and can be required or optional. Ontology vocabulary IDs must be unique and referenced vocabularies must be non-empty.

The current core slice deliberately requires `text_output.field: sentence` and `text_output.required: true`; language and style remain configurable. `ResolvedContract` supplies both prompt constraints and result validation, so never maintain a second hardcoded label list in pipeline code.

### Time and annotation output

All internal timestamps are integer milliseconds. Media/request ranges and final segments use half-open `[start_ms, end_ms)` intervals; sample/evidence timestamps must be inside the range, never equal to its end.

Coarse model timestamps are chunk-local and become absolute in `run_coarse_stage`. Boundary decisions are window-local and become absolute in `run_boundary_stage`. Finalization requires identical unique coarse/refined segment-ID sets, validates values and nonblank sentences, sorts deterministically, and rejects incompatible overlaps. Compatible segments merge only when both `values` and `sentence` match and their gap is within `merge_gap_ms`.

The final document contains `video_id`, `duration_ms`, `task_summary`, `segments`, `quality`, and `provenance`. Quality currently reports boundary shift, segment count, empty-timeline state, merged sources, and review reasons for shifts over 2 seconds or segments shorter than the configured minimum.

### Run identity and provenance

`config_hash` combines normalized config with Schema/Ontology hashes, manifest bytes, mock-fixture bytes, video hashes, prompt versions, stage versions, and backend model identity. The run directory is `run-<64 hex digest>` and repeated identical inputs reuse that identity deterministically.

When changing prompt semantics, update/review `COARSE_PROMPT_VERSION` or `BOUNDARY_PROMPT_VERSION`. When changing stage behavior, update/review the stage constants in `pipeline/runner.py`. Extend `test_run_identity_tracks_content_and_every_stage_version` when adding another behavior-bearing version.

## Change Workflows

### Add or change a segment field

1. Decide whether the field is `enum` or `string`, whether it is required, and whether a new Ontology vocabulary is needed.
2. Update a Schema/Ontology fixture or the caller-supplied contract; do not add the field to pipeline-specific dictionaries.
3. Verify dynamic response-schema construction and value validation in `tests/ontology/test_registry.py`.
4. Update prompt tests if visible instructions or generated JSON Schema change.
5. Update scripted backend responses and finalization/integration expectations when the executable example changes.

### Change config or manifest shape

1. Modify the Pydantic owner (`config/models.py` or `domain/models.py`) and the relevant loader/parser.
2. Preserve `extra="forbid"`, path-base semantics, and deterministic hashing unless the requirement explicitly changes them.
3. Add invalid-input coverage before updating the CLI integration path.
4. Recheck runner prevalidation and output-alias protection for any new input or output path.

### Change output or provenance

1. Update the domain model and the stage that constructs it.
2. Preserve deterministic JSON conversion and sorted export unless the output contract explicitly changes.
3. Update focused model/finalization tests, artifact/export tests, and the CLI end-to-end assertion.
4. Decide whether the change requires a prompt or stage version increment and ensure it participates in run identity.

## Focused Verification

| Change | Command |
| --- | --- |
| Run config | `uv run pytest tests/config/test_loader.py -q` |
| Manifest/domain | `uv run pytest tests/domain -q` |
| Schema/Ontology | `uv run pytest tests/ontology/test_registry.py tests/prompts/test_builders.py -q` |
| Backend response contract | `uv run pytest tests/inference/test_mock_backend.py tests/pipeline/test_stages.py -q` |
| Timeline/output | `uv run pytest tests/pipeline/test_finalize.py tests/artifacts/test_store_export.py -q` |
| Identity and cross-layer behavior | `uv run pytest tests/integration/test_cli_mock_e2e.py -q` |

## Gotchas

- Pydantic validation alone is not the backend trust boundary; stage code also validates the generated JSON Schema and uses strict parsing for model timestamps.
- The output path must not alias the config, manifest, Schema, Ontology, mock fixture, or any input video, including via symlinks.
- Blank lines affect manifest bytes and therefore run identity even though the manifest parser ignores them.
- Ontology `revision` is descriptive; the canonical content hash is the reproducibility key.
- Schema examples in the approved design predate some implemented YAML shape details. Prefer current models and executable fixtures.

## Related Skills

- `.agents/skills/auto-annotation-codebase-nav/SKILL.md`
- `.agents/skills/auto-annotation-run-and-test/SKILL.md`
