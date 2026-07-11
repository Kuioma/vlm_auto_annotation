# Repository Agent Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add concise, evidence-based repository guidance for agents working on the currently executable auto-annotation core slice.

**Architecture:** A short root `AGENTS.md` provides the repository map, supported commands, current implementation boundary, and links into three focused repository-local skills. The skills separately cover codebase navigation, data/configuration contracts, and running/testing so detailed workflows stay maintainable without making the root instructions unwieldy.

**Tech Stack:** Markdown, Python 3.12-3.13, uv, pytest 8, FFmpeg/ffprobe, Pydantic 2, PyYAML 6, jsonschema 4.

## Global Constraints

- Treat the executable mock-backend core slice as authoritative.
- Mark real vLLM transport, orchestration/retries, long-video merging, and other design-only capabilities as planned rather than implemented.
- Do not modify or duplicate guidance under `thirdparty/vllm-main`; it has its own `AGENTS.md`.
- Include only commands and paths evidenced by current repository files.
- Do not add lint, format, type-check, changelog, or decision-log workflows that the repository does not define.
- The workspace is not a Git worktree, so Git diff and commit steps are unavailable.

---

### Task 1: Root Repository Map

**Files:**
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: `pyproject.toml`, `src/auto_annotation/`, `tests/`, and existing design/plan documents.
- Produces: The concise entry point that routes future agents to repository paths, commands, rules, limitations, and local skills.

- [x] **Step 1: Write the repository overview and current implementation boundary**

Document the package purpose, coarse/refine/finalize flow, mock-only backend, local short-video constraint, and separation from `thirdparty/vllm-main`.

- [x] **Step 2: Add evidenced commands and key paths**

Include `uv sync --extra test`, `uv run pytest -q`, targeted pytest invocation, and `uv run auto-annotate run --config <run.yaml>`. State that no lint, format, or type-check command is configured.

- [x] **Step 3: Add repository-specific rules, gotchas, documentation links, and skill links**

Cover integer-millisecond half-open intervals, dynamic Schema/Ontology contracts, versioned run identity, relative-path bases, deterministic atomic output, and planned-vs-implemented scope.

- [x] **Step 4: Check the file is compact and its referenced paths exist**

Run: `test -f AGENTS.md && test -f pyproject.toml && test -d src/auto_annotation && test -d tests`

Expected: exit status 0.

### Task 2: Codebase Navigation Skill

**Files:**
- Create: `.agents/skills/auto-annotation-codebase-nav/SKILL.md`

**Interfaces:**
- Consumes: The module tree under `src/auto_annotation/` and the mirrored test layout under `tests/`.
- Produces: Routing guidance for questions such as “where is the pipeline?”, “trace annotation flow”, and “which module owns this behavior?”.

- [x] **Step 1: Add valid skill frontmatter and path convention**

Use only `name` and `description` in YAML frontmatter and state that the repository root is `../../../` from the skill directory.

- [x] **Step 2: Add the source-of-truth routing table and end-to-end data flow**

Route CLI/configuration, manifest/domain, media, contracts, prompts/inference, pipeline, artifacts/exporters, tests, and future design questions to exact current paths.

- [x] **Step 3: Add navigation gotchas and related-skill links**

Distinguish the root package from the vendored vLLM snapshot and identify planned modules that do not yet exist.

- [x] **Step 4: Verify every routed top-level path exists**

Run: `test -f src/auto_annotation/cli.py && test -f src/auto_annotation/pipeline/runner.py && test -f tests/integration/test_cli_mock_e2e.py`

Expected: exit status 0.

### Task 3: Contracts Skill

**Files:**
- Create: `.agents/skills/auto-annotation-contracts/SKILL.md`

**Interfaces:**
- Consumes: `config`, `manifest`, `domain`, `ontology`, prompt-builder, runner, and fixture implementations/tests.
- Produces: Guidance for changing run configuration, manifests, Schema/Ontology definitions, final output, and provenance without breaking cross-layer invariants.

- [x] **Step 1: Add valid skill frontmatter and answer-routing table**

Use concrete triggers including “change run config”, “add a schema field”, “update ontology”, “change manifest”, and “debug provenance”.

- [x] **Step 2: Document contract relationships and change workflow**

Describe path resolution, local-video validation, dataset/schema/ontology consistency, dynamic segment fields, the stable required `sentence` key, half-open timelines, and run-identity inputs.

- [x] **Step 3: Add focused verification commands and gotchas**

Route contract changes to config, domain, ontology, prompt, pipeline, artifact, and integration tests as appropriate; warn against hardcoding vocabulary IDs in pipeline code.

- [x] **Step 4: Verify referenced implementations and fixtures exist**

Run: `test -f src/auto_annotation/config/models.py && test -f src/auto_annotation/ontology/registry.py && test -f tests/fixtures/schema/subtask-v1.yaml && test -f tests/fixtures/ontology/kitchen-v1.yaml`

Expected: exit status 0.

### Task 4: Run and Test Skill

**Files:**
- Create: `.agents/skills/auto-annotation-run-and-test/SKILL.md`

**Interfaces:**
- Consumes: `pyproject.toml`, CLI/runner behavior, the pytest layout, media-tool guards, and artifact/export implementations.
- Produces: Exact setup, test-selection, CLI, output-inspection, and troubleshooting guidance.

- [x] **Step 1: Add valid skill frontmatter and prerequisites**

Document supported Python versions, uv, and the FFmpeg/ffprobe requirement for the real-media integration path.

- [x] **Step 2: Add commands and a minimal run-config shape**

Use only current mock-backend fields and explain where relative paths resolve. Describe required manifest and fixture inputs without presenting test fixtures as production defaults.

- [x] **Step 3: Add test-selection and troubleshooting tables**

Map each subsystem to its focused pytest file and document skipped media tests, prevalidation failures, unsafe output aliases, single-chunk limits, deterministic run IDs, and artifact locations.

- [x] **Step 4: Verify referenced test files exist**

Run: `test -f tests/media/test_probe_sampling.py && test -f tests/pipeline/test_stages.py && test -f tests/integration/test_cli_mock_e2e.py`

Expected: exit status 0.

### Task 5: Documentation Validation

**Files:**
- Validate: `AGENTS.md`
- Validate: `.agents/skills/auto-annotation-codebase-nav/SKILL.md`
- Validate: `.agents/skills/auto-annotation-contracts/SKILL.md`
- Validate: `.agents/skills/auto-annotation-run-and-test/SKILL.md`

**Interfaces:**
- Consumes: All documentation created in Tasks 1-4.
- Produces: Verified repository-local agent guidance with recorded limitations.

- [x] **Step 1: Re-read every generated document**

Run: `sed -n '1,260p' AGENTS.md` and the same command for each generated `SKILL.md`.

Expected: concise Markdown with no empty sections, stale commands, or claims that planned features are implemented.

- [x] **Step 2: Inspect skill frontmatter**

Run: `for f in .agents/skills/*/SKILL.md; do sed -n '1,/^---$/p' "$f"; done`

Expected: each skill has only `name` and `description` between its opening and closing delimiters.

- [x] **Step 3: Scan for placeholders and invalid planned-feature claims**

Run: `rg -n 'T[B]D|T[O]DO|implement lat[e]r|真实 vLLM.*已实现|real vLLM.*available|long-video merge is available' AGENTS.md .agents/skills`

Expected: no matches.

- [x] **Step 4: Run the repository test suite**

Run: `uv run pytest -q`

Expected: all runnable tests pass; media integration tests may be skipped only when FFmpeg or ffprobe is unavailable.

- [x] **Step 5: Record unavailable validation**

Report that no repository Markdown checker is configured and that `git diff -- AGENTS.md .agents` cannot run because the workspace lacks Git metadata.
