import asyncio
import hashlib
import json
from pathlib import Path

from auto_annotation.artifacts.store import ArtifactStore
from auto_annotation.config.loader import config_hash
from auto_annotation.config.models import (
    MockBackendConfig,
    RunConfig,
    VllmBackendConfig,
)
from auto_annotation.domain.models import (
    AnnotationDocument,
    ManifestItem,
    Provenance,
)
from auto_annotation.exporters.jsonl import write_jsonl
from auto_annotation.inference.base import InferenceBackend
from auto_annotation.inference.mock import ScriptedMockBackend
from auto_annotation.inference.vllm import VLLM_ADAPTER_VERSION, VllmBackend
from auto_annotation.manifest import read_manifest
from auto_annotation.media.materialize import (
    MEDIA_MATERIALIZER_VERSION,
    LocalVideoMaterializer,
)
from auto_annotation.media.probe import probe_video
from auto_annotation.media.sampling import build_chunks, build_sample_plan
from auto_annotation.ontology.registry import ResolvedContract, load_contract
from auto_annotation.pipeline.coarse import run_coarse_stage
from auto_annotation.pipeline.finalize import finalize_annotation
from auto_annotation.pipeline.refine import run_boundary_stage
from auto_annotation.prompts.builders import (
    BOUNDARY_PROMPT_VERSION,
    COARSE_PROMPT_VERSION,
)


COARSE_STAGE_VERSION = "coarse-stage-v1"
REFINE_STAGE_VERSION = "refine-stage-v1"
FINALIZE_STAGE_VERSION = "finalize-stage-v2"
RUNNER_STAGE_VERSION = "runner-stage-v2"


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve(strict=False) == right.resolve(strict=False):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _validate_output_path(
    output_path: Path,
    inputs: list[tuple[str, Path]],
) -> None:
    for input_name, input_path in inputs:
        if _paths_alias(output_path, input_path):
            raise ValueError(
                f"output_path aliases {input_name}: {input_path}"
            )


def _validate_manifest(
    items: list[ManifestItem], contract: ResolvedContract
) -> None:
    if not items:
        raise ValueError("manifest must contain at least one item")
    dataset_ids = {item.dataset_id for item in items}
    if len(dataset_ids) != 1:
        raise ValueError(
            f"mixed dataset_id values are not supported: {sorted(dataset_ids)}"
        )

    seen_video_ids: set[str] = set()
    for item in items:
        if (
            item.video_id in {"", ".", ".."}
            or "/" in item.video_id
            or "\\" in item.video_id
        ):
            raise ValueError(f"unsafe video_id: {item.video_id}")
        if item.video_id in seen_video_ids:
            raise ValueError(f"duplicate video_id across manifest: {item.video_id}")
        seen_video_ids.add(item.video_id)
        if item.schema_version != contract.schema.schema_id:
            raise ValueError(
                f"manifest schema mismatch for {item.video_id}: "
                f"{item.schema_version}"
            )
        if item.ontology_id != contract.ontology.ontology_id:
            raise ValueError(
                f"manifest ontology mismatch for {item.video_id}: "
                f"{item.ontology_id}"
            )


async def _run_items(
    config: RunConfig,
    items: list[ManifestItem],
    contract: ResolvedContract,
    backend: InferenceBackend,
    store: ArtifactStore,
    run_id: str,
    digest: str,
) -> list[AnnotationDocument]:
    results: list[AnnotationDocument] = []
    for item in items:
        info = probe_video(item.video_uri)
        chunks = build_chunks(info, max_chunk_ms=120_000, overlap_ms=8000)
        if len(chunks) != 1:
            raise ValueError("core slice accepts exactly one short-video chunk")
        chunk = chunks[0]
        sample_plan = build_sample_plan(
            info,
            chunk,
            config.sampling.coarse_fps,
        )
        store.write_json(
            run_id,
            item.video_id,
            "coarse-sample-plan",
            sample_plan,
        )

        coarse = await run_coarse_stage(item, sample_plan, contract, backend)
        store.write_json(run_id, item.video_id, "coarse", coarse)

        refined = await run_boundary_stage(
            item,
            chunk,
            info,
            coarse.segments,
            backend,
            refine_fps=config.sampling.refine_fps,
            window_ms=config.sampling.refine_window_ms,
        )
        store.write_json(run_id, item.video_id, "refined", refined)

        finalized = finalize_annotation(
            video_id=item.video_id,
            duration_ms=info.duration_ms,
            coarse=coarse,
            refined=refined,
            contract=contract,
            timeline=config.timeline,
        )
        document = AnnotationDocument(
            **finalized.model_dump(),
            provenance=Provenance(
                schema_version=contract.schema.schema_id,
                ontology_hash=contract.ontology_hash,
                prompt_versions={
                    "coarse": COARSE_PROMPT_VERSION,
                    "boundary": BOUNDARY_PROMPT_VERSION,
                },
                model_id=backend.model_id,
                model_revision=backend.model_revision,
                config_hash=digest,
            ),
        )
        store.write_json(run_id, item.video_id, "final", document)
        results.append(document)
        write_jsonl(config.output_path, results)
    return results


async def _execute_run(
    config: RunConfig,
    items: list[ManifestItem],
    contract: ResolvedContract,
    run_id: str,
    digest: str,
    mock_script: dict[str, list[dict[str, object]]] | None,
) -> list[AnnotationDocument]:
    backend_config = config.backend
    if isinstance(backend_config, MockBackendConfig):
        if mock_script is None:
            raise TypeError("mock backend requires a script")
        backend = ScriptedMockBackend(mock_script)
        return await _run_items(
            config,
            items,
            contract,
            backend,
            ArtifactStore(config.artifact_root),
            run_id,
            digest,
        )
    if isinstance(backend_config, VllmBackendConfig):
        materializer = LocalVideoMaterializer(
            backend_config.media_staging_root
        )
        materializer.validate()
        async with VllmBackend(
            backend_config,
            materializer=materializer,
        ) as backend:
            await backend.healthcheck()
            return await _run_items(
                config,
                items,
                contract,
                backend,
                ArtifactStore(config.artifact_root),
                run_id,
                digest,
            )
    raise TypeError(f"unsupported backend config: {type(backend_config)!r}")


def run_annotation(config: RunConfig) -> list[AnnotationDocument]:
    config_inputs = (
        []
        if config.source_path is None
        else [("config source", config.source_path)]
    )
    backend_inputs: list[tuple[str, Path]] = []
    if isinstance(config.backend, MockBackendConfig):
        backend_inputs.append(("mock fixture_path", config.backend.fixture_path))
    _validate_output_path(
        config.output_path,
        config_inputs
        + [
            ("manifest_path", config.manifest_path),
            ("schema_path", config.schema_path),
            ("ontology_path", config.ontology_path),
        ]
        + backend_inputs,
    )
    contract = load_contract(config.schema_path, config.ontology_path)
    manifest_bytes = config.manifest_path.read_bytes()
    items = read_manifest(config.manifest_path)
    _validate_manifest(items, contract)
    _validate_output_path(
        config.output_path,
        [
            (f"video input {item.video_id}", item.video_uri)
            for item in items
        ],
    )

    mock_script: dict[str, list[dict[str, object]]] | None = None
    backend_identity: dict[str, str]
    if isinstance(config.backend, MockBackendConfig):
        fixture_bytes = config.backend.fixture_path.read_bytes()
        mock_script = json.loads(fixture_bytes)
        backend_identity = {
            "mock_fixture_sha256": _sha256_bytes(fixture_bytes),
            "model_id": ScriptedMockBackend.model_id,
            "model_revision": ScriptedMockBackend.model_revision,
        }
    elif isinstance(config.backend, VllmBackendConfig):
        backend_identity = {
            "media_materializer_version": MEDIA_MATERIALIZER_VERSION,
            "model_id": config.backend.model_id,
            "model_revision": config.backend.model_revision,
            "vllm_adapter_version": VLLM_ADAPTER_VERSION,
            "vllm_version": config.backend.vllm_version,
        }
    else:
        raise TypeError(f"unsupported backend config: {type(config.backend)!r}")
    identity = {
        "schema_hash": contract.schema_hash,
        "ontology_hash": contract.ontology_hash,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "coarse_prompt_version": COARSE_PROMPT_VERSION,
        "boundary_prompt_version": BOUNDARY_PROMPT_VERSION,
        "coarse_stage_version": COARSE_STAGE_VERSION,
        "refine_stage_version": REFINE_STAGE_VERSION,
        "finalize_stage_version": FINALIZE_STAGE_VERSION,
        "runner_stage_version": RUNNER_STAGE_VERSION,
    }
    identity.update(backend_identity)
    identity.update(
        {
            f"video_sha256:{item.video_id}": _sha256_file(item.video_uri)
            for item in items
        }
    )
    digest = config_hash(
        config,
        identity,
    )
    run_id = f"run-{digest.removeprefix('sha256:')}"
    results = asyncio.run(
        _execute_run(
            config,
            items,
            contract,
            run_id,
            digest,
            mock_script,
        )
    )
    return results
