from auto_annotation.domain.models import (
    CoarseAnnotation,
    ManifestItem,
    SamplePlan,
    Segment,
)
from auto_annotation.inference.base import InferenceBackend, validate_response
from auto_annotation.ontology.registry import ResolvedContract
from auto_annotation.prompts.builders import build_coarse_request


async def run_coarse_stage(
    item: ManifestItem,
    plan: SamplePlan,
    contract: ResolvedContract,
    backend: InferenceBackend,
) -> CoarseAnnotation:
    chunk = plan.chunk
    request = build_coarse_request(item, plan, contract)
    response = await backend.generate(request)
    validate_response(request, response.content)
    local_annotation = CoarseAnnotation.model_validate(response.content, strict=True)
    chunk_duration_ms = chunk.end_ms - chunk.start_ms
    absolute_segments = []
    for segment in local_annotation.segments:
        if segment.start_ms < 0 or segment.end_ms > chunk_duration_ms:
            raise ValueError(f"segment outside chunk: {segment.segment_id}")
        if any(
            timestamp < 0 or timestamp >= chunk_duration_ms
            for timestamp in segment.evidence_timestamps_ms
        ):
            raise ValueError(f"evidence outside chunk: {segment.segment_id}")
        contract.validate_values(segment.values)
        absolute_segments.append(
            Segment(
                segment_id=segment.segment_id,
                start_ms=segment.start_ms + chunk.start_ms,
                end_ms=segment.end_ms + chunk.start_ms,
                values=segment.values,
                sentence=segment.sentence,
                evidence_timestamps_ms=[
                    timestamp + chunk.start_ms
                    for timestamp in segment.evidence_timestamps_ms
                ],
            )
        )
    return CoarseAnnotation(
        task_summary=local_annotation.task_summary,
        segments=absolute_segments,
    )
