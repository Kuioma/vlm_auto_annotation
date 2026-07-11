from collections import defaultdict, deque
from copy import deepcopy
from typing import Any

from auto_annotation.inference.base import (
    GenerationRequest,
    GenerationResponse,
    validate_response,
)


class ScriptedMockBackend:
    model_id = "mock/video-annotator"
    model_revision = "fixture-v1"

    def __init__(self, script: dict[str, list[dict[str, Any]]]) -> None:
        self._script = defaultdict(deque)
        for request_id, responses in script.items():
            self._script[request_id].extend(deepcopy(responses))
        self.calls: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        if not self._script[request.request_id]:
            raise RuntimeError(f"mock script exhausted for {request.request_id}")
        content = self._script[request.request_id].popleft()
        validate_response(request, content)
        return GenerationResponse(content=content, raw=deepcopy(content))
