"""Deterministic prompt adapter for DashScope JSON Mode."""

import json

from auto_annotation.inference.base import GenerationRequest


DASHSCOPE_PROMPT_VERSION = "dashscope-qwen-json-mode-v1"
DASHSCOPE_PROMPT_ADAPTER_VERSION = DASHSCOPE_PROMPT_VERSION
DASHSCOPE_PROMPT_MARKER = "[DashScope JSON mode contract]"


def build_dashscope_prompt(request: GenerationRequest) -> str:
    """Add the local response contract to a JSON-mode prompt.

    DashScope's OpenAI-compatible endpoint accepts ``json_object`` rather than
    a provider-specific JSON Schema request.  The complete dynamic schema is
    therefore embedded in the deterministic text prompt and is still checked
    locally after the response returns.
    """

    if DASHSCOPE_PROMPT_MARKER in request.prompt:
        return request.prompt
    schema = json.dumps(
        request.response_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{request.prompt}\n{DASHSCOPE_PROMPT_MARKER}\n"
        "输出必须是单个 JSON object，且只能输出 JSON，不得输出 Markdown、解释或思维过程。"
        "严格遵守下面完整的动态 JSON Schema；不得添加额外字段。"
        "时间单位为毫秒，局部范围、半开区间、Ontology 闭集 ID 和证据时间约束均适用；"
        "不得创造 Schema 或 Ontology 未声明的字段和值。\n"
        f"JSON Schema: {schema}\n"
        "JSON object only."
    )


def adapt_request(request: GenerationRequest) -> GenerationRequest:
    """Return a copy with the DashScope JSON-mode instructions applied."""

    return request.model_copy(update={"prompt": build_dashscope_prompt(request)})


def build_dashscope_request(request: GenerationRequest) -> GenerationRequest:
    """Compatibility name for callers that treat prompt adaptation as a builder."""

    return adapt_request(request)
