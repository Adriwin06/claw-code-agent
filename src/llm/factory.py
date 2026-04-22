from __future__ import annotations

import os
from typing import Any

from src.agent.agent_types import ModelConfig
from .litellm_backend import LiteLLMClient
from .parsers import LLMBackendError


DEFAULT_LLM_BACKEND = 'litellm'
SUPPORTED_LLM_BACKENDS = ('litellm',)

_BACKEND_ALIASES = {
    'lite-llm': 'litellm',
    'lite_llm': 'litellm',
    'litellm': 'litellm',
}


def resolve_llm_backend(backend: str | None = None) -> str:
    raw = backend or os.environ.get('CLAW_LLM_BACKEND', DEFAULT_LLM_BACKEND)
    normalized = _BACKEND_ALIASES.get(str(raw).strip().lower())
    if normalized is None:
        supported = ', '.join(SUPPORTED_LLM_BACKENDS)
        raise LLMBackendError(
            f'Unsupported LLM backend {raw!r}. Expected one of: {supported}'
        )
    return normalized


def build_llm_client(
    model_config: ModelConfig,
    *,
    backend: str | None = None,
) -> Any:
    selected_backend = resolve_llm_backend(backend or model_config.llm_backend)
    if selected_backend == 'litellm':
        return LiteLLMClient(model_config)
    supported = ', '.join(SUPPORTED_LLM_BACKENDS)
    raise LLMBackendError(
        f'Unsupported LLM backend {selected_backend!r}. Expected one of: {supported}'
    )
