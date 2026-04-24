from __future__ import annotations

from typing import Any, Iterator

from src.agent.models.types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
)
from .parsers import (
    LLMBackendError,
    build_response_format,
    coerce_finish_reason,
    iter_stream_events,
    normalize_content,
    parse_tool_calls_from_message,
    parse_usage,
)


def _coerce_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, 'model_dump', None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    dict_method = getattr(payload, 'dict', None)
    if callable(dict_method):
        dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped
    raise LLMBackendError(
        f'LiteLLM returned unsupported payload type: {type(payload).__name__}'
    )


class LiteLLMClient:
    """LiteLLM-backed chat client with the shared runtime client interface."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        payload = _coerce_payload(
            self._completion(
                messages=messages,
                tools=tools,
                output_schema=output_schema,
                stream=False,
            )
        )

        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise LLMBackendError('LiteLLM backend returned no choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMBackendError('LiteLLM backend returned malformed choice data')

        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise LLMBackendError('LiteLLM backend returned no assistant message')

        return AssistantTurn(
            content=normalize_content(message.get('content')),
            tool_calls=tuple(self._parse_tool_calls_from_message(message)),
            finish_reason=coerce_finish_reason(first_choice.get('finish_reason')),
            raw_message=message,
            usage=parse_usage(payload.get('usage')),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]:
        stream = self._completion(
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            stream=True,
        )
        yield StreamEvent(type='message_start')
        for raw_event in stream:
            event = _coerce_payload(raw_event)
            yield from self._parse_stream_payload(event)

    def _completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: OutputSchemaConfig | None,
        stream: bool,
    ) -> Any:
        try:
            from litellm import completion  # type: ignore
        except ImportError as exc:
            raise LLMBackendError(
                'LiteLLM backend requested but litellm is not installed. '
                'Install it with "pip install litellm".'
            ) from exc

        kwargs: dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'temperature': self.config.temperature,
            'api_base': self.config.base_url,
            'api_key': self.config.api_key,
            'timeout': self.config.timeout_seconds,
            'stream': stream,
        }
        response_format = build_response_format(output_schema)
        if response_format is not None:
            kwargs['response_format'] = response_format
        if stream:
            kwargs['stream_options'] = {'include_usage': True}

        try:
            return completion(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive wrapper for provider errors
            raise LLMBackendError(f'LiteLLM request failed: {exc}') from exc

    def _parse_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        return parse_tool_calls_from_message(
            message,
            strict=False,
            include_legacy_function_call=False,
        )

    def _parse_stream_payload(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        yield from iter_stream_events(payload, normalize_list_content=True)
