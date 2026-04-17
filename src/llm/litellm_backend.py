from __future__ import annotations

from typing import Any, Iterator

from ..agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
)
from ..openai_compat import (
    OpenAICompatError,
    _build_response_format,
    _normalize_content,
    _parse_tool_arguments,
    _parse_usage,
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
    raise OpenAICompatError(
        f'LiteLLM returned unsupported payload type: {type(payload).__name__}'
    )


def _coerce_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


class LiteLLMClient:
    """LiteLLM-backed chat client that mirrors OpenAICompatClient's interface."""

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
            raise OpenAICompatError('LiteLLM backend returned no choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatError('LiteLLM backend returned malformed choice data')

        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise OpenAICompatError('LiteLLM backend returned no assistant message')

        return AssistantTurn(
            content=_normalize_content(message.get('content')),
            tool_calls=tuple(self._parse_tool_calls_from_message(message)),
            finish_reason=_coerce_finish_reason(first_choice.get('finish_reason')),
            raw_message=message,
            usage=_parse_usage(payload.get('usage')),
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
            raise OpenAICompatError(
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
        response_format = _build_response_format(output_schema)
        if response_format is not None:
            kwargs['response_format'] = response_format
        if stream:
            kwargs['stream_options'] = {'include_usage': True}

        try:
            return completion(**kwargs)
        except Exception as exc:  # pragma: no cover - defensive wrapper for provider errors
            raise OpenAICompatError(f'LiteLLM request failed: {exc}') from exc

    def _parse_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get('tool_calls')
        if isinstance(raw_tool_calls, list):
            for idx, raw_call in enumerate(raw_tool_calls):
                if not isinstance(raw_call, dict):
                    continue
                function_block = raw_call.get('function') or {}
                if not isinstance(function_block, dict):
                    continue
                name = function_block.get('name')
                if not isinstance(name, str) or not name:
                    continue
                call_id = raw_call.get('id')
                if not isinstance(call_id, str) or not call_id:
                    call_id = f'call_{idx}'
                arguments = _parse_tool_arguments(function_block.get('arguments'))
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return tool_calls

    def _parse_stream_payload(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        usage = _parse_usage(payload.get('usage'))
        if usage.total_tokens:
            yield StreamEvent(type='usage', usage=usage, raw_event=payload)

        choices = payload.get('choices')
        if not isinstance(choices, list):
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                delta = {}

            content = delta.get('content')
            if isinstance(content, str) and content:
                yield StreamEvent(type='content_delta', delta=content, raw_event=choice)
            elif isinstance(content, list):
                normalized = _normalize_content(content)
                if normalized:
                    yield StreamEvent(
                        type='content_delta',
                        delta=normalized,
                        raw_event=choice,
                    )

            tool_calls = delta.get('tool_calls')
            if isinstance(tool_calls, list):
                for raw_tool_call in tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    function_block = raw_tool_call.get('function')
                    if not isinstance(function_block, dict):
                        function_block = {}
                    yield StreamEvent(
                        type='tool_call_delta',
                        tool_call_index=(
                            raw_tool_call.get('index')
                            if isinstance(raw_tool_call.get('index'), int)
                            else 0
                        ),
                        tool_call_id=(
                            raw_tool_call.get('id')
                            if isinstance(raw_tool_call.get('id'), str)
                            else None
                        ),
                        tool_name=(
                            function_block.get('name')
                            if isinstance(function_block.get('name'), str)
                            else None
                        ),
                        arguments_delta=(
                            function_block.get('arguments')
                            if isinstance(function_block.get('arguments'), str)
                            else ''
                        ),
                        raw_event=raw_tool_call,
                    )

            finish_reason = _coerce_finish_reason(choice.get('finish_reason'))
            if finish_reason is not None:
                yield StreamEvent(
                    type='message_stop',
                    finish_reason=finish_reason,
                    raw_event=choice,
                )
