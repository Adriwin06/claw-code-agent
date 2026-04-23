from __future__ import annotations

import json
from typing import Any, Iterator

from src.agent.agent_types import (
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
    UsageStats,
)


class LLMBackendError(RuntimeError):
    """Raised when backend payloads are malformed."""


def join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip('/')
    return f'{base}/{suffix.lstrip("/")}'


def normalize_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get('type') == 'text' and isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            if isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            parts.append(json.dumps(item, ensure_ascii=True))
        return ''.join(parts)
    return str(content)


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        raw_arguments = raw_arguments.strip()
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMBackendError(
                f'Invalid tool arguments returned by model: {raw_arguments!r}'
            ) from exc
        if not isinstance(parsed, dict):
            raise LLMBackendError(
                f'Tool arguments must decode to an object, got {type(parsed).__name__}'
            )
        return parsed
    raise LLMBackendError(
        f'Unsupported tool arguments payload: {type(raw_arguments).__name__}'
    )


def optional_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def parse_usage(payload: Any) -> UsageStats:
    if not isinstance(payload, dict):
        return UsageStats()
    completion_details = payload.get('completion_tokens_details')
    if not isinstance(completion_details, dict):
        completion_details = {}
    return UsageStats(
        input_tokens=(
            optional_int(payload.get('input_tokens'))
            or optional_int(payload.get('prompt_tokens'))
            or optional_int(payload.get('prompt_eval_count'))
        ),
        output_tokens=(
            optional_int(payload.get('output_tokens'))
            or optional_int(payload.get('completion_tokens'))
            or optional_int(payload.get('eval_count'))
        ),
        cache_creation_input_tokens=optional_int(
            payload.get('cache_creation_input_tokens')
        ),
        cache_read_input_tokens=optional_int(payload.get('cache_read_input_tokens')),
        reasoning_tokens=(
            optional_int(payload.get('reasoning_tokens'))
            or optional_int(completion_details.get('reasoning_tokens'))
        ),
    )


def build_response_format(
    schema: OutputSchemaConfig | None,
) -> dict[str, Any] | None:
    if schema is None:
        return None
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': schema.name,
            'schema': schema.schema,
            'strict': schema.strict,
        },
    }


def coerce_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        aliases = {
            'completed': 'stop',
            'complete': 'stop',
            'done': 'stop',
            'end_turn': 'stop',
            'tool_use': 'tool_calls',
            'tool_call': 'tool_calls',
            'tools': 'tool_calls',
        }
        return aliases.get(lowered, lowered)
    return str(value).strip() or None


def parse_tool_calls_from_message(
    message: dict[str, Any],
    *,
    strict: bool,
    include_legacy_function_call: bool,
) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    raw_tool_calls = message.get('tool_calls')
    if isinstance(raw_tool_calls, list):
        for idx, raw_call in enumerate(raw_tool_calls):
            if not isinstance(raw_call, dict):
                if strict:
                    raise LLMBackendError('Malformed tool call payload from model')
                continue
            function_block = raw_call.get('function') or {}
            if not isinstance(function_block, dict):
                if strict:
                    raise LLMBackendError(
                        'Malformed tool call function payload from model'
                    )
                continue
            name = function_block.get('name')
            if not isinstance(name, str) or not name:
                if strict:
                    raise LLMBackendError('Tool call missing function name')
                continue
            call_id = raw_call.get('id')
            if not isinstance(call_id, str) or not call_id:
                call_id = f'call_{idx}'
            arguments = parse_tool_arguments(function_block.get('arguments'))
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return tool_calls

    if include_legacy_function_call and isinstance(message.get('function_call'), dict):
        function_call = message['function_call']
        name = function_call.get('name')
        if not isinstance(name, str) or not name:
            raise LLMBackendError('Function call missing name')
        arguments = parse_tool_arguments(function_call.get('arguments'))
        tool_calls.append(ToolCall(id='call_0', name=name, arguments=arguments))
    return tool_calls


def iter_stream_events(
    payload: dict[str, Any],
    *,
    normalize_list_content: bool,
) -> Iterator[StreamEvent]:
    usage = parse_usage(payload.get('usage'))
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
        elif normalize_list_content and isinstance(content, list):
            normalized = normalize_content(content)
            if normalized:
                yield StreamEvent(type='content_delta', delta=normalized, raw_event=choice)

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

        finish_reason = coerce_finish_reason(choice.get('finish_reason'))
        if finish_reason is not None:
            yield StreamEvent(
                type='message_stop',
                finish_reason=finish_reason,
                raw_event=choice,
            )
