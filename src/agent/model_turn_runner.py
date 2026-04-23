from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable

from src.agent.agent_session import AgentSessionState
from src.agent.agent_types import AssistantTurn, OutputSchemaConfig, ToolCall, UsageStats
from src.llm.parsers import LLMBackendError, coerce_finish_reason


StreamEventAppender = Callable[[Any], None] | None


def query_model_turn(
    *,
    client: Any,
    session: AgentSessionState,
    tool_specs: list[dict[str, object]],
    stream_model_responses: bool,
    output_schema: OutputSchemaConfig | None,
    stream_event_append: StreamEventAppender = None,
) -> AssistantTurn:
    if not stream_model_responses:
        turn = client.complete(
            session.to_openai_messages(),
            tool_specs,
            output_schema=output_schema,
        )
        normalized_finish_reason = normalize_finish_reason(
            turn.finish_reason,
            has_tool_calls=bool(turn.tool_calls),
            content=turn.content,
        )
        if normalized_finish_reason != turn.finish_reason:
            turn = replace(turn, finish_reason=normalized_finish_reason)
        assistant_tool_calls = tuple(
            {
                'id': tool_call.id,
                'type': 'function',
                'function': {
                    'name': tool_call.name,
                    'arguments': json.dumps(
                        tool_call.arguments,
                        ensure_ascii=True,
                    ),
                },
            }
            for tool_call in turn.tool_calls
        )
        session.append_assistant(
            turn.content,
            assistant_tool_calls,
            message_id=f'assistant_{len(session.messages)}',
            stop_reason=normalized_finish_reason,
            usage=turn.usage,
        )
        return turn

    assistant_index = session.start_assistant(
        message_id=f'assistant_{len(session.messages)}'
    )
    usage = UsageStats()
    finish_reason: str | None = None
    for event in client.stream(
        session.to_openai_messages(),
        tool_specs,
        output_schema=output_schema,
    ):
        if stream_event_append is not None:
            stream_event_append(event)
        if event.type == 'content_delta':
            session.append_assistant_delta(assistant_index, event.delta)
        elif event.type == 'tool_call_delta':
            session.merge_assistant_tool_call_delta(
                assistant_index,
                tool_call_index=event.tool_call_index or 0,
                tool_call_id=event.tool_call_id,
                tool_name=event.tool_name,
                arguments_delta=event.arguments_delta,
            )
        elif event.type == 'usage':
            usage = usage + event.usage
        elif event.type == 'message_stop':
            finish_reason = event.finish_reason

    assistant_message = session.messages[assistant_index]
    finish_reason = normalize_finish_reason(
        finish_reason,
        has_tool_calls=bool(assistant_message.tool_calls),
        content=assistant_message.content,
    )
    session.finalize_assistant(
        assistant_index,
        finish_reason=finish_reason,
        usage=usage,
    )
    assistant_message = session.messages[assistant_index]
    return AssistantTurn(
        content=assistant_message.content,
        tool_calls=parse_tool_calls_from_message(assistant_message.tool_calls),
        finish_reason=finish_reason,
        raw_message=assistant_message.to_openai_message(),
        usage=usage,
    )


def parse_tool_calls_from_message(
    tool_calls: tuple[dict[str, object], ...],
) -> tuple[ToolCall, ...]:
    parsed: list[ToolCall] = []
    for index, raw_tool_call in enumerate(tool_calls):
        function_block = raw_tool_call.get('function')
        if not isinstance(function_block, dict):
            continue
        name = function_block.get('name')
        if not isinstance(name, str) or not name:
            continue
        raw_arguments = function_block.get('arguments', '')
        if isinstance(raw_arguments, str) and raw_arguments.strip():
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise LLMBackendError(
                    f'Tool arguments must decode to an object, got {type(arguments).__name__}'
                )
        else:
            arguments = {}
        call_id = raw_tool_call.get('id')
        if not isinstance(call_id, str) or not call_id:
            call_id = f'call_{index}'
        parsed.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            )
        )
    return tuple(parsed)


def normalize_finish_reason(
    finish_reason: str | None,
    *,
    has_tool_calls: bool,
    content: str,
) -> str | None:
    normalized = coerce_finish_reason(finish_reason)
    if normalized is not None:
        return normalized
    if has_tool_calls:
        return 'tool_calls'
    if content.strip():
        return 'stop'
    return None
