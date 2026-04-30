from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.agent.models.session import AgentSessionState
from src.agent.tools.execution import (
    AgentTool,
    ToolExecutionContext,
    execute_tool_streaming,
    serialize_tool_result,
)
from src.agent.models.types import ToolCall, ToolExecutionResult


@dataclass(frozen=True)
class ToolCallExecutionHooks:
    plugin_tool_preflight_messages: Callable[[str], tuple[str, ...]]
    hook_policy_tool_preflight_messages: Callable[[str], tuple[str, ...]]
    plugin_block_message: Callable[[str], str | None]
    hook_policy_block_message: Callable[[str], str | None]
    execute_delegate_agent: Callable[
        [dict[str, object], str, Callable[[dict[str, object]], None] | None],
        ToolExecutionResult,
    ]
    execute_skill: Callable[[dict[str, object]], ToolExecutionResult]
    plugin_tool_result_messages: Callable[[str], tuple[str, ...]]
    hook_policy_tool_result_messages: Callable[[str], tuple[str, ...]]
    append_runtime_tool_followup_events: Callable[[Any, ToolCall, ToolExecutionResult], None]
    build_plugin_tool_runtime_message: Callable[..., str | None]
    refresh_runtime_views_for_tool_result: Callable[[str, ToolExecutionResult], None]
    build_file_history_entry: Callable[[ToolCall, ToolExecutionResult, int], dict[str, object] | None]


@dataclass(frozen=True)
class ToolCallExecutionOutcome:
    tool_result: ToolExecutionResult
    history_entry: dict[str, object] | None = None


def _preview_tool_content(content: str, *, max_chars: int = 220) -> str:
    normalized = ' '.join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + '...'


def execute_runtime_tool_call(
    *,
    tool_call: ToolCall,
    turn_index: int,
    session: AgentSessionState,
    tool_registry: dict[str, AgentTool],
    tool_context: ToolExecutionContext,
    stream_events: Any,
    plugin_runtime: Any,
    hooks: ToolCallExecutionHooks,
) -> ToolCallExecutionOutcome:
    tool_result: ToolExecutionResult | None = None
    tool_message_index = session.start_tool(
        name=tool_call.name,
        tool_call_id=tool_call.id,
        message_id=f'tool_{len(session.messages)}',
        metadata={'phase': 'starting'},
    )
    stream_events.append(
        {
            'type': 'tool_start',
            'tool_name': tool_call.name,
            'tool_call_id': tool_call.id,
            'message_id': session.messages[tool_message_index].message_id,
            'arguments': dict(tool_call.arguments),
        }
    )
    if plugin_runtime is not None:
        plugin_runtime.record_tool_attempt(tool_call.name, blocked=False)
    plugin_preflight_messages = hooks.plugin_tool_preflight_messages(tool_call.name)
    policy_preflight_messages = hooks.hook_policy_tool_preflight_messages(tool_call.name)
    if plugin_preflight_messages:
        stream_events.append(
            {
                'type': 'plugin_tool_preflight',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': session.messages[tool_message_index].message_id,
                'message_count': len(plugin_preflight_messages),
            }
        )
    if policy_preflight_messages:
        stream_events.append(
            {
                'type': 'hook_policy_tool_preflight',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': session.messages[tool_message_index].message_id,
                'message_count': len(policy_preflight_messages),
            }
        )
    plugin_block_message = hooks.plugin_block_message(tool_call.name)
    policy_block_message = hooks.hook_policy_block_message(tool_call.name)
    if plugin_block_message is not None:
        if plugin_runtime is not None:
            blocked_attempts = int(
                plugin_runtime.session_state.get('blocked_tool_attempts', 0)
            )
            plugin_runtime.session_state['blocked_tool_attempts'] = blocked_attempts + 1
        tool_result = ToolExecutionResult(
            name=tool_call.name,
            ok=False,
            content=plugin_block_message,
            metadata={
                'action': 'plugin_block',
                'plugin_blocked': True,
                'plugin_block_message': plugin_block_message,
            },
        )
        stream_events.append(
            {
                'type': 'plugin_tool_block',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': session.messages[tool_message_index].message_id,
                'message': plugin_block_message,
            }
        )
    if policy_block_message is not None:
        tool_result = ToolExecutionResult(
            name=tool_call.name,
            ok=False,
            content=policy_block_message,
            metadata={
                'action': 'hook_policy_block',
                'hook_policy_blocked': True,
                'hook_policy_block_message': policy_block_message,
                'error_kind': 'permission_denied',
            },
        )
        stream_events.append(
            {
                'type': 'hook_policy_tool_block',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': session.messages[tool_message_index].message_id,
                'message': policy_block_message,
            }
        )
    if tool_call.name in ('Agent', 'delegate_agent'):
        if tool_result is None:
            tool_result = hooks.execute_delegate_agent(
                dict(tool_call.arguments),
                tool_call.name,
                stream_events.append,
            )
    elif tool_call.name == 'Skill':
        if tool_result is None:
            tool_result = hooks.execute_skill(dict(tool_call.arguments))
    elif tool_result is None:
        for update in execute_tool_streaming(
            tool_registry,
            tool_call.name,
            tool_call.arguments,
            tool_context,
        ):
            if update.kind == 'delta':
                session.append_tool_delta(
                    tool_message_index,
                    update.content,
                    metadata={'last_stream': update.stream or 'tool'},
                )
                stream_events.append(
                    {
                        'type': 'tool_delta',
                        'tool_name': tool_call.name,
                        'tool_call_id': tool_call.id,
                        'message_id': session.messages[tool_message_index].message_id,
                        'stream': update.stream,
                        'delta': update.content,
                    }
                )
                continue
            tool_result = update.result
    if tool_result is None:
        raise RuntimeError(f'Tool executor returned no final result for {tool_call.name}')
    if plugin_runtime is not None:
        plugin_runtime.record_tool_result(
            tool_call.name,
            ok=tool_result.ok,
            metadata=tool_result.metadata,
        )
    plugin_messages = hooks.plugin_tool_result_messages(tool_call.name)
    policy_messages = hooks.hook_policy_tool_result_messages(tool_call.name)
    if plugin_messages:
        merged_metadata = dict(tool_result.metadata)
        merged_metadata['plugin_messages'] = list(plugin_messages)
        tool_result = ToolExecutionResult(
            name=tool_result.name,
            ok=tool_result.ok,
            content=tool_result.content,
            metadata=merged_metadata,
        )
        for message in plugin_messages:
            stream_events.append(
                {
                    'type': 'plugin_tool_hook',
                    'tool_name': tool_call.name,
                    'tool_call_id': tool_call.id,
                    'message_id': session.messages[tool_message_index].message_id,
                    'message': message,
                }
            )
    if policy_messages:
        merged_metadata = dict(tool_result.metadata)
        merged_metadata['hook_policy_messages'] = list(policy_messages)
        tool_result = ToolExecutionResult(
            name=tool_result.name,
            ok=tool_result.ok,
            content=tool_result.content,
            metadata=merged_metadata,
        )
        for message in policy_messages:
            stream_events.append(
                {
                    'type': 'hook_policy_tool_hook',
                    'tool_name': tool_call.name,
                    'tool_call_id': tool_call.id,
                    'message_id': session.messages[tool_message_index].message_id,
                    'message': message,
                }
            )
    if tool_result.metadata.get('error_kind') == 'permission_denied':
        stream_events.append(
            {
                'type': 'tool_permission_denial',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': session.messages[tool_message_index].message_id,
                'reason': tool_result.content,
                'source': (
                    'hook_policy'
                    if tool_result.metadata.get('action') == 'hook_policy_block'
                    else 'tool_runtime'
                ),
            }
        )
    session.finalize_tool(
        tool_message_index,
        content=serialize_tool_result(tool_result),
        metadata={
            'phase': 'completed',
            'plugin_preflight_messages': list(plugin_preflight_messages),
            'hook_policy_preflight_messages': list(policy_preflight_messages),
            **dict(tool_result.metadata),
        },
        stop_reason='tool_completed',
    )
    stream_events.append(
        {
            'type': 'tool_result',
            'tool_name': tool_call.name,
            'tool_call_id': tool_call.id,
            'message_id': session.messages[tool_message_index].message_id,
            'ok': tool_result.ok,
            'metadata': dict(tool_result.metadata),
            'content': tool_result.content,
            'content_preview': _preview_tool_content(tool_result.content),
        }
    )
    hooks.append_runtime_tool_followup_events(stream_events, tool_call, tool_result)
    plugin_runtime_message = hooks.build_plugin_tool_runtime_message(
        tool_name=tool_call.name,
        preflight_messages=plugin_preflight_messages,
        block_message=plugin_block_message,
        plugin_messages=plugin_messages,
        hook_policy_preflight_messages=policy_preflight_messages,
        hook_policy_block_message=policy_block_message,
        hook_policy_messages=policy_messages,
        delegate_preflight_messages=tuple(
            message
            for message in tool_result.metadata.get(
                'plugin_delegate_preflight_messages',
                [],
            )
            if isinstance(message, str) and message
        ),
        delegate_after_messages=tuple(
            message
            for message in tool_result.metadata.get(
                'plugin_delegate_after_messages',
                [],
            )
            if isinstance(message, str) and message
        ),
    )
    if plugin_runtime_message is not None:
        session.append_user(
            plugin_runtime_message,
            metadata={
                'kind': 'plugin_tool_runtime',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'plugin_blocked': plugin_block_message is not None,
                'plugin_message_count': len(plugin_messages),
                'plugin_preflight_count': len(plugin_preflight_messages),
            },
            message_id=f'plugin_tool_runtime_{tool_call.id}',
        )
        stream_events.append(
            {
                'type': 'plugin_tool_context',
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
                'message_id': f'plugin_tool_runtime_{tool_call.id}',
                'blocked': plugin_block_message is not None,
                'message_count': len(plugin_messages),
                'preflight_count': len(plugin_preflight_messages),
            }
        )
    hooks.refresh_runtime_views_for_tool_result(tool_call.name, tool_result)
    history_entry = hooks.build_file_history_entry(
        tool_call,
        tool_result,
        turn_index,
    )
    return ToolCallExecutionOutcome(
        tool_result=tool_result,
        history_entry=history_entry,
    )
