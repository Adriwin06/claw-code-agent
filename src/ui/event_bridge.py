from __future__ import annotations

from typing import Callable, Sequence

from src.agent.agent_types import AgentRunResult
from src.ui.conversation import (
    ActivityItem,
    ConversationEntry,
    ConversationHistoryItem,
    ConversationTurn,
    build_conversation_history_items,
)
from src.ui.formatting import _friendly_stop_reason, _preview_multiline, _preview_value
from src.ui.state import AgentTuiState


MAX_ACTIVITY_ITEMS = 14


class AgentTuiEventBridge:
    def __init__(
        self,
        state: AgentTuiState,
        *,
        emit_data: Callable[[str], None],
        on_state_change: Callable[[AgentTuiState], None] | None = None,
        on_turns_change: Callable[[tuple[ConversationTurn, ...]], None] | None = None,
        on_history_change: Callable[[tuple[ConversationHistoryItem, ...]], None] | None = None,
        on_activity_change: Callable[[tuple[ActivityItem, ...]], None] | None = None,
    ) -> None:
        self.state = state
        self._emit_data = emit_data
        self._on_state_change = on_state_change
        self._on_turns_change = on_turns_change
        self._on_history_change = on_history_change
        self._on_activity_change = on_activity_change
        self._assistant_open = False
        self._assistant_ends_with_newline = True
        self._tool_stream_key: tuple[str | None, str] | None = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True
        self._announced_tool_plans: set[str] = set()
        self._streamed_assistant_output = False
        self._turns: list[ConversationTurn] = []
        self._activity: list[ActivityItem] = []
        self._activity_index: dict[str, int] = {}
        self._tool_stream_buffers: dict[str, str] = {}
        self._active_turn_id: str | None = None

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    @property
    def history_items(self) -> tuple[ConversationHistoryItem, ...]:
        return build_conversation_history_items(self._turns)

    @property
    def activity_items(self) -> tuple[ActivityItem, ...]:
        return tuple(self._activity)

    def restore_history(
        self,
        turns: Sequence[ConversationTurn],
        *,
        announce_activity: bool = True,
    ) -> None:
        self._turns = [
            ConversationTurn(
                turn_id=turn.turn_id,
                user_prompt=turn.user_prompt,
                assistant_response=turn.assistant_response,
                assistant_status=turn.assistant_status,
                phase_label=turn.phase_label,
                tool_count=turn.tool_count,
                restored=turn.restored,
                stop_reason=turn.stop_reason,
                session_id=turn.session_id,
                entries=[
                    ConversationEntry(
                        entry_id=entry.entry_id,
                        kind=entry.kind,
                        title=entry.title,
                        content=entry.content,
                        status=entry.status,
                        merge_key=entry.merge_key,
                    )
                    for entry in turn.entries
                ],
            )
            for turn in turns
        ]
        self._active_turn_id = self._turns[-1].turn_id if self._turns else None
        self.state.conversation_turns = len(self._turns)
        if turns and announce_activity:
            self._upsert_activity(
                'history',
                label='Conversation restored',
                detail=f'Loaded {len(self._turns)} previous turns',
                status='info',
            )
        self._publish_turns()
        if announce_activity:
            self._publish_activity()
        self._publish_state()

    def begin_prompt(self, prompt: str, *, session_id: str | None = None) -> None:
        self._close_open_blocks()
        self._announced_tool_plans.clear()
        self._tool_stream_buffers.clear()
        self.state.busy = True
        self.state.status = 'Running'
        self.state.phase = 'Preparing'
        self.state.phase_detail = 'Waiting for model call'
        self.state.prompt_count += 1
        if session_id:
            self.state.session_id = session_id
        self._streamed_assistant_output = False
        turn = ConversationTurn(
            turn_id=f'turn-{len(self._turns) + 1}',
            user_prompt=prompt,
            assistant_status='Queued',
            phase_label='Queued',
            session_id=session_id,
        )
        self._turns.append(turn)
        self._active_turn_id = turn.turn_id
        self.state.conversation_turns = len(self._turns)
        self._emit_data(f'[user] {prompt}\n')
        self._upsert_activity(
            'prompt',
            label='Prompt queued',
            detail=_preview_value(prompt, max_chars=180),
            status='info',
        )
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get('type')
        if event_type == 'message_start':
            self.state.status = 'Querying model'
            self.state.phase = 'Thinking'
            self.state.phase_detail = 'Model call started'
            self._update_active_turn(
                assistant_status='Thinking',
                phase_label='Thinking',
            )
            self._upsert_activity(
                'model',
                label='Model thinking',
                detail='Waiting for assistant tokens',
                status='running',
            )
            self._write_status('[thinking] model call started')
            self._publish_turns()
            self._publish_activity()
            self._publish_state()
            return
        if event_type == 'content_delta':
            self._render_assistant_delta(str(event.get('delta', '')))
            return
        if event_type == 'tool_call_delta':
            self._render_tool_plan(event)
            return
        if event_type == 'message_stop':
            finish_reason = event.get('finish_reason')
            if finish_reason == 'tool_calls':
                self.state.status = 'Dispatching tool call'
                self.state.phase = 'Tool dispatch'
                self.state.phase_detail = 'Assistant requested tools'
                self._update_active_turn(
                    assistant_status='Working',
                    phase_label='Tool dispatch',
                )
                self._write_status('[thinking] dispatching tool call')
            elif finish_reason == 'length':
                self.state.status = 'Continuation requested'
                self.state.phase = 'Continuation'
                self.state.phase_detail = 'Response hit the length limit'
                self._update_active_turn(
                    assistant_status='Continuation',
                    phase_label='Continuation',
                )
                self._write_status('[thinking] model output hit the length limit')
            else:
                self.state.phase = 'Finalizing'
                self.state.phase_detail = 'Wrapping up assistant response'
                self._update_active_turn(
                    assistant_status='Finalizing',
                    phase_label='Finalizing',
                )
                self._close_open_blocks()
            self._publish_turns()
            self._publish_state()
            return
        if event_type == 'tool_start':
            self._handle_tool_start(event)
            return
        if event_type == 'tool_delta':
            self._render_tool_delta(event)
            return
        if event_type == 'tool_result':
            self._handle_tool_result(event)
            return
        if event_type == 'usage':
            usage = event.get('usage')
            if isinstance(usage, dict):
                parts = [
                    f"input={usage.get('input_tokens', 0)}",
                    f"output={usage.get('output_tokens', 0)}",
                ]
                reasoning = usage.get('reasoning_tokens', 0)
                if reasoning:
                    parts.append(f'reasoning={reasoning}')
                self._write_status('[usage] ' + ' '.join(parts))
                self._upsert_activity(
                    'usage',
                    label='Usage update',
                    detail=' '.join(parts),
                    status='info',
                )
                self._publish_activity()
            return
        if event_type == 'continuation_request':
            self.state.status = 'Continuation requested'
            self.state.phase = 'Continuation'
            self.state.phase_detail = 'Requesting a follow-up model call'
            self._update_active_turn(
                assistant_status='Continuation',
                phase_label='Continuation',
            )
            self._write_status('[thinking] requesting continuation')
            self._publish_turns()
            self._publish_state()
            return
        if event_type == 'tool_permission_denial':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self.state.status = 'Permission denied'
            self.state.phase = 'Blocked'
            self.state.phase_detail = reason or 'Tool permission denied'
            self._update_active_turn(
                assistant_status='Blocked',
                phase_label='Permission denied',
            )
            self._append_turn_notice(
                kind='error',
                title='Permission Denied',
                content=reason,
                status='error',
            )
            self._upsert_activity(
                'permission',
                label='Tool permission denied',
                detail=reason,
                status='error',
            )
            self._write_status(f'[tool] permission denied: {reason}')
            self._publish_turns()
            self._publish_activity()
            self._publish_state()
            return
        if event_type == 'task_budget_exceeded':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self.state.status = 'Budget exceeded'
            self.state.phase = 'Budget exceeded'
            self.state.phase_detail = reason or 'Budget limit reached'
            self._update_active_turn(
                assistant_status='Budget exceeded',
                phase_label='Budget exceeded',
            )
            self._append_turn_notice(
                kind='error',
                title='Budget Exceeded',
                content=reason,
                status='error',
            )
            self._upsert_activity(
                'budget',
                label='Budget exceeded',
                detail=reason,
                status='error',
            )
            self._write_status(f'[budget] {reason}')
            self._publish_turns()
            self._publish_activity()
            self._publish_state()
            return
        if event_type == 'plugin_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self.state.status = 'Plugin blocked tool'
            self.state.phase = 'Blocked'
            self.state.phase_detail = message or 'Plugin blocked a tool'
            self._append_turn_notice(
                kind='warning',
                title='Plugin Block',
                content=message,
                status='warn',
            )
            self._upsert_activity(
                'plugin_block',
                label='Plugin blocked tool',
                detail=message,
                status='warn',
            )
            self._write_status(f'[plugin] blocked tool: {message}')
            self._publish_activity()
            self._publish_state()
            return
        if event_type == 'hook_policy_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self.state.status = 'Policy blocked tool'
            self.state.phase = 'Blocked'
            self.state.phase_detail = message or 'Policy blocked a tool'
            self._append_turn_notice(
                kind='warning',
                title='Policy Block',
                content=message,
                status='warn',
            )
            self._upsert_activity(
                'policy_block',
                label='Policy blocked tool',
                detail=message,
                status='warn',
            )
            self._write_status(f'[policy] blocked tool: {message}')
            self._publish_activity()
            self._publish_state()
            return
        if event_type in {
            'plugin_tool_preflight',
            'hook_policy_tool_preflight',
            'plugin_tool_hook',
            'hook_policy_tool_hook',
            'plugin_tool_context',
            'plugin_after_turn',
            'hook_policy_after_turn',
        }:
            message = _preview_value(event.get('message'), max_chars=220)
            tool_name = _preview_value(event.get('tool_name'))
            label = event_type.replace('_', ' ')
            detail = message or tool_name
            if detail:
                self._append_turn_notice(
                    kind='status',
                    title=label.replace('_', ' ').title(),
                    content=detail,
                    status='info',
                )
            self._upsert_activity(
                event_type,
                label=label,
                detail=detail,
                status='info',
            )
            self._publish_activity()
            return

    def complete(self, result: AgentRunResult) -> None:
        self._close_open_blocks()
        active_turn = self._active_turn()
        if active_turn is not None:
            if result.final_output and not active_turn.assistant_response:
                active_turn.assistant_response = result.final_output
                self._append_turn_entry(
                    kind='assistant',
                    title='Assistant',
                    content=result.final_output,
                    status='ok',
                    merge_key='assistant',
                )
            active_turn.assistant_status = (
                'Ready'
                if _friendly_stop_reason(result.stop_reason) == 'completed'
                else 'Stopped'
            )
            active_turn.phase_label = 'Completed'
            active_turn.stop_reason = result.stop_reason
            active_turn.session_id = result.session_id or active_turn.session_id
        if not self._streamed_assistant_output and result.final_output:
            self._emit_data(f'[assistant] {result.final_output}\n')
        self.state.busy = False
        self.state.status = 'Ready'
        self.state.phase = 'Ready'
        self.state.phase_detail = 'Waiting for the next prompt'
        self.state.session_id = result.session_id or self.state.session_id
        self.state.last_turns = result.turns
        self.state.last_tool_calls = result.tool_calls
        self.state.total_tokens = result.usage.total_tokens
        self.state.input_tokens = result.usage.input_tokens
        self.state.output_tokens = result.usage.output_tokens
        self.state.total_cost_usd = result.total_cost_usd
        self.state.last_stop_reason = result.stop_reason
        self._upsert_activity(
            'run_complete',
            label='Run completed',
            detail=(
                f'turns={result.turns} tool_calls={result.tool_calls} '
                f'reason={_friendly_stop_reason(result.stop_reason)}'
            ),
            status='ok',
        )
        if result.stop_reason:
            self._emit_data(f'[status] stop_reason={result.stop_reason}\n')
        if result.session_id:
            self._emit_data(f'[session] session_id={result.session_id}\n')
        self._emit_data(
            '[usage] '
            f'total_tokens={result.usage.total_tokens} '
            f'input_tokens={result.usage.input_tokens} '
            f'output_tokens={result.usage.output_tokens} '
            f'total_cost_usd={result.total_cost_usd:.6f}\n'
        )
        if result.scratchpad_directory:
            self._emit_data(f'[scratchpad] {result.scratchpad_directory}\n')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def fail(self, error: BaseException) -> None:
        self._close_open_blocks()
        active_turn = self._active_turn()
        if active_turn is not None:
            active_turn.assistant_status = 'Error'
            active_turn.phase_label = 'Error'
            if not active_turn.assistant_response:
                active_turn.assistant_response = f'Error: {error}'
            self._append_turn_notice(
                kind='error',
                title='Error',
                content=str(error),
                status='error',
            )
        self.state.busy = False
        self.state.status = 'Error'
        self.state.phase = 'Error'
        self.state.phase_detail = _preview_value(str(error), max_chars=220)
        self.state.last_stop_reason = error.__class__.__name__
        self._upsert_activity(
            'error',
            label='Run failed',
            detail=_preview_value(str(error), max_chars=220),
            status='error',
        )
        self._emit_data(f'[error] {error}\n')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def reset_display(self) -> None:
        self._assistant_open = False
        self._assistant_ends_with_newline = True
        self._tool_stream_key = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True
        self._streamed_assistant_output = False
        self._activity.clear()
        self._activity_index.clear()
        self._tool_stream_buffers.clear()
        self.state.activity_events = 0
        self._publish_activity()
        self._publish_state()

    def _handle_tool_start(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        tool_call_id = (
            str(event['tool_call_id'])
            if isinstance(event.get('tool_call_id'), str)
            else _preview_value(tool_name)
        )
        if isinstance(tool_name, str) and tool_name:
            self.state.last_tool = tool_name
            self.state.status = f'Tool: {tool_name}'
            self.state.phase = 'Running tool'
            self.state.phase_detail = tool_name
        self._update_active_turn(
            assistant_status='Working',
            phase_label=(f'Tool: {tool_name}' if isinstance(tool_name, str) and tool_name else 'Tool'),
            increment_tool_count=True,
        )
        self._append_turn_notice(
            kind='tool',
            title=f'Tool Call: {tool_name or "tool"}',
            content=self._render_tool_start(event),
            status='info',
        )
        self._upsert_activity(
            f'tool:{tool_call_id}',
            label='Tool started',
            detail=self._render_tool_start(event),
            status='running',
        )
        self._write_status(self._render_tool_start(event))
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _render_assistant_delta(self, delta: str) -> None:
        if not delta:
            return
        self._close_tool_stream()
        if not self._assistant_open:
            self._emit_data('[assistant] ')
            self._assistant_open = True
            self._assistant_ends_with_newline = False
        self._emit_data(delta)
        self._streamed_assistant_output = True
        self._assistant_ends_with_newline = delta.endswith('\n')
        active_turn = self._active_turn()
        if active_turn is not None:
            active_turn.assistant_response += delta
            active_turn.assistant_status = 'Responding'
            active_turn.phase_label = 'Responding'
        self._append_turn_entry(
            kind='assistant',
            title='Assistant',
            content=delta,
            status='ok',
            merge_key='assistant',
        )
        self.state.phase = 'Responding'
        self.state.phase_detail = 'Assistant is composing a reply'
        self._publish_turns()
        self._publish_state()

    def _render_tool_plan(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        if not isinstance(tool_name, str) or not tool_name:
            return
        key = str(event.get('tool_call_id') or f"index:{event.get('tool_call_index', 0)}")
        if key in self._announced_tool_plans:
            return
        self._announced_tool_plans.add(key)
        self.state.phase = 'Planning tool'
        self.state.phase_detail = tool_name
        self._update_active_turn(
            assistant_status='Planning tool',
            phase_label=f'Planning {tool_name}',
        )
        self._append_turn_notice(
            kind='thinking',
            title='Thinking',
            content=f'Planning tool call: {tool_name}',
            status='info',
        )
        self._upsert_activity(
            f'plan:{key}',
            label='Tool planned',
            detail=tool_name,
            status='info',
        )
        self._write_status(f'[thinking] planning tool call: {tool_name}')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _render_tool_start(self, event: dict[str, object]) -> str:
        tool_name = event.get('tool_name')
        arguments = event.get('arguments')
        if not isinstance(tool_name, str) or not tool_name:
            return '[tool] starting'
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name == 'bash':
            command = _preview_value(arguments.get('command'))
            return f'[command] {command or "(empty command)"}'
        if tool_name in {'write_file', 'edit_file', 'read_file', 'notebook_edit'}:
            path = _preview_value(arguments.get('path'))
            return f'[file] {tool_name} {path or "(unknown path)"}'
        if tool_name == 'mcp_call_tool':
            server = _preview_value(arguments.get('server')) or '(auto)'
            remote_tool = _preview_value(arguments.get('tool_name')) or '(unknown)'
            return f'[mcp] server={server} tool={remote_tool}'
        if tool_name.startswith('mcp_'):
            summary = _preview_value(arguments)
            return f'[mcp] {tool_name} {summary}'.rstrip()
        summary = _preview_value(arguments)
        return f'[tool] {tool_name} {summary}'.rstrip() if summary else f'[tool] {tool_name}'

    def _render_tool_delta(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        if not isinstance(tool_name, str) or not tool_name:
            return
        if tool_name != 'bash' and not tool_name.startswith('mcp_'):
            return
        delta = str(event.get('delta', ''))
        if not delta:
            return
        stream_name = event.get('stream')
        if not isinstance(stream_name, str) or not stream_name:
            stream_name = 'tool'
        key = (str(event.get('tool_call_id')), stream_name)
        self._close_assistant()
        if self._tool_stream_key != key:
            self._close_tool_stream()
            self._emit_data(f'[tool-output:{tool_name}:{stream_name}]\n')
            self._tool_stream_key = key
            self._tool_stream_open = True
            self._tool_stream_ends_with_newline = True
        self._emit_data(delta)
        self._tool_stream_open = True
        self._tool_stream_ends_with_newline = delta.endswith('\n')
        stream_key = f'{key[0]}:{stream_name}'
        previous = self._tool_stream_buffers.get(stream_key, '')
        combined = previous + delta
        self._tool_stream_buffers[stream_key] = combined[-400:]
        self._append_turn_entry(
            kind='tool_output',
            title=f'{tool_name} {stream_name}',
            content=delta,
            status='running',
            merge_key=f'tool-output:{stream_key}',
        )
        self._upsert_activity(
            f'stream:{stream_key}',
            label=f'{tool_name} {stream_name}',
            detail=_preview_multiline(self._tool_stream_buffers[stream_key]),
            status='running',
        )
        self._publish_activity()

    def _handle_tool_result(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        ok = bool(event.get('ok'))
        tool_call_id = (
            str(event['tool_call_id'])
            if isinstance(event.get('tool_call_id'), str)
            else _preview_value(tool_name)
        )
        self.state.status = 'Processing tool result'
        self.state.phase = 'Processing result'
        self.state.phase_detail = _preview_value(tool_name) or 'Tool result'
        self._update_active_turn(
            assistant_status='Processing result',
            phase_label='Processing result',
        )
        self._append_turn_notice(
            kind='tool_result',
            title='Tool Result',
            content=self._render_tool_result(event),
            status='ok' if ok else 'error',
        )
        self._upsert_activity(
            f'tool:{tool_call_id}',
            label='Tool finished',
            detail=self._render_tool_result(event),
            status='ok' if ok else 'error',
        )
        self._write_status(self._render_tool_result(event))
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _render_tool_result(self, event: dict[str, object]) -> str:
        tool_name = event.get('tool_name')
        ok = bool(event.get('ok'))
        metadata = event.get('metadata')
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = 'tool'
        if not isinstance(metadata, dict):
            metadata = {}
        action = metadata.get('action')
        path = metadata.get('path')
        if action == 'bash':
            exit_code = metadata.get('exit_code')
            return f'[command] exit_code={exit_code} ok={ok}'
        if isinstance(path, str) and path:
            return f'[file] updated {path}'
        if action == 'mcp_call_tool' or tool_name.startswith('mcp_'):
            server = _preview_value(metadata.get('server_name')) or '(auto)'
            remote_tool = _preview_value(metadata.get('tool_name')) or tool_name
            return f'[mcp] server={server} tool={remote_tool} ok={ok}'
        cwd_update = metadata.get('cwd_update')
        if isinstance(cwd_update, str) and cwd_update:
            return f'[cwd] {cwd_update}'
        return f'[tool] {tool_name} ok={ok}'

    def _write_status(self, line: str) -> None:
        if not line:
            return
        self._close_open_blocks()
        self._emit_data(line + '\n')

    def _close_open_blocks(self) -> None:
        self._close_assistant()
        self._close_tool_stream()

    def _close_assistant(self) -> None:
        if self._assistant_open and not self._assistant_ends_with_newline:
            self._emit_data('\n')
        self._assistant_open = False
        self._assistant_ends_with_newline = True

    def _close_tool_stream(self) -> None:
        if self._tool_stream_open and not self._tool_stream_ends_with_newline:
            self._emit_data('\n')
        self._tool_stream_key = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True

    def _active_turn(self) -> ConversationTurn | None:
        if not self._turns or self._active_turn_id is None:
            return self._turns[-1] if self._turns else None
        for turn in reversed(self._turns):
            if turn.turn_id == self._active_turn_id:
                return turn
        return self._turns[-1]

    def _update_active_turn(
        self,
        *,
        assistant_status: str | None = None,
        phase_label: str | None = None,
        increment_tool_count: bool = False,
    ) -> None:
        turn = self._active_turn()
        if turn is None:
            return
        if assistant_status is not None:
            turn.assistant_status = assistant_status
        if phase_label is not None:
            turn.phase_label = phase_label
        if increment_tool_count:
            turn.tool_count += 1

    def _append_turn_entry(
        self,
        *,
        kind: str,
        title: str,
        content: str = '',
        status: str = 'info',
        merge_key: str | None = None,
    ) -> None:
        turn = self._active_turn()
        if turn is None:
            return
        if merge_key and turn.entries and turn.entries[-1].merge_key == merge_key:
            entry = turn.entries[-1]
            entry.content += content
            entry.status = status
            return
        turn.entries.append(
            ConversationEntry(
                entry_id=f'{turn.turn_id}-entry-{len(turn.entries) + 1}',
                kind=kind,
                title=title,
                content=content,
                status=status,
                merge_key=merge_key,
            )
        )

    def _append_turn_notice(
        self,
        *,
        kind: str,
        title: str,
        content: str,
        status: str = 'info',
    ) -> None:
        if not content:
            return
        self._append_turn_entry(
            kind=kind,
            title=title,
            content=content,
            status=status,
        )

    def _upsert_activity(
        self,
        key: str,
        *,
        label: str,
        detail: str = '',
        status: str = 'info',
    ) -> None:
        item = ActivityItem(
            key=key,
            label=label,
            detail=detail,
            status=status,
        )
        index = self._activity_index.get(key)
        if index is None:
            self._activity.append(item)
        else:
            self._activity[index] = item
        if len(self._activity) > MAX_ACTIVITY_ITEMS:
            self._activity = self._activity[-MAX_ACTIVITY_ITEMS:]
        self._activity_index = {
            activity.key: idx for idx, activity in enumerate(self._activity)
        }
        self.state.activity_events = len(self._activity)

    def _publish_turns(self) -> None:
        self.state.conversation_turns = len(self._turns)
        if self._on_turns_change is not None:
            self._on_turns_change(tuple(self._turns))
        if self._on_history_change is not None:
            self._on_history_change(build_conversation_history_items(self._turns))

    def _publish_activity(self) -> None:
        if self._on_activity_change is not None:
            self._on_activity_change(tuple(self._activity))

    def _publish_state(self) -> None:
        if self._on_state_change is not None:
            self._on_state_change(self.state)
