from __future__ import annotations

import json
from typing import Callable, Sequence

from src.agent.models.types import AgentRunResult
from src.ui.conversation import (
    ActivityItem,
    ConversationEntry,
    ConversationHistoryItem,
    ConversationTurn,
    build_conversation_history_items,
)
from src.ui.formatting import (
    _friendly_stop_reason,
    _preview_multiline,
    _preview_value,
    sanitize_assistant_display_text,
)
from src.ui.state import AgentTuiState


MAX_ACTIVITY_ITEMS = 14


def _coerce_positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


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
        on_changes_change: Callable[[dict[str, object] | None], None] | None = None,
    ) -> None:
        self.state = state
        self._emit_data = emit_data
        self._on_state_change = on_state_change
        self._on_turns_change = on_turns_change
        self._on_history_change = on_history_change
        self._on_activity_change = on_activity_change
        self._on_changes_change = on_changes_change
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
        self._delegate_output_buffers: dict[str, str] = {}
        self._active_turn_id: str | None = None
        self._workspace_change_summary: dict[str, object] | None = None

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
        self._workspace_change_summary = None
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
        self._publish_changes()
        self._publish_state()

    def begin_prompt(self, prompt: str, *, session_id: str | None = None) -> None:
        self._close_open_blocks()
        self._announced_tool_plans.clear()
        self._tool_stream_buffers.clear()
        self._delegate_output_buffers.clear()
        self._workspace_change_summary = None
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
        self._publish_changes()
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
        if event_type == 'workspace_change':
            self._handle_workspace_change(event)
            return
        if event_type == 'workspace_change_summary':
            self._handle_workspace_change_summary(event)
            return
        if event_type == 'workspace_change_recap':
            self._handle_workspace_change_recap(event)
            return
        if event_type == 'delegate_subtask_start':
            self._handle_delegate_subtask_start(event)
            return
        if event_type == 'delegate_subtask_event':
            self._handle_delegate_subtask_event(event)
            return
        if event_type == 'delegate_batch_result':
            self._handle_delegate_batch_result(event)
            return
        if event_type == 'delegate_subtask_result':
            self._handle_delegate_subtask_result(event)
            return
        if event_type == 'delegate_group_result':
            self._handle_delegate_group_result(event)
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
            'plugin_delegate_preflight',
            'plugin_delegate_after',
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
        friendly_stop = _friendly_stop_reason(result.stop_reason)
        completed = friendly_stop == 'completed'
        active_turn = self._active_turn()
        if active_turn is not None:
            final_output = sanitize_assistant_display_text(result.final_output)
            if final_output and not active_turn.assistant_response:
                active_turn.assistant_response = final_output
                self._append_turn_entry(
                    kind='assistant',
                    title='Assistant',
                    content=final_output,
                    status='ok',
                    merge_key='assistant',
                )
            active_turn.assistant_status = 'Ready' if completed else 'Stopped'
            active_turn.phase_label = 'Completed' if completed else 'Stopped'
            active_turn.stop_reason = result.stop_reason
            active_turn.session_id = result.session_id or active_turn.session_id
            if not completed:
                self._append_turn_notice(
                    kind='warning',
                    title='Run Stopped',
                    content=(
                        f'stop_reason={friendly_stop}\n'
                        f'turns={result.turns} tool_calls={result.tool_calls}'
                    ),
                    status='warn',
                )
        if not self._streamed_assistant_output and result.final_output:
            final_output = sanitize_assistant_display_text(result.final_output)
            if final_output:
                self._emit_data(f'[assistant] {final_output}\n')
        self._append_workspace_change_recap_from_result(result)
        self.state.busy = False
        self.state.status = 'Ready'
        self.state.phase = 'Ready'
        self.state.phase_detail = (
            'Waiting for the next prompt'
            if completed
            else f'Waiting for the next prompt; last run stopped: {friendly_stop}'
        )
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
                f'reason={friendly_stop}'
            ),
            status='ok' if completed else 'warn',
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

    def _append_workspace_change_recap_from_result(
        self,
        result: AgentRunResult,
    ) -> None:
        for event in reversed(result.events):
            if (
                isinstance(event, dict)
                and event.get('type') == 'workspace_change_recap'
            ):
                self._handle_workspace_change_recap(event)
                return

    def request_cancel(self, reason: str = 'Stop requested') -> None:
        self.state.status = 'Stopping'
        self.state.phase = 'Stopping'
        self.state.phase_detail = reason
        self._update_active_turn(
            assistant_status='Stopping',
            phase_label='Stopping',
        )
        self._append_turn_notice(
            kind='warning',
            title='Stop Requested',
            content=reason,
            status='warn',
        )
        self._upsert_activity(
            'cancel_requested',
            label='Stop requested',
            detail=reason,
            status='warn',
        )
        self._emit_data(f'[status] {reason}\n')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def cancel(self, reason: str = 'Stopped by user') -> None:
        self._close_open_blocks()
        active_turn = self._active_turn()
        if active_turn is not None:
            active_turn.assistant_status = 'Stopped'
            active_turn.phase_label = 'Stopped'
            active_turn.stop_reason = 'cancelled'
            self._append_turn_notice(
                kind='warning',
                title='Stopped',
                content=reason,
                status='warn',
            )
        self.state.busy = False
        self.state.status = 'Ready'
        self.state.phase = 'Ready'
        self.state.phase_detail = 'Waiting for the next prompt'
        self.state.last_stop_reason = 'cancelled'
        self._upsert_activity(
            'cancelled',
            label='Run stopped',
            detail=reason,
            status='warn',
        )
        self._emit_data(f'[status] stop_reason=cancelled ({reason})\n')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def fail(self, error: BaseException) -> None:
        self._close_open_blocks()
        active_turn = self._active_turn()
        stop_reason = error.__class__.__name__
        if active_turn is not None:
            active_turn.assistant_status = 'Error'
            active_turn.phase_label = 'Error'
            active_turn.stop_reason = stop_reason
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
        self.state.last_stop_reason = stop_reason
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
        self._delegate_output_buffers.clear()
        self._workspace_change_summary = None
        self.state.activity_events = 0
        self._publish_activity()
        self._publish_changes()
        self._publish_state()

    def _handle_delegate_subtask_start(self, event: dict[str, object]) -> None:
        label = _preview_value(event.get('label')) or 'subtask'
        self.state.status = f'Sub-agent: {label}'
        self.state.phase = 'Delegating'
        self.state.phase_detail = f'{label} started'
        self._update_active_turn(
            assistant_status='Delegating',
            phase_label=f'Sub-agent: {label}',
        )
        self._upsert_turn_entry(
            kind='delegate_progress',
            title=f'Sub-Agent Progress: {label}',
            content='starting',
            status='running',
            merge_key=self._delegate_progress_key(event),
        )
        self._upsert_activity(
            self._delegate_activity_key(event),
            label='Sub-agent running',
            detail=f'{label}: started',
            status='running',
        )
        self._write_status(f'[delegate] started {label}')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _handle_delegate_subtask_event(self, event: dict[str, object]) -> None:
        if event.get('child_event_type') == 'workspace_change':
            summary = self._render_workspace_change_summary(event)
            if summary:
                event = {**event, 'summary': summary}
        if event.get('child_event_type') == 'content_delta':
            self._handle_delegate_content_delta(event)
            return
        label = _preview_value(event.get('label')) or 'subtask'
        summary = self._render_delegate_subtask_event(event)
        if not summary:
            return
        status = self._delegate_event_status(event)
        self.state.status = f'Sub-agent: {label}'
        self.state.phase = 'Delegating'
        self.state.phase_detail = summary
        self._update_active_turn(
            assistant_status='Delegating',
            phase_label=f'Sub-agent: {label}',
        )
        self._upsert_turn_entry(
            kind='delegate_progress',
            title=f'Sub-Agent Progress: {label}',
            content=summary,
            status=status,
            merge_key=self._delegate_progress_key(event),
        )
        self._upsert_activity(
            self._delegate_activity_key(event),
            label='Sub-agent running',
            detail=f'{label}: {summary}',
            status=status,
        )
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _handle_delegate_content_delta(self, event: dict[str, object]) -> None:
        label = _preview_value(event.get('label')) or 'subtask'
        delta = event.get('delta')
        if not isinstance(delta, str) or not delta:
            delta = _preview_value(event.get('delta_preview'), max_chars=180)
        if not delta:
            return
        delta = sanitize_assistant_display_text(delta)
        if not delta:
            return
        output_key = self._delegate_output_key(event)
        previous = self._delegate_output_buffers.get(output_key, '')
        combined = previous + delta
        self._delegate_output_buffers[output_key] = combined
        self.state.status = f'Sub-agent: {label}'
        self.state.phase = 'Delegating'
        self.state.phase_detail = f'{label} is responding'
        self._update_active_turn(
            assistant_status='Delegating',
            phase_label=f'Sub-agent: {label}',
        )
        self._append_or_merge_turn_entry(
            kind='delegate_output',
            title=f'Sub-Agent Output: {label}',
            content=delta,
            status='running',
            merge_key=output_key,
        )
        self._upsert_activity(
            self._delegate_activity_key(event),
            label='Sub-agent running',
            detail=f'{label}: {_preview_multiline(combined)}',
            status='running',
        )
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _handle_delegate_batch_result(self, event: dict[str, object]) -> None:
        batch_index = _preview_value(event.get('batch_index'))
        status = _preview_value(event.get('status')) or 'completed'
        labels = event.get('labels')
        if isinstance(labels, list):
            label_text = ', '.join(str(label) for label in labels)
        else:
            label_text = _preview_value(labels)
        title = f'Delegate Batch {batch_index or ""}'.strip()
        content = (
            f'status={status}\n'
            f'labels={label_text or "(none)"}\n'
            f'completed={event.get("completed_children", 0)} '
            f'failed={event.get("failed_children", 0)} '
            f'skipped={event.get("skipped_children", 0)}'
        )
        self._append_turn_notice(
            kind='status',
            title=title,
            content=content,
            status='ok' if status == 'completed' else 'warn',
        )
        self._upsert_activity(
            f'delegate_batch:{batch_index or "unknown"}',
            label='Delegate batch finished',
            detail=f'{status}: {label_text or "(none)"}',
            status='ok' if status == 'completed' else 'warn',
        )
        self._publish_turns()
        self._publish_activity()

    def _handle_delegate_subtask_result(self, event: dict[str, object]) -> None:
        label = _preview_value(event.get('label')) or 'subtask'
        stop_reason = _preview_value(event.get('stop_reason')) or 'stop'
        output_preview = _preview_value(event.get('output_preview'), max_chars=260)
        full_output = self._delegate_result_output(event)
        output_key = self._delegate_output_key(event)
        result_status = (
            'ok' if stop_reason not in {'backend_error', 'budget_exceeded'} else 'error'
        )
        if full_output:
            existing_output = self._find_turn_entry_by_merge_key(output_key)
            if existing_output is None:
                self._append_turn_entry(
                    kind='delegate_output',
                    title=f'Sub-Agent Output: {label}',
                    content=full_output,
                    status=result_status,
                    merge_key=output_key,
                )
            else:
                existing_output.content = full_output
                existing_output.status = result_status
            self._delegate_output_buffers[output_key] = full_output
        content = (
            f'label={label}\n'
            f'batch_index={event.get("batch_index", "")}\n'
            f'session_id={_preview_value(event.get("session_id"))}\n'
            f'turns={event.get("turns", 0)} tool_calls={event.get("tool_calls", 0)}\n'
            f'stop_reason={stop_reason}'
        )
        if output_preview and not full_output:
            content += f'\noutput_preview={output_preview}'
        self._append_turn_notice(
            kind='delegate_result',
            title=f'Sub-Agent: {label}',
            content=content,
            status=result_status,
        )
        self._upsert_activity(
            f'delegate_child:{label}',
            label='Sub-agent finished',
            detail=f'{label}: {stop_reason}',
            status='ok' if stop_reason not in {'backend_error', 'budget_exceeded'} else 'error',
        )
        self._publish_turns()
        self._publish_activity()

    def _render_delegate_subtask_event(self, event: dict[str, object]) -> str:
        child_event_type = _preview_value(event.get('child_event_type'))
        if child_event_type == 'message_start':
            return 'model call started'
        if child_event_type == 'message_stop':
            reason = _preview_value(event.get('finish_reason')) or 'stop'
            return f'model call finished reason={reason}'
        if child_event_type == 'tool_start':
            return self._render_delegate_tool_start(event)
        if child_event_type == 'tool_result':
            return self._render_delegate_tool_result(event)
        if child_event_type == 'workspace_change':
            summary = _preview_value(event.get('summary'), max_chars=180)
            if summary:
                return f'workspace changed: {summary}'
            return 'workspace changed'
        if child_event_type == 'tool_permission_denial':
            reason = _preview_value(event.get('reason'), max_chars=180)
            return f'permission denied: {reason or "tool blocked"}'
        if child_event_type == 'task_budget_exceeded':
            reason = _preview_value(event.get('reason'), max_chars=180)
            return f'budget exceeded: {reason or "limit reached"}'
        if child_event_type == 'continuation_request':
            return 'requested response continuation'
        if child_event_type == 'usage':
            usage = event.get('usage')
            if isinstance(usage, dict):
                return (
                    f"usage input={usage.get('input_tokens', 0)} "
                    f"output={usage.get('output_tokens', 0)}"
                )
            return 'usage updated'
        if child_event_type in {
            'plugin_tool_preflight',
            'hook_policy_tool_preflight',
            'plugin_tool_hook',
            'hook_policy_tool_hook',
            'plugin_tool_context',
            'plugin_delegate_preflight',
            'plugin_delegate_after',
        }:
            tool_name = _preview_value(event.get('tool_name')) or 'tool'
            return f'{child_event_type.replace("_", " ")}: {tool_name}'
        if child_event_type == 'content_delta':
            return ''
        return child_event_type

    def _render_delegate_tool_start(self, event: dict[str, object]) -> str:
        tool_name = _preview_value(event.get('tool_name')) or 'tool'
        arguments = event.get('arguments')
        if not isinstance(arguments, dict):
            return f'tool started: {tool_name}'
        if tool_name == 'bash':
            command = _preview_value(arguments.get('command'), max_chars=160)
            return f'tool started: bash {command or "(empty command)"}'
        if tool_name in {'write_file', 'edit_file', 'read_file', 'notebook_edit'}:
            path = _preview_value(arguments.get('path'), max_chars=180)
            return f'tool started: {tool_name} {path or "(unknown path)"}'
        return f'tool started: {tool_name}'

    def _render_delegate_tool_result(self, event: dict[str, object]) -> str:
        tool_name = _preview_value(event.get('tool_name')) or 'tool'
        ok = bool(event.get('ok'))
        metadata = event.get('metadata')
        if not isinstance(metadata, dict):
            metadata = {}
        action = metadata.get('action')
        if action == 'bash':
            exit_code = metadata.get('exit_code')
            return f'tool finished: bash ok={ok} exit_code={exit_code}'
        path = metadata.get('path')
        if isinstance(path, str) and path:
            return f'tool finished: {tool_name} ok={ok} path={path}'
        preview = _preview_value(
            metadata.get('output_preview')
            or metadata.get('preview')
            or event.get('content_preview'),
            max_chars=160,
        )
        if preview:
            return f'tool finished: {tool_name} ok={ok} {preview}'
        return f'tool finished: {tool_name} ok={ok}'

    def _delegate_event_status(self, event: dict[str, object]) -> str:
        child_event_type = event.get('child_event_type')
        if child_event_type in {'tool_permission_denial', 'task_budget_exceeded'}:
            return 'error'
        if child_event_type == 'tool_result' and not bool(event.get('ok')):
            return 'error'
        return 'running'

    def _delegate_activity_key(self, event: dict[str, object]) -> str:
        label = _preview_value(event.get('label')) or 'subtask'
        batch_index = _preview_value(event.get('batch_index')) or 'unknown'
        return f'delegate_child:{batch_index}:{label}'

    def _delegate_output_key(self, event: dict[str, object]) -> str:
        label = _preview_value(event.get('label')) or 'subtask'
        batch_index = _preview_value(event.get('batch_index')) or 'unknown'
        return f'delegate-output:{batch_index}:{label}'

    def _delegate_progress_key(self, event: dict[str, object]) -> str:
        label = _preview_value(event.get('label')) or 'subtask'
        batch_index = _preview_value(event.get('batch_index')) or 'unknown'
        return f'delegate-progress:{batch_index}:{label}'

    def _delegate_result_output(self, event: dict[str, object]) -> str:
        for key in ('output', 'final_output'):
            value = event.get(key)
            if isinstance(value, str) and value:
                return sanitize_assistant_display_text(value)
        return ''

    def _render_list_field(self, value: object) -> str:
        if isinstance(value, list):
            rendered = ', '.join(str(item) for item in value if str(item))
            return rendered or '(none)'
        rendered = _preview_value(value)
        return rendered or '(none)'

    def _handle_delegate_group_result(self, event: dict[str, object]) -> None:
        group_id = _preview_value(event.get('group_id')) or 'group'
        group_status = _preview_value(event.get('group_status')) or 'completed'
        strategy = _preview_value(event.get('strategy')) or 'serial'
        content = (
            f'group_id={group_id}\n'
            f'status={group_status}\n'
            f'strategy={strategy}\n'
            f'subtasks={event.get("subtask_count", 0)} '
            f'completed={event.get("completed_children", 0)} '
            f'failed={event.get("failed_children", 0)}'
        )
        self._append_turn_notice(
            kind='status',
            title='Sub-Agent Group',
            content=content,
            status='ok' if group_status == 'completed' else 'warn',
        )
        self._upsert_activity(
            f'delegate_group:{group_id}',
            label='Sub-agent group finished',
            detail=f'{group_status}; strategy={strategy}',
            status='ok' if group_status == 'completed' else 'warn',
        )
        self._publish_turns()
        self._publish_activity()

    def _handle_tool_start(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        rendered_start = self._render_tool_start(event)
        delegate_tool = self._is_delegate_tool(tool_name)
        tool_call_id = (
            str(event['tool_call_id'])
            if isinstance(event.get('tool_call_id'), str)
            else _preview_value(tool_name)
        )
        if isinstance(tool_name, str) and tool_name:
            self.state.last_tool = tool_name
            if delegate_tool:
                self.state.status = 'Sub-agent requested'
                self.state.phase = 'Delegating'
                self.state.phase_detail = self._delegate_request_label(event)
            else:
                self.state.status = f'Tool: {tool_name}'
                self.state.phase = 'Running tool'
                self.state.phase_detail = tool_name
        self._update_active_turn(
            assistant_status='Delegating' if delegate_tool else 'Working',
            phase_label=(
                self._delegate_request_label(event)
                if delegate_tool
                else (
                    f'Tool: {tool_name}'
                    if isinstance(tool_name, str) and tool_name
                    else 'Tool'
                )
            ),
            increment_tool_count=True,
        )
        self._append_turn_notice(
            kind='status' if delegate_tool else 'tool',
            title=(
                f'Sub-Agent Requested: {self._delegate_request_label(event)}'
                if delegate_tool
                else f'Tool Call: {tool_name or "tool"}'
            ),
            content=self._render_tool_start_detail(event, rendered_start),
            status='info',
        )
        self._upsert_activity(
            f'tool:{tool_call_id}',
            label='Sub-agent requested' if delegate_tool else 'Tool started',
            detail=rendered_start,
            status='running',
        )
        self._write_status(rendered_start)
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _render_assistant_delta(self, delta: str) -> None:
        if not delta:
            return
        delta = sanitize_assistant_display_text(delta)
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
        plan_content = self._render_tool_plan_summary(event)
        tool_call_id = _preview_value(event.get('tool_call_id'))
        if tool_call_id:
            plan_content += f' id={tool_call_id}'
        self.state.phase = 'Planning tool'
        self.state.phase_detail = tool_name
        self._update_active_turn(
            assistant_status='Planning tool',
            phase_label=f'Planning {tool_name}',
        )
        self._append_turn_notice(
            kind='thinking',
            title=f'Thinking: {tool_name}',
            content=plan_content,
            status='info',
        )
        self._upsert_activity(
            f'plan:{key}',
            label='Tool planned',
            detail=plan_content,
            status='info',
        )
        self._write_status(f'[thinking] {plan_content}')
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
        if self._is_delegate_tool(tool_name):
            return self._render_delegate_tool_request(arguments)
        if tool_name == 'bash':
            command = _preview_value(arguments.get('command'))
            return f'[command] {command or "(empty command)"}'
        if tool_name in {'write_file', 'edit_file', 'read_file', 'notebook_edit'}:
            path = _preview_value(arguments.get('path'))
            return f'[file] {tool_name} {path or "(unknown path)"}'
        if tool_name == 'mcp_call_tool':
            server = _preview_value(arguments.get('server')) or '(auto)'
            remote_tool = _preview_value(arguments.get('tool_name')) or '(unknown)'
            tool_arguments = arguments.get('arguments')
            tool_args_preview = _preview_value(tool_arguments, max_chars=240)
            if tool_args_preview:
                return f'[mcp] server={server} tool={remote_tool} args={tool_args_preview}'
            return f'[mcp] server={server} tool={remote_tool}'
        if tool_name.startswith('mcp_'):
            summary = _preview_value(arguments)
            return f'[mcp] {tool_name} {summary}'.rstrip()
        summary = _preview_value(arguments)
        return f'[tool] {tool_name} {summary}'.rstrip() if summary else f'[tool] {tool_name}'

    def _render_tool_start_detail(self, event: dict[str, object], summary: str) -> str:
        lines = [summary]
        tool_call_id = _preview_value(event.get('tool_call_id'))
        if tool_call_id:
            lines.append(f'tool_call_id={tool_call_id}')
        arguments = event.get('arguments')
        tool_name = event.get('tool_name')
        if self._is_delegate_tool(tool_name) and isinstance(arguments, dict):
            lines.extend(self._render_delegate_arguments(arguments))
            return '\n'.join(lines)
        if (
            isinstance(tool_name, str)
            and tool_name in {'write_file', 'edit_file', 'notebook_edit'}
            and isinstance(arguments, dict)
        ):
            lines.extend(self._render_file_write_arguments(arguments))
            return '\n'.join(lines)
        if isinstance(arguments, dict):
            lines.append('arguments:')
            lines.append(self._format_json_for_display(arguments, max_chars=1800))
        elif arguments is not None:
            lines.append('arguments:')
            lines.append(_preview_value(arguments, max_chars=1200))
        return '\n'.join(lines)

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
        tool_label = _preview_value(tool_name) or 'tool'
        delegate_tool = self._is_delegate_tool(tool_name)
        rendered_result = self._render_tool_result(event)
        rendered_result_detail = self._render_tool_result_detail(event, rendered_result)
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
            kind='delegate_result' if delegate_tool else 'tool_result',
            title=(
                f'Sub-Agent Tool Result: {tool_label}'
                if delegate_tool
                else f'Tool Result: {tool_label}'
            ),
            content=rendered_result_detail,
            status='ok' if ok else 'error',
        )
        self._upsert_activity(
            f'tool:{tool_call_id}',
            label='Sub-agent tool finished' if delegate_tool else 'Tool finished',
            detail=rendered_result,
            status='ok' if ok else 'error',
        )
        self._write_status(rendered_result)
        self._publish_turns()
        self._publish_activity()
        self._publish_changes()
        self._publish_state()

    def _handle_workspace_change(self, event: dict[str, object]) -> None:
        summary = self._render_workspace_change_summary(event)
        if not summary:
            return
        detail = self._render_workspace_change_detail(event, summary)
        sequence = _preview_value(event.get('sequence')) or 'latest'
        self.state.status = 'Workspace changed'
        self.state.phase = 'Workspace change'
        self.state.phase_detail = summary
        self.state.workspace_change_events += 1
        self.state.workspace_changed_files += _coerce_positive_int(
            event.get('file_count')
        )
        self.state.workspace_added_lines += _coerce_positive_int(
            event.get('added_lines')
        )
        self.state.workspace_removed_lines += _coerce_positive_int(
            event.get('removed_lines')
        )
        self._update_active_turn(
            assistant_status='Working',
            phase_label='Workspace changed',
        )
        self._append_turn_notice(
            kind='workspace_change',
            title='Workspace Changes',
            content=detail,
            status='ok',
        )
        self._upsert_activity(
            f'workspace_change:{sequence}',
            label='Workspace changed',
            detail=summary,
            status='ok',
        )
        self._write_status(f'[changes] {summary}')
        self._publish_turns()
        self._publish_activity()
        self._publish_state()

    def _handle_workspace_change_summary(self, event: dict[str, object]) -> None:
        if not self._render_workspace_change_recap_summary(event):
            return
        self._workspace_change_summary = dict(event)
        self._publish_changes()

    def _handle_workspace_change_recap(self, event: dict[str, object]) -> None:
        summary = self._render_workspace_change_recap_summary(event)
        if not summary:
            return
        self._workspace_change_summary = dict(event)
        detail = self._render_workspace_change_recap_detail(event, summary)
        self._upsert_turn_entry(
            kind='workspace_change_recap',
            title='Changed Files',
            content=detail,
            status='ok',
            merge_key='workspace-change-recap',
        )
        self._upsert_activity(
            'workspace_change_recap',
            label='Changed files recap',
            detail=summary,
            status='ok',
        )
        self._write_status(f'[changes-summary] {summary}')
        self._publish_turns()
        self._publish_activity()
        self._publish_changes()
        self._publish_state()

    def _render_workspace_change_summary(self, event: dict[str, object]) -> str:
        summary = _preview_value(event.get('summary'), max_chars=220)
        if summary:
            return summary
        file_count = _coerce_positive_int(event.get('file_count'))
        if file_count <= 0:
            files = event.get('files')
            if isinstance(files, list):
                file_count = len(files)
        if file_count <= 0:
            return ''
        return (
            f'{file_count} file(s): '
            f'{_coerce_positive_int(event.get("added_files"))} added, '
            f'{_coerce_positive_int(event.get("modified_files"))} modified, '
            f'{_coerce_positive_int(event.get("deleted_files"))} deleted; '
            f'+{_coerce_positive_int(event.get("added_lines"))} '
            f'-{_coerce_positive_int(event.get("removed_lines"))}'
        )

    def _render_workspace_change_detail(
        self,
        event: dict[str, object],
        summary: str,
    ) -> str:
        lines = [summary]
        tool_name = _preview_value(event.get('tool_name'))
        if tool_name:
            lines.append(f'tool={tool_name}')
        files = event.get('files')
        if isinstance(files, list):
            for file_payload in files:
                if not isinstance(file_payload, dict):
                    continue
                rendered = self._render_workspace_change_file(file_payload)
                if rendered:
                    lines.extend(['', rendered])
        truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
        if truncated_file_count:
            lines.extend(['', f'... {truncated_file_count} more changed file(s)'])
        return '\n'.join(lines)

    def _render_workspace_change_file(self, file_payload: dict[str, object]) -> str:
        path = _preview_value(file_payload.get('path'), max_chars=220)
        if not path:
            return ''
        status = _preview_value(file_payload.get('status')) or 'changed'
        added = _coerce_positive_int(file_payload.get('added_lines'))
        removed = _coerce_positive_int(file_payload.get('removed_lines'))
        header = f'- {status} {path} (+{added} -{removed})'
        if file_payload.get('binary'):
            return header + '\n  binary or non-text file'
        if file_payload.get('content_truncated'):
            return header + '\n  large file; diff omitted'
        diff = file_payload.get('diff')
        if not isinstance(diff, str) or not diff.strip():
            return header
        if file_payload.get('diff_truncated'):
            header += ' [diff truncated]'
        return f'{header}\n\n```diff\n{diff.rstrip()}\n```'

    def _render_workspace_change_recap_summary(
        self,
        event: dict[str, object],
    ) -> str:
        file_count = _coerce_positive_int(event.get('file_count'))
        if file_count <= 0:
            files = event.get('files')
            if isinstance(files, list):
                file_count = len(files)
        if file_count <= 0:
            return ''
        label = 'file' if file_count == 1 else 'files'
        return (
            f'{file_count} {label} changed '
            f'+{_coerce_positive_int(event.get("added_lines"))} '
            f'-{_coerce_positive_int(event.get("removed_lines"))}'
        )

    def _render_workspace_change_recap_detail(
        self,
        event: dict[str, object],
        summary: str,
    ) -> str:
        lines = [summary, '']
        files = event.get('files')
        if isinstance(files, list):
            for file_payload in files:
                if not isinstance(file_payload, dict):
                    continue
                line = self._render_workspace_change_recap_file(file_payload)
                if line:
                    lines.append(line)
        truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
        if truncated_file_count:
            lines.append(f'... {truncated_file_count} more changed file(s)')
        return '\n'.join(lines).rstrip()

    def _render_workspace_change_recap_file(
        self,
        file_payload: dict[str, object],
    ) -> str:
        path = _preview_value(file_payload.get('path'), max_chars=220)
        if not path:
            return ''
        added = _coerce_positive_int(file_payload.get('added_lines'))
        removed = _coerce_positive_int(file_payload.get('removed_lines'))
        status = _preview_value(file_payload.get('status')) or 'modified'
        suffix = '' if status == 'modified' else f' {status}'
        return f'- {path} +{added} -{removed}{suffix}'

    def _render_tool_result(self, event: dict[str, object]) -> str:
        tool_name = event.get('tool_name')
        ok = bool(event.get('ok'))
        metadata = event.get('metadata')
        content_preview = _preview_value(event.get('content_preview'), max_chars=220)
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = 'tool'
        if not isinstance(metadata, dict):
            metadata = {}
        action = metadata.get('action')
        path = metadata.get('path')
        if action == 'bash':
            exit_code = metadata.get('exit_code')
            output_preview = _preview_value(metadata.get('output_preview'), max_chars=120)
            if output_preview:
                return f'[command] exit_code={exit_code} ok={ok} output={output_preview}'
            return f'[command] exit_code={exit_code} ok={ok}'
        if action == 'web_search':
            query = _preview_value(metadata.get('query'), max_chars=90)
            result_count = metadata.get('result_count')
            top_url = ''
            top_urls = metadata.get('top_urls')
            if isinstance(top_urls, list) and top_urls:
                top_url = _preview_value(top_urls[0], max_chars=120)
            parts = [f'[search] ok={ok}']
            if query:
                parts.append(f'query={query}')
            if isinstance(result_count, int):
                parts.append(f'results={result_count}')
            if top_url:
                parts.append(f'top={top_url}')
            return ' '.join(parts)
        if action == 'web_fetch':
            url = _preview_value(metadata.get('url'), max_chars=120)
            fetched_chars = metadata.get('fetched_chars')
            preview = _preview_value(metadata.get('preview'), max_chars=120)
            parts = [f'[web_fetch] ok={ok}']
            if url:
                parts.append(f'url={url}')
            if isinstance(fetched_chars, int):
                parts.append(f'chars={fetched_chars}')
            if metadata.get('truncated') is True:
                parts.append('truncated=True')
            if preview:
                parts.append(f'preview={preview}')
            return ' '.join(parts)
        if isinstance(path, str) and path:
            file_action = action if isinstance(action, str) and action else tool_name
            if file_action in {'write_file', 'edit_file', 'notebook_edit'}:
                label = 'updated'
            elif file_action == 'read_file':
                label = 'read'
            else:
                label = file_action
            return f'[file] {label} {path} ok={ok}'
        if action == 'mcp_call_tool' or tool_name.startswith('mcp_'):
            server = _preview_value(metadata.get('server_name') or metadata.get('requested_server')) or '(auto)'
            remote_tool = _preview_value(metadata.get('tool_name')) or tool_name
            summary = f'[mcp] server={server} tool={remote_tool} ok={ok}'
            if not ok and content_preview:
                summary += f' error={content_preview}'
            return summary
        cwd_update = metadata.get('cwd_update')
        if isinstance(cwd_update, str) and cwd_update:
            return f'[cwd] {cwd_update}'
        for candidate in (
            metadata.get('output_preview'),
            metadata.get('preview'),
            metadata.get('arguments_preview'),
            metadata.get('answer_preview'),
            content_preview,
        ):
            preview = _preview_value(candidate, max_chars=180)
            if preview:
                return f'[tool] {tool_name} ok={ok} {preview}'
        if isinstance(action, str) and action and action != tool_name:
            return f'[tool] {tool_name} action={action} ok={ok}'
        return f'[tool] {tool_name} ok={ok}'

    def _render_tool_result_detail(self, event: dict[str, object], summary: str) -> str:
        lines = [summary]
        tool_call_id = _preview_value(event.get('tool_call_id'))
        if tool_call_id:
            lines.append(f'tool_call_id={tool_call_id}')
        metadata = event.get('metadata')
        action = metadata.get('action') if isinstance(metadata, dict) else None
        if action == 'web_fetch':
            preview = _preview_value(event.get('content_preview'), max_chars=360)
            if preview:
                lines.append(f'preview={preview}')
            if isinstance(metadata, dict):
                for key in ('url', 'fetched_chars', 'truncated'):
                    value = metadata.get(key)
                    if value is not None:
                        lines.append(f'{key}={_preview_value(value, max_chars=180)}')
            return '\n'.join(lines)
        if action in {'delegate_agent', 'Agent'}:
            preview = _preview_value(event.get('content_preview'), max_chars=520)
            if preview:
                lines.append(f'summary={preview}')
            if isinstance(metadata, dict):
                for key in (
                    'subagent_type',
                    'subtask_count',
                    'completed_children',
                    'failed_children',
                    'group_status',
                    'child_stop_reason',
                ):
                    value = metadata.get(key)
                    if value is not None:
                        lines.append(f'{key}={_preview_value(value, max_chars=160)}')
            return '\n'.join(lines)
        content = event.get('content')
        if isinstance(content, str) and content:
            lines.append('content:')
            lines.append(self._clip_multiline_content(content.rstrip()))
        elif isinstance(event.get('content_preview'), str) and event.get('content_preview'):
            lines.append('content_preview:')
            lines.append(str(event.get('content_preview')).rstrip())
        if isinstance(metadata, dict) and metadata:
            lines.append('metadata:')
            lines.append(self._format_json_for_display(metadata, max_chars=1800))
        return '\n'.join(lines)

    def _is_delegate_tool(self, tool_name: object) -> bool:
        return tool_name in {'Agent', 'delegate_agent'}

    def _delegate_request_label(self, event: dict[str, object]) -> str:
        arguments = event.get('arguments')
        subagent_type = ''
        if isinstance(arguments, dict):
            subagent_type = _preview_value(arguments.get('subagent_type'), max_chars=40)
        return subagent_type or 'general'

    def _render_delegate_tool_request(self, arguments: dict[str, object]) -> str:
        subagent_type = (
            _preview_value(arguments.get('subagent_type'), max_chars=40)
            or 'general'
        )
        label = _preview_value(arguments.get('label'), max_chars=80)
        subtasks = arguments.get('subtasks')
        parts = [f'[delegate] subagent={subagent_type}']
        if label:
            parts.append(f'label={label}')
        if isinstance(subtasks, list):
            parts.append(f'subtasks={len(subtasks)}')
        return ' '.join(parts)

    def _render_delegate_arguments(self, arguments: dict[str, object]) -> list[str]:
        lines: list[str] = []
        for key in (
            'subagent_type',
            'label',
            'strategy',
            'max_turns',
            'max_parallel_subtasks',
        ):
            value = arguments.get(key)
            if value is not None:
                lines.append(f'{key}={_preview_value(value, max_chars=160)}')
        subtasks = arguments.get('subtasks')
        if isinstance(subtasks, list):
            lines.append(f'subtasks={len(subtasks)}')
            for index, item in enumerate(subtasks[:5], start=1):
                if isinstance(item, dict):
                    label = (
                        _preview_value(item.get('label'), max_chars=80)
                        or f'subtask_{index}'
                    )
                    lines.append(f'- {label}')
                else:
                    lines.append(f'- subtask_{index}')
            if len(subtasks) > 5:
                lines.append(f'- ... plus {len(subtasks) - 5} more')
        return lines

    def _render_file_write_arguments(self, arguments: dict[str, object]) -> list[str]:
        lines: list[str] = []
        path = _preview_value(arguments.get('path'), max_chars=220)
        if path:
            lines.append(f'path={path}')
        content = arguments.get('content')
        if isinstance(content, str):
            lines.append(f'content_chars={len(content)}')
            preview = _preview_value(content, max_chars=360)
            if preview:
                lines.append(f'content_preview={preview}')
        for key in ('old_text', 'new_text'):
            value = arguments.get(key)
            if isinstance(value, str):
                lines.append(f'{key}_chars={len(value)}')
                preview = _preview_value(value, max_chars=260)
                if preview:
                    lines.append(f'{key}_preview={preview}')
        return lines

    def _render_tool_plan_summary(self, event: dict[str, object]) -> str:
        tool_name = _preview_value(event.get('tool_name')) or 'tool'
        if tool_name in {'Agent', 'delegate_agent'}:
            return 'Planning sub-agent delegation'
        if tool_name in {'write_file', 'edit_file', 'notebook_edit'}:
            return f'Planning file update: {tool_name}'
        arguments_delta = _preview_value(event.get('arguments_delta'), max_chars=160)
        if arguments_delta:
            return f'Planning tool call: {tool_name} args~ {arguments_delta}'
        return f'Planning tool call: {tool_name}'

    def _clip_multiline_content(self, content: str, *, max_chars: int = 1800) -> str:
        if len(content) <= max_chars:
            return content
        return content[:max_chars].rstrip() + '\n...[truncated for display]...'

    def _format_json_for_display(self, payload: object, *, max_chars: int = 6000) -> str:
        try:
            rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            rendered = str(payload)
        if len(rendered) <= max_chars:
            return rendered
        head = rendered[: max_chars - 20]
        return head + '\n...[truncated]...'

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

    def _append_or_merge_turn_entry(
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
        if merge_key:
            for entry in reversed(turn.entries):
                if entry.merge_key == merge_key:
                    entry.kind = kind
                    entry.title = title
                    entry.content += content
                    entry.status = status
                    return
        self._append_turn_entry(
            kind=kind,
            title=title,
            content=content,
            status=status,
            merge_key=merge_key,
        )

    def _upsert_turn_entry(
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
        if merge_key:
            for entry in reversed(turn.entries):
                if entry.merge_key == merge_key:
                    entry.kind = kind
                    entry.title = title
                    entry.content = content
                    entry.status = status
                    return
        self._append_turn_entry(
            kind=kind,
            title=title,
            content=content,
            status=status,
            merge_key=merge_key,
        )

    def _find_turn_entry_by_merge_key(
        self,
        merge_key: str,
    ) -> ConversationEntry | None:
        turn = self._active_turn()
        if turn is None:
            return None
        for entry in reversed(turn.entries):
            if entry.merge_key == merge_key:
                return entry
        return None

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

    def _publish_changes(self) -> None:
        if self._on_changes_change is not None:
            self._on_changes_change(
                dict(self._workspace_change_summary)
                if self._workspace_change_summary is not None
                else None
            )

    def _publish_state(self) -> None:
        if self._on_state_change is not None:
            self._on_state_change(self.state)
