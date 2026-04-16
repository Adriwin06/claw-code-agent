from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .agent_slash_commands import find_slash_command, get_slash_command_specs
from .agent_runtime import LocalCodingAgent
from .agent_types import AgentRunResult
from .session_store import StoredAgentSession, load_agent_session, usage_from_payload


MAX_ACTIVITY_ITEMS = 14


def _render_permissions(agent: LocalCodingAgent) -> str:
    permissions = agent.runtime_config.permissions
    labels: list[str] = []
    if permissions.allow_file_write:
        labels.append('write')
    if permissions.allow_shell_commands:
        labels.append('shell')
    if permissions.allow_destructive_shell_commands:
        labels.append('unsafe')
    return ', '.join(labels) if labels else 'read-only'


@dataclass(frozen=True)
class SlashCommandSuggestion:
    primary_name: str
    aliases: tuple[str, ...]
    description: str

    @property
    def insertion_text(self) -> str:
        return f'/{self.primary_name} '

    @property
    def label(self) -> str:
        return f'/{self.primary_name}'


def build_slash_command_suggestions() -> tuple[SlashCommandSuggestion, ...]:
    return tuple(
        SlashCommandSuggestion(
            primary_name=spec.names[0],
            aliases=spec.names,
            description=spec.description,
        )
        for spec in get_slash_command_specs()
    )


def extract_slash_command_query(text: str) -> str | None:
    trimmed = text.lstrip()
    if not trimmed.startswith('/'):
        return None
    without_slash = trimmed[1:]
    if any(character.isspace() for character in without_slash):
        return None
    return without_slash.lower()


def filter_slash_command_suggestions(
    text: str,
    suggestions: tuple[SlashCommandSuggestion, ...] | None = None,
) -> list[SlashCommandSuggestion]:
    query = extract_slash_command_query(text)
    if query is None:
        return []
    if suggestions is None:
        suggestions = build_slash_command_suggestions()
    if not query:
        return list(suggestions)

    scored: list[tuple[int, int, SlashCommandSuggestion]] = []
    for index, suggestion in enumerate(suggestions):
        aliases = tuple(alias.lower() for alias in suggestion.aliases)
        if any(alias == query for alias in aliases):
            score = 0
        elif any(alias.startswith(query) for alias in aliases):
            score = 1
        elif any(query in alias for alias in aliases):
            score = 2
        else:
            continue
        scored.append((score, index, suggestion))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [suggestion for _, _, suggestion in scored]


def render_slash_command_suggestion_detail(suggestion: SlashCommandSuggestion) -> str:
    lines = [
        suggestion.label,
        '',
        suggestion.description,
    ]
    if len(suggestion.aliases) > 1:
        aliases = ', '.join(f'/{alias}' for alias in suggestion.aliases[1:])
        lines.extend(
            [
                '',
                f'aliases={aliases}',
            ]
        )
    lines.extend(
        [
            '',
            'Controls',
            '',
            'Up/Down: move selection',
            'Tab: insert command',
            'Enter: insert when incomplete, run when exact',
            'Mouse: click a command to insert it',
        ]
    )
    return '\n'.join(lines)


def should_route_key_to_prompt(
    *,
    character: str | None,
    prompt_focused: bool,
    busy: bool,
) -> bool:
    if busy or prompt_focused:
        return False
    if not isinstance(character, str) or len(character) != 1:
        return False
    return character.isprintable()


def build_working_section_id(conversation_id: str, turn_id: str) -> str:
    return f'working-{conversation_id}-{turn_id}'


def build_working_section_instance_id(
    conversation_id: str,
    turn_id: str,
    *,
    section_index: int,
    section_count: int,
) -> str:
    base_id = build_working_section_id(conversation_id, turn_id)
    if section_count <= 1:
        return base_id
    return f'{base_id}-section-{section_index}'


@dataclass
class ConversationTurn:
    turn_id: str
    user_prompt: str
    assistant_response: str = ''
    assistant_status: str = 'Queued'
    phase_label: str = 'Queued'
    tool_count: int = 0
    restored: bool = False
    stop_reason: str | None = None
    session_id: str | None = None
    entries: list['ConversationEntry'] = field(default_factory=list)

    def prompt_preview(self, max_chars: int = 44) -> str:
        return _preview_value(self.user_prompt, max_chars=max_chars) or '(empty prompt)'

    def assistant_preview(self, max_chars: int = 60) -> str:
        if not self.assistant_response.strip():
            return self.assistant_status
        return _preview_value(self.assistant_response, max_chars=max_chars)


@dataclass(frozen=True)
class ConversationHistoryItem:
    turn_id: str
    label: str
    prompt_preview: str
    assistant_preview: str
    status: str


@dataclass
class ConversationThread:
    conversation_id: str
    title: str
    turns: tuple[ConversationTurn, ...] = ()
    session_id: str | None = None
    expanded: bool = True


@dataclass(frozen=True)
class SidebarItem:
    option_id: str
    label: str
    kind: str
    conversation_id: str
    turn_id: str | None = None


@dataclass
class ActivityItem:
    key: str
    label: str
    detail: str = ''
    status: str = 'info'


@dataclass
class ConversationEntry:
    entry_id: str
    kind: str
    title: str
    content: str = ''
    status: str = 'info'
    merge_key: str | None = None


def _should_hide_session_message(payload: dict[str, object]) -> bool:
    role = str(payload.get('role', ''))
    if role == 'system':
        return True
    content = '' if payload.get('content') is None else str(payload.get('content', ''))
    message_id = payload.get('message_id')
    metadata = payload.get('metadata')
    if isinstance(message_id, str) and (
        message_id.startswith('user_context_')
        or message_id.startswith('plugin_tool_runtime_')
    ):
        return True
    if isinstance(metadata, dict):
        kind = metadata.get('kind')
        if isinstance(kind, str) and kind.endswith('_runtime'):
            return True
    return content.lstrip().startswith('<system-reminder>')


def restore_conversation_turns(
    messages: Sequence[dict[str, object]],
) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    current_turn: ConversationTurn | None = None
    restored_index = 0

    for payload in messages:
        if not isinstance(payload, dict) or _should_hide_session_message(payload):
            continue
        role = str(payload.get('role', ''))
        content = '' if payload.get('content') is None else str(payload.get('content', ''))
        stop_reason = (
            str(payload['stop_reason'])
            if isinstance(payload.get('stop_reason'), str)
            else None
        )
        if role == 'user':
            restored_index += 1
            current_turn = ConversationTurn(
                turn_id=f'restored-{restored_index}',
                user_prompt=content,
                assistant_status='Restored',
                phase_label='Restored',
                restored=True,
                stop_reason=stop_reason,
            )
            turns.append(current_turn)
            continue
        if role == 'assistant':
            if current_turn is None:
                restored_index += 1
                current_turn = ConversationTurn(
                    turn_id=f'restored-{restored_index}',
                    user_prompt='(conversation resumed)',
                    assistant_status='Restored',
                    phase_label='Restored',
                    restored=True,
                )
                turns.append(current_turn)
            if content:
                if current_turn.assistant_response:
                    current_turn.assistant_response += '\n\n' + content
                else:
                    current_turn.assistant_response = content
                current_turn.entries.append(
                    ConversationEntry(
                        entry_id=f'{current_turn.turn_id}-assistant-{len(current_turn.entries) + 1}',
                        kind='assistant',
                        title='Assistant',
                        content=content,
                        status='ok',
                    )
                )
            current_turn.assistant_status = 'Ready'
            current_turn.phase_label = 'Restored'
            current_turn.stop_reason = stop_reason or current_turn.stop_reason
            continue
        if role == 'tool' and current_turn is not None:
            current_turn.tool_count += 1
            current_turn.entries.append(
                ConversationEntry(
                    entry_id=f'{current_turn.turn_id}-tool-{len(current_turn.entries) + 1}',
                    kind='tool',
                    title='Tool Result',
                    content=content,
                    status='info',
                )
            )

    return tuple(turns)


def build_conversation_history_items(
    turns: Sequence[ConversationTurn],
) -> tuple[ConversationHistoryItem, ...]:
    items: list[ConversationHistoryItem] = []
    for index, turn in enumerate(turns, start=1):
        prompt_preview = turn.prompt_preview(max_chars=34)
        items.append(
            ConversationHistoryItem(
                turn_id=turn.turn_id,
                label=f'{index:02d}. {prompt_preview}',
                prompt_preview=prompt_preview,
                assistant_preview=turn.assistant_preview(max_chars=56),
                status=turn.assistant_status,
            )
        )
    return tuple(items)


def hydrate_state_from_stored_session(
    state: 'AgentTuiState',
    stored_session: StoredAgentSession,
) -> None:
    usage = usage_from_payload(stored_session.usage)
    state.status = 'Resumed'
    state.phase = 'Resumed'
    state.phase_detail = f'Loaded session {stored_session.session_id}'
    state.session_id = stored_session.session_id
    state.last_turns = stored_session.turns
    state.last_tool_calls = stored_session.tool_calls
    state.total_tokens = usage.total_tokens
    state.input_tokens = usage.input_tokens
    state.output_tokens = usage.output_tokens
    state.total_cost_usd = stored_session.total_cost_usd


@dataclass
class AgentTuiState:
    workspace: str
    model: str
    permissions: str
    status: str = 'Idle'
    phase: str = 'Idle'
    phase_detail: str = 'Waiting for input'
    session_id: str | None = None
    prompt_count: int = 0
    last_turns: int = 0
    last_tool_calls: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    last_tool: str | None = None
    last_stop_reason: str | None = None
    streaming_enabled: bool = False
    max_turns: int = 0
    command_timeout_seconds: float = 0.0
    conversation_turns: int = 0
    activity_events: int = 0
    busy: bool = False

    @classmethod
    def from_agent(cls, agent: LocalCodingAgent) -> 'AgentTuiState':
        return cls(
            workspace=str(agent.runtime_config.cwd),
            model=agent.model_config.model,
            permissions=_render_permissions(agent),
            streaming_enabled=agent.runtime_config.stream_model_responses,
            max_turns=agent.runtime_config.max_turns,
            command_timeout_seconds=agent.runtime_config.command_timeout_seconds,
        )

    def render(self) -> str:
        lines = [
            'Session',
            '',
            f'status={self.status}',
            f'phase={self.phase}',
            f'phase_detail={self.phase_detail}',
            f'workspace={self.workspace}',
            f'model={self.model}',
            f'permissions={self.permissions}',
            f'streaming={self.streaming_enabled}',
            f'max_turns={self.max_turns}',
            f'command_timeout_seconds={self.command_timeout_seconds:.1f}',
            f'prompt_count={self.prompt_count}',
            f'conversation_turns={self.conversation_turns}',
            f'activity_events={self.activity_events}',
            f'last_turns={self.last_turns}',
            f'last_tool_calls={self.last_tool_calls}',
            f'total_tokens={self.total_tokens}',
            f'input_tokens={self.input_tokens}',
            f'output_tokens={self.output_tokens}',
            f'total_cost_usd={self.total_cost_usd:.6f}',
        ]
        if self.session_id:
            lines.append(f'session_id={self.session_id}')
        if self.last_tool:
            lines.append(f'last_tool={self.last_tool}')
        if self.last_stop_reason:
            lines.append(f'last_stop_reason={self.last_stop_reason}')
        return '\n'.join(lines)


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
                'Ready' if result.stop_reason in {None, 'stop'} else 'Stopped'
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


def _preview_value(value: object, *, max_chars: int = 120) -> str:
    if value is None:
        return ''
    text = str(value)
    text = ' '.join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3] + '...'
    return text


def _preview_multiline(text: str, *, max_chars: int = 220, max_lines: int = 4) -> str:
    if not text:
        return ''
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ''
    clipped = lines[-max_lines:]
    rendered = '\n'.join(clipped)
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:]
        if not rendered.startswith('...'):
            rendered = '...' + rendered[3:]
    return rendered


def _friendly_stop_reason(stop_reason: str | None) -> str:
    if stop_reason in {None, '', 'stop'}:
        return 'completed'
    return stop_reason


def run_agent_tui(
    agent: LocalCodingAgent,
    *,
    initial_prompt: str | None = None,
    resume_session_id: str | None = None,
) -> int:
    try:
        from textual import events, work
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        try:
            from textual.containers import VerticalScroll
        except ImportError:
            from textual.containers import ScrollableContainer as VerticalScroll
        from textual.widgets import (
            Button,
            Collapsible,
            Footer,
            Header,
            Input,
            Markdown as TextualMarkdown,
            OptionList,
            Static,
        )
        from textual.widgets.option_list import Option
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(
            'Textual is not installed. Install it with `pip install -e .[tui]` '
            'or `pip install textual` before using `agent-tui`.'
        ) from exc

    class TurnCard(Vertical):
        def __init__(
            self,
            turn: ConversationTurn,
            *,
            conversation_id: str,
            turn_index: int,
            selected: bool,
            active: bool,
            busy: bool,
            phase: str,
            phase_detail: str,
            spinner_index: int,
            collapsed_sections: dict[str, bool],
        ) -> None:
            classes = 'turn-card selected' if selected else 'turn-card'
            super().__init__(classes=classes)
            self._turn = turn
            self._conversation_id = conversation_id
            self._turn_index = turn_index
            self._selected = selected
            self._active = active
            self._busy = busy
            self._phase = phase
            self._phase_detail = phase_detail
            self._spinner_index = spinner_index
            self._collapsed_sections = dict(collapsed_sections)

        def compose(self) -> ComposeResult:
            working_section_count = self._working_section_count()
            working_section_index = 0
            yield self._static_widget(
                f'Turn {self._turn_index}',
                classes='turn-title',
            )
            yield self._static_widget(
                self._turn.user_prompt or '(empty prompt)',
                classes='turn-user',
            )
            work_entries: list[ConversationEntry] = []
            rendered_assistant = False
            for entry in self._turn.entries:
                if entry.kind == 'assistant':
                    if work_entries:
                        working_section_index += 1
                        yield from self._compose_working_section(
                            work_entries,
                            section_index=working_section_index,
                            section_count=working_section_count,
                            include_live_state=False,
                        )
                        work_entries = []
                    rendered_assistant = True
                    yield self._markdown_widget(
                        entry.content or '(empty)',
                        classes='turn-assistant',
                    )
                    continue
                work_entries.append(entry)
            if work_entries or (self._active and self._busy):
                working_section_index += 1
                yield from self._compose_working_section(
                    work_entries,
                    section_index=working_section_index,
                    section_count=working_section_count,
                    include_live_state=self._active and self._busy,
                )
            if not rendered_assistant and not self._busy:
                yield self._static_widget(
                    f'[{self._turn.assistant_status}] no assistant output',
                    classes='turn-empty',
                )
            yield self._static_widget(
                self._footer_text(),
                classes='turn-footer',
            )

        def _compose_working_section(
            self,
            entries: Sequence[ConversationEntry],
            *,
            section_index: int,
            section_count: int,
            include_live_state: bool,
        ):
            if not entries and not include_live_state:
                return
            with self._collapsible_widget(
                widget_id=self._working_section_id(
                    section_index=section_index,
                    section_count=section_count,
                ),
                title=self._working_title(entries, include_live_state=include_live_state),
                collapsed=self._working_collapsed(
                    section_index=section_index,
                    section_count=section_count,
                ),
                classes='turn-working',
            ):
                yield self._markdown_widget(
                    self._working_markdown(entries, include_live_state=include_live_state),
                    classes='turn-working-body',
                )

        def _working_section_count(self) -> int:
            count = 0
            work_entries: list[ConversationEntry] = []
            for entry in self._turn.entries:
                if entry.kind == 'assistant':
                    if work_entries:
                        count += 1
                        work_entries = []
                    continue
                work_entries.append(entry)
            if work_entries or (self._active and self._busy):
                count += 1
            return count

        def _working_title(
            self,
            entries: Sequence[ConversationEntry],
            *,
            include_live_state: bool,
        ) -> str:
            if include_live_state:
                spinner_frames = ('|', '/', '-', '\\')
                spinner = spinner_frames[self._spinner_index % len(spinner_frames)]
                base = f'{spinner} Working'
            else:
                base = 'Working'
            tool_events = sum(1 for entry in entries if entry.kind in {'tool', 'tool_result'})
            if tool_events:
                base = f'{base} ({tool_events})'
            return base

        def _working_markdown(
            self,
            entries: Sequence[ConversationEntry],
            *,
            include_live_state: bool,
        ) -> str:
            lines: list[str] = []
            if include_live_state:
                lines.extend(
                    [
                        f'**State:** {self._phase}',
                        '',
                        self._phase_detail or self._phase,
                        '',
                    ]
                )
            for entry in entries:
                heading = self._working_entry_heading(entry)
                lines.append(f'**{heading}**')
                body = entry.content.strip() or '(empty)'
                if entry.kind in {'tool', 'tool_output', 'tool_result'}:
                    lines.extend(['```text', body.rstrip(), '```', ''])
                else:
                    lines.extend([body, ''])
            rendered = '\n'.join(lines).strip()
            return rendered or '_No internal work details._'

        def _working_entry_heading(self, entry: ConversationEntry) -> str:
            if entry.kind == 'thinking':
                return 'Thinking'
            if entry.kind == 'tool':
                return 'Tool Call'
            if entry.kind == 'tool_output':
                return 'Tool Output'
            if entry.kind == 'tool_result':
                return 'Tool Result'
            if entry.kind == 'error':
                return 'Error'
            if entry.kind == 'warning':
                return 'Warning'
            if entry.kind == 'status':
                return 'Status'
            return entry.title

        def _working_section_id(self, *, section_index: int, section_count: int) -> str:
            return build_working_section_instance_id(
                self._conversation_id,
                self._turn.turn_id,
                section_index=section_index,
                section_count=section_count,
            )

        def _working_collapsed(self, *, section_index: int, section_count: int) -> bool:
            section_id = self._working_section_id(
                section_index=section_index,
                section_count=section_count,
            )
            collapsed = self._collapsed_sections.get(section_id)
            if isinstance(collapsed, bool):
                return collapsed
            base_id = build_working_section_id(self._conversation_id, self._turn.turn_id)
            base_collapsed = self._collapsed_sections.get(base_id)
            if isinstance(base_collapsed, bool):
                return base_collapsed
            return not (self._active and self._busy)

        def _static_widget(self, content: str, *, classes: str | None = None) -> Static:
            widget = Static(content)
            if classes:
                for class_name in classes.split():
                    widget.add_class(class_name)
            return widget

        def _markdown_widget(self, content: str, *, classes: str | None = None) -> TextualMarkdown:
            widget = TextualMarkdown(content)
            if classes:
                for class_name in classes.split():
                    widget.add_class(class_name)
            return widget

        def _collapsible_widget(
            self,
            *,
            widget_id: str | None = None,
            title: str,
            collapsed: bool,
            classes: str | None = None,
        ) -> Collapsible:
            widget = Collapsible(title=title, collapsed=collapsed)
            if widget_id is not None:
                widget.id = widget_id
            if classes:
                for class_name in classes.split():
                    widget.add_class(class_name)
            return widget

        def _footer_text(self) -> str:
            status_label = 'in progress' if self._active and self._busy else 'done'
            parts = [status_label]
            if self._turn.tool_count:
                parts.append(f'tools={self._turn.tool_count}')
            friendly_reason = _friendly_stop_reason(self._turn.stop_reason)
            if friendly_reason != 'completed':
                parts.append(f'reason={friendly_reason}')
            if self._turn.restored:
                parts.append('restored')
            return ' | '.join(parts)

    class ConversationFeed(Vertical):
        def __init__(self) -> None:
            super().__init__(id='conversation-feed')
            self._conversation_id: str = 'conversation-0'
            self._turns: tuple[ConversationTurn, ...] = ()
            self._selected_turn_id: str | None = None
            self._state_phase: str = 'Idle'
            self._state_phase_detail: str = ''
            self._state_busy: bool = False
            self._spinner_index: int = 0
            self._collapsed_sections: dict[str, bool] = {}

        def set_data(
            self,
            *,
            conversation_id: str,
            turns: tuple[ConversationTurn, ...],
            selected_turn_id: str | None,
            phase: str,
            phase_detail: str,
            busy: bool,
            spinner_index: int,
            collapsed_sections: dict[str, bool],
        ) -> None:
            self._conversation_id = conversation_id
            self._turns = turns
            self._selected_turn_id = selected_turn_id
            self._state_phase = phase
            self._state_phase_detail = phase_detail
            self._state_busy = busy
            self._spinner_index = spinner_index
            self._collapsed_sections = dict(collapsed_sections)
            self.refresh(recompose=True, layout=True)

        def compose(self) -> ComposeResult:
            if not self._turns:
                empty = Static('No conversation yet. Submit a prompt to start.')
                empty.add_class('conversation-empty')
                yield empty
                return
            last_turn_id = self._turns[-1].turn_id
            for index, turn in enumerate(self._turns, start=1):
                yield TurnCard(
                    turn,
                    conversation_id=self._conversation_id,
                    turn_index=index,
                    selected=(turn.turn_id == self._selected_turn_id),
                    active=(turn.turn_id == last_turn_id),
                    busy=self._state_busy and turn.turn_id == last_turn_id,
                    phase=self._state_phase,
                    phase_detail=self._state_phase_detail,
                    spinner_index=self._spinner_index,
                    collapsed_sections=self._collapsed_sections,
                )

    class AgentTuiApp(App[None]):
        TITLE = 'Claw Code Agent'
        CSS = """
        Screen {
            layout: vertical;
            background: #0b1117;
            color: #e6edf3;
        }

        Header {
            background: #16202b;
        }

        Footer {
            background: #16202b;
        }

        #body {
            height: 1fr;
            margin: 1;
        }

        #left-rail {
            width: 30;
            min-width: 24;
        }

        #history-toolbar {
            height: auto;
            margin-bottom: 1;
        }

        #history-title {
            width: 1fr;
            color: #8b949e;
            text-style: bold;
            padding: 0 1;
            content-align: left middle;
        }

        #new-conversation-button {
            width: 9;
            min-width: 8;
        }

        #center-column {
            width: 1fr;
            min-width: 60;
            margin-left: 1;
        }

        #right-rail {
            width: 34;
            min-width: 28;
        }

        #history-list {
            height: 1fr;
            border: round #3b4b5c;
            background: #111923;
        }

        #conversation-scroll {
            height: 1fr;
            border: round #2d3742;
            background: #081019;
            overflow-y: auto;
        }

        #conversation-feed {
            width: 1fr;
            height: auto;
            min-height: 100%;
            padding: 1 1 2 1;
        }

        .conversation-empty {
            color: #8b949e;
            padding: 1 1 2 1;
        }

        .turn-card {
            width: 1fr;
            height: auto;
            border: round #3b4b5c;
            margin-bottom: 1;
            padding: 0 1 1 1;
        }

        .turn-card.selected {
            border: round #d29922;
        }

        .turn-title {
            color: #d29922;
            text-style: bold;
            margin: 0 0 1 0;
        }

        .turn-user {
            width: 1fr;
            height: auto;
            border: round #2f81f7;
            padding: 0 1;
            margin-bottom: 1;
        }

        .turn-assistant {
            width: 1fr;
            height: auto;
            border: round #3fb950;
            padding: 0 1;
            margin-bottom: 1;
        }

        .turn-empty {
            width: 1fr;
            height: auto;
            color: #8b949e;
            padding: 0 1;
            margin-bottom: 1;
        }

        .turn-footer {
            width: 1fr;
            height: auto;
            color: #8b949e;
            margin-top: 1;
        }

        .turn-working {
            width: 1fr;
            height: auto;
            margin-bottom: 1;
        }

        .turn-working-body {
            width: 1fr;
            height: auto;
            padding: 0 1;
        }

        #details {
            height: 1fr;
            border: round #2f81f7;
            background: #111923;
            padding: 1 2;
        }

        #command-picker {
            height: 12;
            margin: 0 1;
        }

        #command-options {
            width: 34;
            min-width: 28;
            border: round #2f81f7;
        }

        #command-description {
            width: 1fr;
            border: round #3b4b5c;
            padding: 1 2;
            background: #111923;
        }

        #prompt {
            margin: 0 1 1 1;
        }
        """
        BINDINGS = [
            ('ctrl+q', 'quit', 'Quit'),
            ('ctrl+n', 'new_conversation', 'New'),
            ('ctrl+pageup', 'previous_conversation', 'Prev'),
            ('ctrl+pagedown', 'next_conversation', 'Next'),
            ('ctrl+j', 'focus_prompt', 'Prompt'),
            ('ctrl+r', 'refresh_panels', 'Refresh'),
            ('ctrl+f', 'toggle_auto_follow', 'Follow'),
            ('ctrl+h', 'focus_history', 'History'),
        ]

        def __init__(
            self,
            runtime_agent: LocalCodingAgent,
            *,
            first_prompt: str | None,
            resumed_session_id: str | None,
        ) -> None:
            super().__init__()
            self._agent = runtime_agent
            self._first_prompt = first_prompt.strip() if first_prompt else None
            self._active_session_id = resumed_session_id
            self._state = AgentTuiState.from_agent(runtime_agent)
            self._command_suggestions = build_slash_command_suggestions()
            self._visible_command_suggestions: list[SlashCommandSuggestion] = []
            self._conversation_counter = 1
            self._conversations: list[ConversationThread] = [
                ConversationThread(
                    conversation_id='conversation-1',
                    title=('Resumed conversation' if resumed_session_id else 'Conversation 1'),
                )
            ]
            self._active_conversation_id = 'conversation-1'
            self._sidebar_items: tuple[SidebarItem, ...] = ()
            self._conversation_turns: tuple[ConversationTurn, ...] = ()
            self._history_items: tuple[ConversationHistoryItem, ...] = ()
            self._activity_items: tuple[ActivityItem, ...] = ()
            self._selected_turn_id: str | None = None
            self._auto_follow = True
            self._spinner_index = 0
            self._collapsed_sections: dict[str, bool] = {}
            self._restored_session: StoredAgentSession | None = None
            if resumed_session_id:
                try:
                    self._restored_session = load_agent_session(
                        resumed_session_id,
                        directory=self._agent.runtime_config.session_directory,
                    )
                except FileNotFoundError:
                    self._restored_session = None
                else:
                    hydrate_state_from_stored_session(self._state, self._restored_session)
                    self._conversations[0].session_id = resumed_session_id
            self._bridge = self._make_bridge(self._state)

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id='body'):
                with Vertical(id='left-rail'):
                    with Horizontal(id='history-toolbar'):
                        yield Static('Conversations', id='history-title')
                        yield Button('New', id='new-conversation-button')
                    yield OptionList(id='history-list')
                with Vertical(id='center-column'):
                    with VerticalScroll(id='conversation-scroll'):
                        yield ConversationFeed()
                with Vertical(id='right-rail'):
                    yield Static(id='details')
            with Horizontal(id='command-picker'):
                yield OptionList(id='command-options')
                yield Static(id='command-description')
            yield Input(
                placeholder='Type a prompt or slash command and press Enter',
                id='prompt',
            )
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(0.12, self._tick_spinner)
            self._set_command_picker_visible(False)
            if self._restored_session is not None:
                restored_turns = restore_conversation_turns(self._restored_session.messages)
                self._bridge.restore_history(restored_turns)
            self._refresh_all_panels()
            self.call_after_refresh(self.action_focus_prompt)
            if self._first_prompt:
                self._submit_prompt(self._first_prompt)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != 'prompt':
                return
            self._refresh_command_picker(event.value)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            prompt = event.value.strip()
            self.query_one('#prompt', Input).value = ''
            self._refresh_command_picker('')
            if not prompt:
                return
            if prompt in {'/exit', '/quit'}:
                self.exit()
                return
            lowered_prompt = prompt.lower()
            if lowered_prompt in {'/new', '/new-chat', '/new-conversation'}:
                self.action_new_conversation()
                return
            if lowered_prompt in {'/back', '/prev', '/previous'}:
                self.action_previous_conversation()
                return
            if lowered_prompt in {'/next', '/forward'}:
                self.action_next_conversation()
                return
            self._submit_prompt(prompt)

        def on_key(self, event: events.Key) -> None:
            prompt = self.query_one('#prompt', Input)
            if self._route_key_to_prompt(event, prompt):
                return
            if self.focused is not prompt or not self._command_picker_visible:
                return
            if event.key == 'down':
                event.stop()
                event.prevent_default()
                self._move_command_selection(1)
                return
            if event.key == 'up':
                event.stop()
                event.prevent_default()
                self._move_command_selection(-1)
                return
            if event.key == 'tab':
                event.stop()
                event.prevent_default()
                self._apply_highlighted_command()
                return
            if event.key == 'escape':
                event.stop()
                event.prevent_default()
                self._set_command_picker_visible(False)
                return
            if event.key == 'enter' and self._should_accept_completion_on_enter(prompt.value):
                event.stop()
                event.prevent_default()
                self._apply_highlighted_command()

        def _route_key_to_prompt(self, event: events.Key, prompt: Input) -> bool:
            if not should_route_key_to_prompt(
                character=getattr(event, 'character', None),
                prompt_focused=self.focused is prompt,
                busy=self._state.busy or prompt.disabled,
            ):
                return False
            prompt.focus()
            prompt.value = f'{prompt.value}{event.character}'
            self._refresh_command_picker(prompt.value)
            event.stop()
            event.prevent_default()
            return True

        def on_option_list_option_highlighted(
            self,
            event: OptionList.OptionHighlighted,
        ) -> None:
            if event.option_list.id == 'command-options':
                suggestion = self._suggestion_for_option_id(event.option_id)
                if suggestion is not None:
                    self.query_one('#command-description', Static).update(
                        render_slash_command_suggestion_detail(suggestion)
                    )
                return

        def on_option_list_option_selected(
            self,
            event: OptionList.OptionSelected,
        ) -> None:
            if event.option_list.id == 'command-options':
                self._apply_command_suggestion(event.option_id)
                event.stop()
                return
            if event.option_list.id == 'history-list':
                item = self._sidebar_item_for_option_id(event.option_id)
                if item is None:
                    return
                if item.kind == 'conversation':
                    self._set_conversation_expanded(item.conversation_id, True)
                    self._switch_to_conversation(item.conversation_id)
                    self._refresh_history_list()
                elif item.kind == 'turn':
                    self._switch_to_conversation(item.conversation_id)
                    self._selected_turn_id = item.turn_id
                    self._refresh_conversation_view()
                    self._refresh_details_panel()
                event.stop()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == 'new-conversation-button':
                self.action_new_conversation()
                event.stop()

        def action_focus_prompt(self) -> None:
            self.query_one('#prompt', Input).focus()

        def action_new_conversation(self) -> None:
            if self._state.busy:
                return
            self._conversation_counter += 1
            conversation_id = f'conversation-{self._conversation_counter}'
            self._conversations.append(
                ConversationThread(
                    conversation_id=conversation_id,
                    title=f'Conversation {self._conversation_counter}',
                )
            )
            self._switch_to_conversation(conversation_id)
            self.call_after_refresh(self.action_focus_prompt)

        def action_previous_conversation(self) -> None:
            self._switch_conversation_by_offset(-1)

        def action_next_conversation(self) -> None:
            self._switch_conversation_by_offset(1)

        def action_refresh_panels(self) -> None:
            self._refresh_all_panels()

        def action_toggle_auto_follow(self) -> None:
            self._auto_follow = not self._auto_follow
            self._refresh_details_panel()
            if self._auto_follow:
                self._scroll_conversation_to_end()

        def action_focus_history(self) -> None:
            self.query_one('#history-list', OptionList).focus()

        def on_collapsible_toggled(self, event) -> None:
            collapsible = getattr(event, 'collapsible', None)
            if collapsible is None:
                collapsible = getattr(event, 'control', None)
            widget_id = getattr(collapsible, 'id', None) if collapsible is not None else None
            if not isinstance(widget_id, str) or not widget_id.startswith('working-'):
                return
            collapsed = getattr(collapsible, 'collapsed', None)
            if isinstance(collapsed, bool):
                self._collapsed_sections[widget_id] = collapsed

        @property
        def _command_picker_visible(self) -> bool:
            return bool(self.query_one('#command-picker', Horizontal).display)

        def _set_command_picker_visible(self, visible: bool) -> None:
            self.query_one('#command-picker', Horizontal).display = visible

        def _make_bridge(self, state: AgentTuiState) -> AgentTuiEventBridge:
            return AgentTuiEventBridge(
                state,
                emit_data=lambda _text: None,
                on_state_change=self._handle_state_change,
                on_turns_change=self._handle_turns_change,
                on_history_change=self._handle_history_change,
                on_activity_change=self._handle_activity_change,
            )

        def _active_conversation(self) -> ConversationThread:
            for conversation in self._conversations:
                if conversation.conversation_id == self._active_conversation_id:
                    return conversation
            return self._conversations[0]

        def _conversation_index(self, conversation_id: str) -> int:
            for index, conversation in enumerate(self._conversations):
                if conversation.conversation_id == conversation_id:
                    return index
            return 0

        def _switch_conversation_by_offset(self, offset: int) -> None:
            if self._state.busy or not self._conversations:
                return
            current_index = self._conversation_index(self._active_conversation_id)
            target_index = max(0, min(len(self._conversations) - 1, current_index + offset))
            if target_index == current_index:
                return
            target_conversation = self._conversations[target_index]
            self._set_conversation_expanded(target_conversation.conversation_id, True)
            self._switch_to_conversation(target_conversation.conversation_id)
            self._refresh_history_list()
            self.call_after_refresh(self.action_focus_prompt)

        def _set_conversation_expanded(self, conversation_id: str, expanded: bool) -> None:
            for conversation in self._conversations:
                if conversation.conversation_id == conversation_id:
                    conversation.expanded = expanded
                    return

        def _sync_active_conversation(self) -> None:
            conversation = self._active_conversation()
            conversation.turns = tuple(self._conversation_turns)
            conversation.session_id = self._active_session_id
            if conversation.turns:
                conversation.title = conversation.turns[0].prompt_preview(max_chars=28)

        def _switch_to_conversation(self, conversation_id: str) -> None:
            if self._state.busy:
                return
            for conversation in self._conversations:
                if conversation.conversation_id == conversation_id:
                    self._active_conversation_id = conversation_id
                    self._active_session_id = conversation.session_id
                    self._state = AgentTuiState.from_agent(self._agent)
                    if conversation.session_id:
                        self._state.session_id = conversation.session_id
                    self._bridge = self._make_bridge(self._state)
                    self._selected_turn_id = (
                        conversation.turns[-1].turn_id if conversation.turns else None
                    )
                    self._bridge.restore_history(conversation.turns, announce_activity=False)
                    self._refresh_all_panels()
                    return

        def _rebuild_sidebar_items(self) -> tuple[SidebarItem, ...]:
            items: list[SidebarItem] = []
            for index, conversation in enumerate(self._conversations, start=1):
                active_marker = '●' if conversation.conversation_id == self._active_conversation_id else '○'
                prefix = '▼' if conversation.expanded else '▶'
                title = conversation.title or f'Conversation {index}'
                items.append(
                    SidebarItem(
                        option_id=f'conversation:{conversation.conversation_id}',
                        label=f'{active_marker} {prefix} {title}',
                        kind='conversation',
                        conversation_id=conversation.conversation_id,
                    )
                )
                if not conversation.expanded:
                    continue
                for turn_index, turn in enumerate(conversation.turns, start=1):
                    items.append(
                        SidebarItem(
                            option_id=f'turn:{conversation.conversation_id}:{turn.turn_id}',
                            label=f'  {turn_index:02d}. {turn.prompt_preview(max_chars=28)}',
                            kind='turn',
                            conversation_id=conversation.conversation_id,
                            turn_id=turn.turn_id,
                        )
                    )
            return tuple(items)

        def _sidebar_item_for_option_id(self, option_id: str | None) -> SidebarItem | None:
            if option_id is None:
                return None
            for item in self._sidebar_items:
                if item.option_id == option_id:
                    return item
            return None

        def _refresh_all_panels(self) -> None:
            self._refresh_conversation_view()
            self._refresh_history_list()
            self._refresh_details_panel()

        def _handle_state_change(self, state: AgentTuiState) -> None:
            workspace_name = Path(state.workspace).name or state.workspace
            self.sub_title = f'{workspace_name} | {state.status}'
            self._active_conversation().session_id = state.session_id
            prompt = self.query_one('#prompt', Input)
            prompt.disabled = state.busy
            if not state.busy:
                prompt.focus()
            else:
                self._set_command_picker_visible(False)
            self._refresh_details_panel()
            self._refresh_conversation_view()
            if self._auto_follow and state.busy:
                self._scroll_conversation_to_end()

        def _handle_turns_change(self, turns: tuple[ConversationTurn, ...]) -> None:
            self._conversation_turns = turns
            self._sync_active_conversation()
            if turns and self._selected_turn_id is None:
                self._selected_turn_id = turns[-1].turn_id
            if self._selected_turn_id not in {turn.turn_id for turn in turns}:
                self._selected_turn_id = turns[-1].turn_id if turns else None
            self._refresh_conversation_view()
            self._refresh_history_list()
            self._refresh_details_panel()
            if self._auto_follow:
                self._scroll_conversation_to_end()

        def _handle_history_change(
            self,
            items: tuple[ConversationHistoryItem, ...],
        ) -> None:
            self._history_items = items
            self._sync_active_conversation()
            self._refresh_history_list()

        def _handle_activity_change(self, items: tuple[ActivityItem, ...]) -> None:
            self._activity_items = items
            self._refresh_details_panel()

        def _refresh_command_picker(self, value: str) -> None:
            suggestions = filter_slash_command_suggestions(
                value,
                suggestions=self._command_suggestions,
            )
            self._visible_command_suggestions = suggestions
            option_list = self.query_one('#command-options', OptionList)
            description = self.query_one('#command-description', Static)
            option_list.clear_options()
            if not suggestions or self._state.busy:
                description.update('')
                self._set_command_picker_visible(False)
                return
            option_list.add_options(
                Option(suggestion.label, id=suggestion.primary_name)
                for suggestion in suggestions
            )
            option_list.highlighted = 0
            description.update(render_slash_command_suggestion_detail(suggestions[0]))
            self._set_command_picker_visible(True)

        def _move_command_selection(self, delta: int) -> None:
            if not self._visible_command_suggestions:
                return
            option_list = self.query_one('#command-options', OptionList)
            current = option_list.highlighted
            if current is None:
                current = 0
            next_index = max(0, min(len(self._visible_command_suggestions) - 1, current + delta))
            option_list.highlighted = next_index
            option_list.scroll_to_highlight()

        def _should_accept_completion_on_enter(self, value: str) -> bool:
            if not self._visible_command_suggestions:
                return False
            query = extract_slash_command_query(value)
            if query is None:
                return False
            if not query:
                return True
            return find_slash_command(query) is None

        def _apply_highlighted_command(self) -> None:
            if not self._visible_command_suggestions:
                return
            option_list = self.query_one('#command-options', OptionList)
            highlighted = option_list.highlighted
            if highlighted is None or highlighted >= len(self._visible_command_suggestions):
                highlighted = 0
            suggestion = self._visible_command_suggestions[highlighted]
            self._apply_command_suggestion(suggestion.primary_name)

        def _apply_command_suggestion(self, option_id: str | None) -> None:
            suggestion = self._suggestion_for_option_id(option_id)
            if suggestion is None:
                return
            prompt = self.query_one('#prompt', Input)
            prompt.value = suggestion.insertion_text
            self._refresh_command_picker(prompt.value)
            prompt.focus()

        def _suggestion_for_option_id(
            self,
            option_id: str | None,
        ) -> SlashCommandSuggestion | None:
            if option_id is None:
                return None
            for suggestion in self._visible_command_suggestions:
                if suggestion.primary_name == option_id:
                    return suggestion
            return None

        def _selected_turn(self) -> ConversationTurn | None:
            if not self._conversation_turns:
                return None
            if self._selected_turn_id is None:
                return self._conversation_turns[-1]
            for turn in self._conversation_turns:
                if turn.turn_id == self._selected_turn_id:
                    return turn
            return self._conversation_turns[-1]

        def _refresh_conversation_view(self) -> None:
            scroll_container = self.query_one('#conversation-scroll', VerticalScroll)
            previous_scroll_y = getattr(scroll_container, 'scroll_y', 0.0)
            conversation = self.query_one('#conversation-feed', ConversationFeed)
            conversation.set_data(
                conversation_id=self._active_conversation_id,
                turns=self._conversation_turns,
                selected_turn_id=self._selected_turn_id,
                phase=self._state.phase,
                phase_detail=self._state.phase_detail,
                busy=self._state.busy,
                spinner_index=self._spinner_index,
                collapsed_sections=self._collapsed_sections,
            )
            scroll_container.refresh(layout=True)
            if self._auto_follow and self._state.busy:
                self._scroll_conversation_to_end()
                return
            try:
                scroll_container.scroll_to(y=previous_scroll_y, animate=False)
            except Exception:
                return

        def _refresh_history_list(self) -> None:
            option_list = self.query_one('#history-list', OptionList)
            option_list.clear_options()
            self._sidebar_items = self._rebuild_sidebar_items()
            if not self._sidebar_items:
                return
            option_list.add_options(
                Option(item.label, id=item.option_id) for item in self._sidebar_items
            )
            selected_index = 0
            active_turn_option_id = (
                f'turn:{self._active_conversation_id}:{self._selected_turn_id}'
                if self._selected_turn_id is not None
                else f'conversation:{self._active_conversation_id}'
            )
            for index, item in enumerate(self._sidebar_items):
                if item.option_id == active_turn_option_id:
                    selected_index = index
                    break
            option_list.highlighted = selected_index

        def _refresh_details_panel(self) -> None:
            turn = self._selected_turn()
            lines = [
                'Details',
                '',
                f'status={self._state.status}',
                f'phase={self._state.phase}',
                f'model={self._state.model}',
                f'permissions={self._state.permissions}',
                f'session_id={self._state.session_id or "none"}',
                f'auto_follow={self._auto_follow}',
                f'prompts={self._state.prompt_count}',
                f'tokens={self._state.total_tokens}',
                f'cost_usd={self._state.total_cost_usd:.6f}',
            ]
            if turn is not None:
                lines.extend(
                    [
                        '',
                        'Selected Turn',
                        '',
                        f'prompt={turn.prompt_preview(max_chars=120)}',
                        f'assistant={turn.assistant_preview(max_chars=120)}',
                        f'tools={turn.tool_count}',
                    ]
                )
            if self._activity_items:
                latest = self._activity_items[-1]
                lines.extend(['', f'last_activity={latest.label}'])
            self.query_one('#details', Static).update('\n'.join(lines))

        def _scroll_conversation_to_end(self) -> None:
            container = self.query_one('#conversation-scroll', VerticalScroll)
            try:
                container.scroll_end(animate=False)
            except AttributeError:
                return

        def _tick_spinner(self) -> None:
            if not self._state.busy:
                return
            self._spinner_index = (self._spinner_index + 1) % 4
            self._refresh_conversation_view()

        def _submit_prompt(self, prompt: str) -> None:
            if self._state.busy:
                return
            self._bridge.begin_prompt(prompt, session_id=self._active_session_id)
            self._run_prompt(prompt)

        @work(thread=True, exclusive=True)
        def _run_prompt(self, prompt: str) -> None:
            try:
                stored_session = None
                if self._active_session_id:
                    stored_session = load_agent_session(
                        self._active_session_id,
                        directory=self._agent.runtime_config.session_directory,
                    )
                if stored_session is None:
                    result = self._agent.run(
                        prompt,
                        event_handler=self._handle_event_from_worker,
                    )
                else:
                    result = self._agent.resume(
                        prompt,
                        stored_session,
                        event_handler=self._handle_event_from_worker,
                    )
            except BaseException as exc:
                self.call_from_thread(self._bridge.fail, exc)
                return
            self.call_from_thread(self._finish_prompt, result)

        def _handle_event_from_worker(self, event: dict[str, object]) -> None:
            self.call_from_thread(self._bridge.handle_event, dict(event))

        def _finish_prompt(self, result: AgentRunResult) -> None:
            self._active_session_id = result.session_id or self._active_session_id
            self._bridge.complete(result)

    app = AgentTuiApp(
        agent,
        first_prompt=initial_prompt,
        resumed_session_id=resume_session_id,
    )
    app.run()
    return 0
