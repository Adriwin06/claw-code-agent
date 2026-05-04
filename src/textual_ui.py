from __future__ import annotations

import re
import time
from pathlib import Path
from threading import Event
from typing import Sequence

from src.agent.commands.slash import find_slash_command
from src.agent.runtime.agent import LocalCodingAgent
from src.agent.models.types import AgentRunResult
from src.session.session_store import StoredAgentSession, load_agent_session
from .ui.conversation import (
    ActivityItem,
    ConversationEntry,
    ConversationHistoryItem,
    ConversationThread,
    ConversationTurn,
    SidebarItem,
    build_conversation_history_items,
    restore_conversation_turns,
)
from .ui.conversation_store import (
    ConversationHistoryStore,
    workspace_history_identity,
    workspace_history_key,
)
from .ui.event_bridge import AgentTuiEventBridge
from .ui.formatting import _friendly_stop_reason, _preview_value
from .ui.ids import (
    build_working_section_id,
    build_working_section_instance_id,
    should_route_key_to_prompt,
)
from .ui.state import AgentTuiState, hydrate_state_from_stored_session
from .ui.slash_commands import (
    SlashCommandSuggestion,
    build_slash_command_suggestions,
    extract_slash_command_query,
    filter_slash_command_suggestions,
    render_slash_command_suggestion_detail,
)


def render_details_panel(
    state: AgentTuiState,
    turn: ConversationTurn | None,
    activity_items: Sequence[ActivityItem],
    *,
    auto_follow: bool,
) -> str:
    max_turns_label = 'unlimited' if state.max_turns is None else str(state.max_turns)
    workspace_path = Path(state.workspace)
    workspace_identity = workspace_history_identity(workspace_path)
    history_key = workspace_history_key(workspace_path)
    lines = [
        'Run',
        '',
        f'status={state.status}',
        f'phase={state.phase}',
        f'phase_detail={state.phase_detail}',
        f'model={state.model}',
        f'permissions={state.permissions}',
        f'workspace={state.workspace}',
        f'workspace_identity={workspace_identity}',
        f'history_key={history_key}',
        f'streaming={state.streaming_enabled}',
        f'max_turns={max_turns_label}',
        f'command_timeout_seconds={state.command_timeout_seconds:.1f}',
        f'session_id={state.session_id or "none"}',
        f'auto_follow={auto_follow}',
        f'busy={state.busy}',
        f'last_tool={state.last_tool or "none"}',
        f'last_stop_reason={_friendly_stop_reason(state.last_stop_reason)}',
        f'input_tokens={state.input_tokens}',
        f'output_tokens={state.output_tokens}',
        f'prompts={state.prompt_count}',
        f'conversation_turns={state.conversation_turns}',
        f'activity_events={state.activity_events}',
        f'last_turns={state.last_turns}',
        f'last_tool_calls={state.last_tool_calls}',
        f'tokens={state.total_tokens}',
        f'cost_usd={state.total_cost_usd:.6f}',
        '',
        'Search',
        '',
        f'enabled={state.search_enabled}',
        f'context_size={state.search_context_size}',
        f'default_max_results={state.search_default_max_results}',
        f'providers={state.search_provider_count}',
        f'manifests={state.search_manifest_count}',
        f'active_provider={state.search_active_provider}',
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
                f'stop_reason={_friendly_stop_reason(turn.stop_reason)}',
            ]
        )
    if activity_items:
        lines.extend(['', 'Activity', ''])
        for item in activity_items[-5:]:
            detail = _preview_activity_detail(item.detail)
            suffix = f': {detail}' if detail else f' ({item.status})'
            lines.append(f'- {item.label}{suffix}')
        latest = activity_items[-1]
        lines.extend(['', f'last_activity={latest.label}'])
    lines.extend(
        [
            '',
            'Next Actions',
            '',
            'Enter: submit prompt',
            'Ctrl+U: reuse selected prompt',
            'Ctrl+T: rerun selected turn',
            'Ctrl+N: new conversation',
            'Ctrl+F: toggle follow',
            'Ctrl+D: delete conversation',
            '/new /prev /next /retry /reuse /delete',
            'Ctrl+C: stop current run',
            'PgUp/PgDn: scroll conversation',
        ]
    )
    return '\n'.join(lines)


def _preview_activity_detail(detail: str, *, max_chars: int = 86) -> str:
    return _preview_value(detail, max_chars=max_chars)


def render_working_entries_markdown(
    entries: Sequence[ConversationEntry],
    *,
    include_live_state: bool,
    phase: str = '',
    phase_detail: str = '',
) -> str:
    if _has_delegate_entries(entries):
        return _render_delegate_working_markdown(
            entries,
            include_live_state=include_live_state,
            phase=phase,
            phase_detail=phase_detail,
        )
    lines: list[str] = []
    if include_live_state:
        lines.extend(
            [
                f'**State:** {phase}',
                '',
                phase_detail or phase,
                '',
            ]
        )
    for entry in entries:
        heading = _working_entry_heading(entry)
        lines.append(f'**{heading}**')
        body = entry.content.strip() or '(empty)'
        if entry.kind in {'tool', 'tool_output', 'tool_result'}:
            lines.extend(['', '```text', body.rstrip(), '```', ''])
        else:
            lines.extend(['', body, ''])
    rendered = '\n'.join(lines).strip()
    return rendered or '_No internal work details._'


def _has_delegate_entries(entries: Sequence[ConversationEntry]) -> bool:
    return any(_is_delegate_entry(entry) for entry in entries)


def _is_delegate_entry(entry: ConversationEntry) -> bool:
    if entry.kind.startswith('delegate_'):
        return True
    if entry.title.startswith(('Sub-Agent', 'Delegate ')):
        return True
    return entry.content.lstrip().startswith('[delegate]')


def _render_delegate_working_markdown(
    entries: Sequence[ConversationEntry],
    *,
    include_live_state: bool,
    phase: str,
    phase_detail: str,
) -> str:
    output_entries = [entry for entry in entries if entry.kind == 'delegate_output']
    result_entries = [entry for entry in entries if entry.kind == 'delegate_result']
    progress_entries = [entry for entry in entries if entry.kind == 'delegate_progress']
    warning_entries = [
        entry for entry in entries if entry.kind in {'warning', 'error'} and entry.content.strip()
    ]

    agent_name = _delegate_agent_name(entries)
    label = _delegate_task_label(entries)
    status = _delegate_status(entries, include_live_state=include_live_state)
    stats = _delegate_result_stats(result_entries)
    latest_progress = _delegate_latest_progress(progress_entries)

    heading_parts = [agent_name]
    if label and label != agent_name:
        heading_parts.append(label)
    heading_parts.append(status)
    if stats:
        heading_parts.append(stats)
    lines = [f'**{" · ".join(heading_parts)}**']

    if include_live_state and not output_entries:
        detail = latest_progress or phase_detail or phase or 'running'
        lines.extend(['', detail])
    elif latest_progress and not output_entries:
        lines.extend(['', latest_progress])

    for entry in output_entries:
        body = entry.content.strip()
        if not body:
            continue
        if len(output_entries) > 1:
            output_label = _delegate_label_from_title(entry.title) or 'Output'
            lines.extend(['', f'**{output_label}**', '', body])
        else:
            lines.extend(['', body])

    for entry in warning_entries:
        lines.extend(['', f'**{_working_entry_heading(entry)}**', '', entry.content.strip()])

    rendered = '\n'.join(lines).strip()
    return rendered or '_Sub-agent is starting._'


def _delegate_agent_name(entries: Sequence[ConversationEntry]) -> str:
    for entry in entries:
        if entry.title.startswith('Sub-Agent Requested:'):
            name = entry.title.split(':', 1)[1].strip()
            if name:
                return name
        for pattern in (r'\bsubagent=([^\s]+)', r'\bsubagent_type=([^\s]+)'):
            match = re.search(pattern, entry.content)
            if match:
                return match.group(1)
    return 'Sub-agent'


def _delegate_task_label(entries: Sequence[ConversationEntry]) -> str:
    for entry in entries:
        if entry.kind in {'delegate_output', 'delegate_progress', 'delegate_result'}:
            label = _delegate_label_from_title(entry.title)
            if label:
                return label
    for entry in entries:
        match = re.search(r'\blabel=([^\s]+)', entry.content)
        if match:
            return match.group(1)
    return ''


def _delegate_label_from_title(title: str) -> str:
    if ':' not in title:
        return ''
    prefix, _, suffix = title.partition(':')
    if prefix.startswith('Sub-Agent'):
        return suffix.strip()
    return ''


def _delegate_status(
    entries: Sequence[ConversationEntry],
    *,
    include_live_state: bool,
) -> str:
    for entry in reversed(entries):
        if entry.kind.startswith('delegate_') and entry.status == 'error':
            return 'failed'
    for entry in reversed(entries):
        if entry.kind == 'delegate_result' and entry.status == 'ok':
            return 'completed'
    if include_live_state or any(
        entry.kind.startswith('delegate_') and entry.status == 'running'
        for entry in entries
    ):
        return 'running'
    return 'completed'


def _delegate_result_stats(entries: Sequence[ConversationEntry]) -> str:
    for entry in reversed(entries):
        turns = re.search(r'\bturns=([0-9]+)', entry.content)
        tools = re.search(r'\btool_calls=([0-9]+)', entry.content)
        stop = re.search(r'\bstop_reason=([^\s]+)', entry.content)
        parts: list[str] = []
        if turns:
            parts.append(f'turns {turns.group(1)}')
        if tools:
            parts.append(f'tools {tools.group(1)}')
        if stop and stop.group(1) not in {'stop', 'completed'}:
            parts.append(stop.group(1))
        if parts:
            return ', '.join(parts)
    return ''


def _delegate_latest_progress(entries: Sequence[ConversationEntry]) -> str:
    for entry in reversed(entries):
        content = entry.content.strip()
        if content:
            return content
    return ''


def _working_entry_heading(entry: ConversationEntry) -> str:
    title = entry.title.strip()
    if title:
        return title
    if entry.kind == 'thinking':
        return 'Thinking'
    if entry.kind == 'tool':
        return 'Tool Call'
    if entry.kind == 'tool_output':
        return 'Tool Output'
    if entry.kind == 'tool_result':
        return 'Tool Result'
    if entry.kind == 'delegate_output':
        return 'Sub-Agent Output'
    if entry.kind == 'delegate_result':
        return 'Sub-Agent Result'
    if entry.kind == 'error':
        return 'Error'
    if entry.kind == 'warning':
        return 'Warning'
    if entry.kind == 'status':
        return 'Status'
    return entry.title


def conversation_turns_render_signature(
    turns: Sequence[ConversationTurn],
) -> tuple[object, ...]:
    return tuple(
        (
            turn.turn_id,
            turn.user_prompt,
            turn.assistant_response,
            turn.assistant_status,
            turn.phase_label,
            turn.tool_count,
            turn.restored,
            turn.stop_reason,
            turn.session_id,
            tuple(
                (
                    entry.entry_id,
                    entry.kind,
                    entry.title,
                    entry.content,
                    entry.status,
                    entry.merge_key,
                )
                for entry in turn.entries
            ),
        )
        for turn in turns
    )


def run_agent_tui(
    agent: LocalCodingAgent,
    *,
    initial_prompt: str | None = None,
    resume_session_id: str | None = None,
    history_store: ConversationHistoryStore | None = None,
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
            has_delegate_activity = any(
                entry.title.startswith('Sub-Agent') for entry in entries
            )
            if include_live_state:
                spinner_frames = ('|', '/', '-', '\\')
                spinner = spinner_frames[self._spinner_index % len(spinner_frames)]
                label = 'Delegating' if has_delegate_activity else 'Working'
                base = f'{spinner} {label}'
            elif has_delegate_activity:
                base = 'Sub-agents'
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
            return render_working_entries_markdown(
                entries,
                include_live_state=include_live_state,
                phase=self._phase,
                phase_detail=self._phase_detail,
            )

        def _working_entry_heading(self, entry: ConversationEntry) -> str:
            return _working_entry_heading(entry)

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
            friendly_reason = _friendly_stop_reason(self._turn.stop_reason)
            if self._active and self._busy:
                status_label = 'in progress'
            elif self._turn.assistant_status.lower() == 'error' or any(
                entry.kind == 'error' for entry in self._turn.entries
            ):
                status_label = 'error'
            elif friendly_reason == 'completed':
                status_label = 'done'
            else:
                status_label = 'stopped'
            parts = [status_label]
            if self._turn.tool_count:
                parts.append(f'tools={self._turn.tool_count}')
            if friendly_reason != 'completed':
                parts.append(f'reason={friendly_reason}')
            if self._turn.restored:
                parts.append('restored')
            return ' | '.join(parts)

    class _TuiRunCancelled(Exception):
        pass

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
            self._render_signature: tuple[object, ...] | None = None

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
            render_signature = (
                conversation_id,
                self._turns_render_signature(turns),
                selected_turn_id,
                phase,
                phase_detail,
                busy,
                spinner_index,
                tuple(sorted(collapsed_sections.items())),
            )
            if render_signature == self._render_signature:
                return
            self._render_signature = render_signature
            self._conversation_id = conversation_id
            self._turns = turns
            self._selected_turn_id = selected_turn_id
            self._state_phase = phase
            self._state_phase_detail = phase_detail
            self._state_busy = busy
            self._spinner_index = spinner_index
            self._collapsed_sections = dict(collapsed_sections)
            self.refresh(recompose=True, layout=True)

        def _turns_render_signature(
            self,
            turns: tuple[ConversationTurn, ...],
        ) -> tuple[object, ...]:
            return conversation_turns_render_signature(turns)

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

        #actions-toolbar {
            height: auto;
            margin-bottom: 1;
        }

        #run-toolbar {
            height: auto;
            margin-bottom: 1;
        }

        #actions-toolbar Button,
        #run-toolbar Button {
            width: 1fr;
            min-width: 10;
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
            ('ctrl+u', 'reuse_selected_prompt', 'Reuse'),
            ('ctrl+t', 'retry_selected_turn', 'Retry'),
            ('ctrl+d', 'delete_conversation', 'Delete'),
            ('ctrl+r', 'refresh_panels', 'Refresh'),
            ('ctrl+f', 'toggle_auto_follow', 'Follow'),
            ('ctrl+c', 'stop_generation', 'Stop'),
            ('ctrl+h', 'focus_history', 'History'),
        ]

        def __init__(
            self,
            runtime_agent: LocalCodingAgent,
            *,
            first_prompt: str | None,
            resumed_session_id: str | None,
            history_store: ConversationHistoryStore | None,
        ) -> None:
            super().__init__()
            self._agent = runtime_agent
            self._first_prompt = first_prompt.strip() if first_prompt else None
            self._active_session_id = resumed_session_id
            self._workspace = self._agent.runtime_config.cwd
            self._history_store = history_store or ConversationHistoryStore()
            self._state = AgentTuiState.from_agent(runtime_agent)
            self._command_suggestions = build_slash_command_suggestions()
            self._visible_command_suggestions: list[SlashCommandSuggestion] = []
            snapshot = self._history_store.load_workspace(self._workspace)
            self._conversations = list(snapshot.conversations)
            if not self._conversations:
                self._conversations = [
                    ConversationThread(
                        conversation_id='conversation-1',
                        title=('Resumed conversation' if resumed_session_id else 'Conversation 1'),
                    )
                ]
            self._conversation_counter = self._max_conversation_counter(self._conversations)
            self._active_conversation_id = self._initial_active_conversation_id(
                snapshot.active_conversation_id,
                resumed_session_id=resumed_session_id,
            )
            self._active_session_id = self._active_conversation().session_id or resumed_session_id
            self._sidebar_items: tuple[SidebarItem, ...] = ()
            self._conversation_turns: tuple[ConversationTurn, ...] = ()
            self._history_items: tuple[ConversationHistoryItem, ...] = ()
            self._activity_items: tuple[ActivityItem, ...] = ()
            self._selected_turn_id: str | None = None
            self._auto_follow = True
            self._follow_scroll_pending = False
            self._restore_scroll_pending = False
            self._restore_scroll_y: float | None = None
            self._last_follow_scroll_at = 0.0
            self._cancel_requested = Event()
            self._active_worker = None
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
                    self._active_conversation().session_id = resumed_session_id
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
                    with Horizontal(id='actions-toolbar'):
                        yield Button('Reuse', id='reuse-prompt-button')
                        yield Button('Retry', id='retry-turn-button')
                        yield Button('Delete', id='delete-conversation-button')
                    with Horizontal(id='run-toolbar'):
                        yield Button('Stop', id='stop-run-button')
                        yield Button('Follow On', id='follow-button')
                    yield Static(id='details')
            with Horizontal(id='command-picker'):
                yield OptionList(id='command-options')
                yield Static(id='command-description')
            yield Input(
                placeholder='Type a task, /retry, /reuse, /new, /delete, or another slash command',
                id='prompt',
            )
            yield Footer()

        def on_mount(self) -> None:
            self.set_interval(0.25, self._tick_spinner)
            self._set_command_picker_visible(False)
            if self._restored_session is not None:
                restored_turns = restore_conversation_turns(self._restored_session.messages)
                self._bridge.restore_history(restored_turns)
            else:
                conversation = self._active_conversation()
                self._active_session_id = conversation.session_id
                self._selected_turn_id = (
                    conversation.turns[-1].turn_id if conversation.turns else None
                )
                self._bridge.restore_history(
                    conversation.turns,
                    announce_activity=bool(conversation.turns),
                )
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
            if lowered_prompt in {'/retry', '/rerun'}:
                self.action_retry_selected_turn()
                return
            if lowered_prompt in {'/reuse', '/edit-selected'}:
                self.action_reuse_selected_prompt()
                return
            if lowered_prompt in {'/delete', '/delete-conversation'}:
                self.action_delete_conversation()
                return
            self._submit_prompt(prompt)

        def on_key(self, event: events.Key) -> None:
            prompt = self.query_one('#prompt', Input)
            if self._handle_conversation_scroll_key(event, prompt):
                return
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

        def on_mouse_scroll_up(self, event) -> None:
            if self._event_targets_conversation(event):
                self._set_auto_follow(False)

        def on_mouse_scroll_down(self, event) -> None:
            if self._event_targets_conversation(event):
                self.call_after_refresh(self._sync_auto_follow_after_user_scroll)

        def _handle_conversation_scroll_key(self, event: events.Key, prompt: Input) -> bool:
            key = event.key
            if key not in {'pageup', 'pagedown', 'home', 'end'}:
                return False
            if self.focused is prompt and key in {'home', 'end'}:
                return False
            event.stop()
            event.prevent_default()
            if key == 'pageup':
                self._set_auto_follow(False)
                self._scroll_conversation_by_pages(-1)
                return True
            if key == 'pagedown':
                self._scroll_conversation_by_pages(1)
                self.call_after_refresh(self._sync_auto_follow_after_user_scroll)
                return True
            if key == 'home':
                self._set_auto_follow(False)
                self._scroll_conversation_to_y(0.0)
                return True
            self._set_auto_follow(True, scroll=True)
            return True

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
                return
            if event.button.id == 'reuse-prompt-button':
                self.action_reuse_selected_prompt()
                event.stop()
                return
            if event.button.id == 'retry-turn-button':
                self.action_retry_selected_turn()
                event.stop()
                return
            if event.button.id == 'delete-conversation-button':
                self.action_delete_conversation()
                event.stop()
                return
            if event.button.id == 'stop-run-button':
                self.action_stop_generation()
                event.stop()
                return
            if event.button.id == 'follow-button':
                self.action_toggle_auto_follow()
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
            self._persist_history()
            self.call_after_refresh(self.action_focus_prompt)

        def action_previous_conversation(self) -> None:
            self._switch_conversation_by_offset(-1)

        def action_next_conversation(self) -> None:
            self._switch_conversation_by_offset(1)

        def action_refresh_panels(self) -> None:
            self._refresh_all_panels()

        def action_toggle_auto_follow(self) -> None:
            self._set_auto_follow(not self._auto_follow, scroll=True)

        def action_stop_generation(self) -> None:
            if not self._state.busy:
                return
            self._cancel_requested.set()
            self._bridge.request_cancel(
                'Stop requested; waiting for the current model or tool call to yield'
            )
            self._refresh_details_panel()

        def action_focus_history(self) -> None:
            self.query_one('#history-list', OptionList).focus()

        def action_reuse_selected_prompt(self) -> None:
            if self._state.busy:
                return
            turn = self._selected_turn()
            if turn is None:
                return
            prompt = self.query_one('#prompt', Input)
            prompt.value = turn.user_prompt
            self._refresh_command_picker(prompt.value)
            prompt.focus()

        def action_retry_selected_turn(self) -> None:
            if self._state.busy:
                return
            turn = self._selected_turn()
            if turn is None:
                return
            self._submit_prompt(turn.user_prompt)

        def action_delete_conversation(self) -> None:
            if self._state.busy or not self._conversations:
                return
            delete_index = self._conversation_index(self._active_conversation_id)
            deleted = self._conversations[delete_index]
            self._delete_session_file(deleted.session_id)
            remaining = [
                conversation
                for conversation in self._conversations
                if conversation.conversation_id != deleted.conversation_id
            ]
            if not remaining:
                self._conversation_counter = 1
                remaining = [
                    ConversationThread(
                        conversation_id='conversation-1',
                        title='Conversation 1',
                    )
                ]
                target_index = 0
            else:
                target_index = max(0, min(delete_index, len(remaining) - 1))
            self._conversations = remaining
            target = self._conversations[target_index]
            self._active_conversation_id = target.conversation_id
            self._active_session_id = target.session_id
            self._state = AgentTuiState.from_agent(self._agent)
            if target.session_id:
                self._state.session_id = target.session_id
            self._bridge = self._make_bridge(self._state)
            self._selected_turn_id = target.turns[-1].turn_id if target.turns else None
            self._bridge.restore_history(target.turns, announce_activity=False)
            self._persist_history()
            self._refresh_all_panels()
            self.call_after_refresh(self.action_focus_prompt)

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

        def _initial_active_conversation_id(
            self,
            stored_active_id: str | None,
            *,
            resumed_session_id: str | None,
        ) -> str:
            if resumed_session_id:
                for conversation in self._conversations:
                    if conversation.session_id == resumed_session_id:
                        return conversation.conversation_id
                if (
                    len(self._conversations) == 1
                    and not self._conversations[0].turns
                    and self._conversations[0].session_id is None
                ):
                    self._conversations[0].title = 'Resumed conversation'
                    self._conversations[0].session_id = resumed_session_id
                    return self._conversations[0].conversation_id
                self._conversation_counter += 1
                conversation_id = f'conversation-{self._conversation_counter}'
                self._conversations.append(
                    ConversationThread(
                        conversation_id=conversation_id,
                        title='Resumed conversation',
                        session_id=resumed_session_id,
                    )
                )
                return conversation_id
            if stored_active_id:
                for conversation in self._conversations:
                    if conversation.conversation_id == stored_active_id:
                        return conversation.conversation_id
            return self._conversations[-1].conversation_id

        @staticmethod
        def _max_conversation_counter(conversations: Sequence[ConversationThread]) -> int:
            counter = 0
            for conversation in conversations:
                prefix, _, suffix = conversation.conversation_id.rpartition('-')
                if prefix != 'conversation':
                    continue
                try:
                    counter = max(counter, int(suffix))
                except ValueError:
                    continue
            return max(counter, len(conversations), 1)

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
                    self._persist_history()
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
                    self._persist_history()
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
            state.refresh_from_agent(self._agent)
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
            self._refresh_conversation_view(allow_auto_follow=False)
            if not state.busy:
                self._persist_history()

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

        def _refresh_conversation_view(self, *, allow_auto_follow: bool = True) -> None:
            scroll_container = self.query_one('#conversation-scroll', VerticalScroll)
            previous_scroll_y = getattr(scroll_container, 'scroll_y', 0.0)
            last_turn_id = (
                self._conversation_turns[-1].turn_id if self._conversation_turns else None
            )
            selected_latest_turn = (
                self._selected_turn_id is None or self._selected_turn_id == last_turn_id
            )
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
            if allow_auto_follow and self._auto_follow and selected_latest_turn:
                self._schedule_conversation_scroll_to_end()
                return
            self._schedule_conversation_scroll_restore(previous_scroll_y)

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
            self._state.refresh_from_agent(self._agent)
            turn = self._selected_turn()
            reuse_button = self.query_one('#reuse-prompt-button', Button)
            retry_button = self.query_one('#retry-turn-button', Button)
            stop_button = self.query_one('#stop-run-button', Button)
            delete_button = self.query_one('#delete-conversation-button', Button)
            can_use_turn = turn is not None and not self._state.busy
            reuse_button.disabled = not can_use_turn
            retry_button.disabled = not can_use_turn
            stop_button.disabled = not self._state.busy
            delete_button.disabled = self._state.busy
            self._refresh_follow_button()
            self.query_one('#details', Static).update(
                render_details_panel(
                    self._state,
                    turn,
                    self._activity_items,
                    auto_follow=self._auto_follow,
                )
            )

        def _set_auto_follow(self, enabled: bool, *, scroll: bool = False) -> None:
            changed = self._auto_follow != enabled
            self._auto_follow = enabled
            if not enabled:
                self._follow_scroll_pending = False
            if changed:
                self._refresh_follow_button()
                self._refresh_details_panel()
            if enabled and scroll:
                self._schedule_conversation_scroll_to_end(force=True)

        def _refresh_follow_button(self) -> None:
            try:
                button = self.query_one('#follow-button', Button)
            except Exception:
                return
            label = 'Follow On' if self._auto_follow else 'Follow Off'
            try:
                button.label = label
            except Exception:
                try:
                    button.update(label)
                except Exception:
                    return

        def _event_targets_conversation(self, event) -> bool:
            target = getattr(event, 'widget', None)
            if target is None:
                target = getattr(event, 'sender', None)
            if target is None:
                return True
            while target is not None:
                if getattr(target, 'id', None) in {'conversation-scroll', 'conversation-feed'}:
                    return True
                target = getattr(target, 'parent', None)
            return False

        def _conversation_scroll_container(self):
            return self.query_one('#conversation-scroll', VerticalScroll)

        def _conversation_at_end(self) -> bool:
            container = self._conversation_scroll_container()
            try:
                scroll_y = float(getattr(container, 'scroll_y', 0.0))
                max_scroll_y = float(getattr(container, 'max_scroll_y', 0.0))
            except (TypeError, ValueError):
                return True
            return max_scroll_y - scroll_y <= 1.0

        def _sync_auto_follow_after_user_scroll(self) -> None:
            if self._conversation_at_end():
                self._set_auto_follow(True)

        def _scroll_conversation_by_pages(self, pages: int) -> None:
            container = self._conversation_scroll_container()
            size = getattr(container, 'size', None)
            page_height = max(int(getattr(size, 'height', 20)) - 2, 1)
            current_y = float(getattr(container, 'scroll_y', 0.0))
            max_y = float(getattr(container, 'max_scroll_y', 0.0))
            target_y = max(0.0, min(max_y, current_y + (pages * page_height)))
            self._scroll_conversation_to_y(target_y)

        def _scroll_conversation_to_y(self, y: float) -> None:
            container = self._conversation_scroll_container()
            try:
                container.scroll_to(y=y, animate=False, immediate=True)
            except TypeError:
                try:
                    container.scroll_to(y=y, animate=False)
                except Exception:
                    return
            except Exception:
                return

        def _scroll_conversation_to_end(self) -> None:
            self._follow_scroll_pending = False
            if not self._auto_follow:
                return
            container = self.query_one('#conversation-scroll', VerticalScroll)
            try:
                container.scroll_end(animate=False, immediate=True)
            except TypeError:
                try:
                    container.scroll_end(animate=False)
                except AttributeError:
                    return
            except AttributeError:
                return

        def _schedule_conversation_scroll_to_end(self, *, force: bool = False) -> None:
            self._restore_scroll_y = None
            if self._follow_scroll_pending:
                return
            now = time.monotonic()
            if (
                not force
                and self._state.busy
                and now - self._last_follow_scroll_at < 0.35
            ):
                return
            self._last_follow_scroll_at = now
            self._follow_scroll_pending = True
            self.call_after_refresh(self._scroll_conversation_to_end)

        def _schedule_conversation_scroll_restore(self, scroll_y: float) -> None:
            if self._follow_scroll_pending:
                return
            self._restore_scroll_y = scroll_y
            if self._restore_scroll_pending:
                return
            self._restore_scroll_pending = True
            self.call_after_refresh(self._restore_conversation_scroll)

        def _restore_conversation_scroll(self) -> None:
            self._restore_scroll_pending = False
            if self._follow_scroll_pending:
                return
            scroll_y = self._restore_scroll_y
            self._restore_scroll_y = None
            if scroll_y is None:
                return
            container = self.query_one('#conversation-scroll', VerticalScroll)
            try:
                container.scroll_to(y=scroll_y, animate=False, immediate=True)
            except TypeError:
                try:
                    container.scroll_to(y=scroll_y, animate=False)
                except Exception:
                    return
            except Exception:
                return

        def _tick_spinner(self) -> None:
            if not self._state.busy:
                return
            self._spinner_index = (self._spinner_index + 1) % 4
            self._refresh_details_panel()

        def _submit_prompt(self, prompt: str) -> None:
            if self._state.busy:
                return
            self._cancel_requested.clear()
            self._set_auto_follow(True, scroll=True)
            self._bridge.begin_prompt(prompt, session_id=self._active_session_id)
            self._active_worker = self._run_prompt(prompt)

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
                if self._cancel_requested.is_set():
                    raise _TuiRunCancelled()
            except _TuiRunCancelled:
                self.call_from_thread(self._finish_cancelled_prompt)
                return
            except BaseException as exc:
                self.call_from_thread(self._finish_failed_prompt, exc)
                return
            self.call_from_thread(self._finish_prompt, result)

        def _handle_event_from_worker(self, event: dict[str, object]) -> None:
            if self._cancel_requested.is_set():
                raise _TuiRunCancelled()
            self.call_from_thread(self._bridge.handle_event, dict(event))

        def _finish_cancelled_prompt(self) -> None:
            self._active_worker = None
            self._cancel_requested.clear()
            self._bridge.cancel('Stopped by user')
            self._persist_history()

        def _finish_failed_prompt(self, error: BaseException) -> None:
            self._active_worker = None
            self._cancel_requested.clear()
            self._bridge.fail(error)
            self._persist_history()

        def _finish_prompt(self, result: AgentRunResult) -> None:
            self._active_worker = None
            if self._cancel_requested.is_set():
                self._finish_cancelled_prompt()
                return
            self._active_session_id = result.session_id or self._active_session_id
            self._bridge.complete(result)
            self._persist_history()

        def _persist_history(self) -> None:
            try:
                if not any(
                    conversation.turns or conversation.session_id
                    for conversation in self._conversations
                ):
                    try:
                        self._history_store.workspace_path(self._workspace).unlink()
                    except FileNotFoundError:
                        pass
                    return
                self._history_store.save_workspace_conversations(
                    self._workspace,
                    self._conversations,
                    active_conversation_id=self._active_conversation_id,
                )
            except OSError:
                return

        def _delete_session_file(self, session_id: str | None) -> None:
            if not session_id:
                return
            session_directory = self._agent.runtime_config.session_directory.resolve()
            session_path = (session_directory / f'{session_id}.json').resolve()
            if session_path.parent != session_directory:
                return
            try:
                session_path.unlink()
            except FileNotFoundError:
                return
            except OSError:
                return

    app = AgentTuiApp(
        agent,
        first_prompt=initial_prompt,
        resumed_session_id=resume_session_id,
        history_store=history_store,
    )
    app.run()
    return 0
