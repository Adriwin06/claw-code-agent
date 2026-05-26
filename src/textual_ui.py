from __future__ import annotations

from datetime import datetime, timezone
import re
from pathlib import Path
from threading import Event
from typing import Sequence

try:
    from rich.text import Text
except ModuleNotFoundError:  # pragma: no cover - Rich is provided by Textual.
    Text = None  # type: ignore[assignment]

from src.agent.commands.slash import find_slash_command
from src.agent.runtime.agent import LocalCodingAgent
from src.agent.models.types import AgentRunResult
from src.session.session_store import StoredAgentSession, load_agent_session
from .ui.attachments import (
    PromptAttachment,
    build_display_prompt_with_attachments,
    build_prompt_image_blocks,
    build_prompt_with_references,
    copy_clipboard_image_attachment,
    copy_external_attachment,
    extract_prompt_file_paths,
    extract_unresolved_prompt_file_path_candidates,
    parse_pasted_file_paths,
    render_attachment_summary,
    workspace_reference_for_pasted_path,
)
from .ui.conversation import (
    ActivityItem,
    ConversationEntry,
    ConversationHistoryItem,
    ConversationThread,
    ConversationTurn,
    SidebarItem,
    build_conversation_history_items,
    restore_conversation_turns,
    sanitize_user_prompt_display_text,
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
from .ui.workspace_files import (
    WorkspacePathSuggestion,
    build_workspace_path_suggestions,
    extract_workspace_path_references,
    extract_workspace_reference_query,
    filter_workspace_path_suggestions,
    render_workspace_path_suggestion_detail,
)


def render_details_panel(
    state: AgentTuiState,
    turn: ConversationTurn | None,
    activity_items: Sequence[ActivityItem],
    *,
    hide_gitignored_paths: bool = True,
    color: bool = False,
) -> object:
    max_turns_label = 'unlimited' if state.max_turns is None else str(state.max_turns)
    workspace_path = Path(state.workspace)
    workspace_identity = workspace_history_identity(workspace_path)
    history_key = workspace_history_key(workspace_path)
    sections: list[tuple[str, list[str]]] = [
        (
            'Run',
            [
                f'status={state.status}',
                f'phase={state.phase}',
                f'phase_detail={state.phase_detail}',
                f'busy={state.busy}',
                f'last_tool={state.last_tool or "none"}',
                f'last_stop_reason={_friendly_stop_reason(state.last_stop_reason)}',
            ],
        ),
        (
            'Config',
            [
                f'model={state.model}',
                f'permissions={state.permissions}',
                f'workspace={state.workspace}',
                f'workspace_identity={workspace_identity}',
                f'history_key={history_key}',
                f'streaming={state.streaming_enabled}',
                f'max_turns={max_turns_label}',
                f'command_timeout_seconds={state.command_timeout_seconds:.1f}',
                f'session_id={state.session_id or "none"}',
            ],
        ),
        (
            'Usage',
            [
                f'input_tokens={state.input_tokens}',
                f'output_tokens={state.output_tokens}',
                f'prompts={state.prompt_count}',
                f'conversation_turns={state.conversation_turns}',
                f'activity_events={state.activity_events}',
                f'last_turns={state.last_turns}',
                f'last_tool_calls={state.last_tool_calls}',
                f'tokens={state.total_tokens}',
                f'cost_usd={state.total_cost_usd:.6f}',
            ],
        ),
        (
            'Workspace',
            [
                f'workspace_change_events={state.workspace_change_events}',
                f'workspace_changed_files={state.workspace_changed_files}',
                f'workspace_added_lines={state.workspace_added_lines}',
                f'workspace_removed_lines={state.workspace_removed_lines}',
            ],
        ),
        (
            'Search',
            [
                f'enabled={state.search_enabled}',
                f'context_size={state.search_context_size}',
                f'default_max_results={state.search_default_max_results}',
                f'providers={state.search_provider_count}',
                f'manifests={state.search_manifest_count}',
                f'active_provider={state.search_active_provider}',
            ],
        ),
    ]
    if turn is not None:
        sections.append(
            (
                'Selected Turn',
                [
                    f'prompt={turn.prompt_preview(max_chars=120)}',
                    f'assistant={turn.assistant_preview(max_chars=120)}',
                    f'tools={turn.tool_count}',
                    f'stop_reason={_friendly_stop_reason(turn.stop_reason)}',
                ],
            )
        )
    if activity_items:
        activity_lines: list[str] = []
        for item in activity_items[-5:]:
            detail = _preview_activity_detail(item.detail)
            suffix = f': {detail}' if detail else f' ({item.status})'
            activity_lines.append(f'- {item.label}{suffix}')
        latest = activity_items[-1]
        activity_lines.append(f'last_activity={latest.label}')
        sections.append(('Activity', activity_lines))
    sections.append(
        (
            'Next Actions',
            [
                'Enter: submit prompt',
                'Ctrl+U: reuse selected prompt',
                'Ctrl+T: rerun selected turn',
                'Ctrl+N: new conversation',
                'Ctrl+D: delete conversation',
                '/new /prev /next /retry /reuse /delete',
                '@path: reference workspace files or folders',
                f'Ctrl+G: {"show" if hide_gitignored_paths else "hide"} gitignored @ files',
                'Paste/drop file paths: attach external files',
                'Ctrl+I: attach clipboard image',
                'Ctrl+C: stop current run',
                'PgUp/PgDn: scroll conversation',
            ],
        )
    )
    if color and Text is not None:
        return _render_details_panel_text(sections)
    return _render_details_panel_plain(sections)


_DETAIL_SECTION_STYLES = {
    'Run': '#58a6ff',
    'Config': '#a371f7',
    'Usage': '#d29922',
    'Workspace': '#56d4dd',
    'Search': '#3fb950',
    'Selected Turn': '#ffab70',
    'Activity': '#f78166',
    'Next Actions': '#8b949e',
}


def _render_details_panel_plain(sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    lines: list[str] = []
    for title, section_lines in sections:
        if lines:
            lines.append('')
        lines.append(title)
        lines.append('-' * max(10, len(title)))
        lines.extend(section_lines)
    return '\n'.join(lines)


def _render_details_panel_text(sections: Sequence[tuple[str, Sequence[str]]]) -> object:
    rendered = Text()
    for section_index, (title, section_lines) in enumerate(sections):
        if section_index:
            rendered.append('\n')
        section_style = _DETAIL_SECTION_STYLES.get(title, '#58a6ff')
        rendered.append(f'{title}\n', style=f'bold {section_style}')
        rendered.append(f'{"-" * max(10, len(title))}\n', style=section_style)
        for line in section_lines:
            _append_detail_panel_line(rendered, line)
            rendered.append('\n')
    rendered.rstrip()
    return rendered


def _append_detail_panel_line(rendered: object, line: str) -> None:
    if Text is None:
        return
    text = rendered
    if line.startswith('- '):
        text.append('- ', style='#8b949e')
        _append_activity_line_text(text, line[2:])
        return
    if '=' in line:
        key, value = line.split('=', 1)
        text.append(key, style='#8b949e')
        text.append('=', style='#6e7681')
        text.append(value, style=_detail_value_style(key, value))
        return
    if ':' in line:
        action, detail = line.split(':', 1)
        text.append(action, style='bold #79c0ff')
        text.append(':', style='#6e7681')
        text.append(detail, style='#c9d1d9')
        return
    text.append(line, style='#c9d1d9')


def _append_activity_line_text(rendered: object, line: str) -> None:
    if Text is None:
        return
    if ':' in line:
        label, detail = line.split(':', 1)
        rendered.append(label, style='#e6edf3')
        rendered.append(':', style='#6e7681')
        rendered.append(detail, style='#c9d1d9')
        return
    rendered.append(line, style='#e6edf3')


def _detail_value_style(key: str, value: str) -> str:
    normalized = value.lower()
    if normalized in {'true', 'ready', 'running', 'completed', 'resumed'}:
        return 'bold #3fb950'
    if normalized in {'false', 'none', 'idle'}:
        return '#8b949e'
    if normalized in {'error', 'failed', 'cancelled'}:
        return 'bold #f85149'
    if key in {'workspace', 'workspace_identity', 'history_key'}:
        return '#79c0ff'
    if key in {'permissions', 'model', 'active_provider'}:
        return '#d2a8ff'
    if (
        key.endswith('_tokens')
        or key.endswith('_turns')
        or key.endswith('_calls')
        or key.endswith('_events')
        or key.endswith('_files')
        or key.endswith('_lines')
        or key in {'tokens', 'prompts', 'providers', 'manifests', 'tools', 'cost_usd'}
    ):
        return '#ffa657'
    return '#e6edf3'


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


def render_changes_panel_title(
    event: dict[str, object] | None,
    *,
    color: bool = False,
) -> object:
    if not event:
        return 'Changes'
    file_count = _coerce_positive_int(event.get('file_count'))
    added = _coerce_positive_int(event.get('added_lines'))
    removed = _coerce_positive_int(event.get('removed_lines'))
    if file_count <= 0:
        return 'Changes'
    label = 'file' if file_count == 1 else 'files'
    if color and Text is not None:
        title = Text(f'Changes: {file_count} {label} changed ')
        _append_colored_change_counts(title, added=added, removed=removed)
        return title
    return f'Changes: {file_count} {label} changed +{added} -{removed}'


def render_changes_panel_body(
    event: dict[str, object] | None,
    *,
    color: bool = False,
) -> object:
    if not event:
        return 'No changes in this conversation.'
    if color and Text is not None:
        return _render_changes_panel_body_text(event)
    title = render_changes_panel_title(event).removeprefix('Changes: ')
    lines = [title, '']
    files = event.get('files')
    if isinstance(files, list):
        for file_payload in files:
            if not isinstance(file_payload, dict):
                continue
            line = _render_changes_panel_file(file_payload)
            if line:
                lines.append(line)
    truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
    if truncated_file_count:
        lines.append(f'... {truncated_file_count} more changed file(s)')
    return '\n'.join(lines).rstrip()


def render_changes_detail_panel_title(
    event: dict[str, object] | None,
    *,
    color: bool = False,
) -> object:
    if not event:
        return 'Added / Removed'
    added = _coerce_positive_int(event.get('added_lines'))
    removed = _coerce_positive_int(event.get('removed_lines'))
    if color and Text is not None:
        title = Text('Added / Removed ')
        _append_colored_change_counts(title, added=added, removed=removed)
        return title
    return f'Added / Removed +{added} -{removed}'


def render_changes_detail_panel_body(
    event: dict[str, object] | None,
    *,
    color: bool = False,
) -> object:
    if not event:
        return 'No added or removed lines available.'
    if color and Text is not None:
        return _render_changes_detail_panel_body_text(event)

    lines: list[str] = []
    files = event.get('files')
    if isinstance(files, list):
        for file_payload in files:
            if not isinstance(file_payload, dict):
                continue
            rendered = _render_changes_detail_file(file_payload)
            if rendered:
                if lines:
                    lines.append('')
                lines.extend(rendered)
    truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
    if truncated_file_count:
        if lines:
            lines.append('')
        lines.append(f'... {truncated_file_count} more changed file(s)')
    if not lines:
        return 'No added or removed lines available.'
    return '\n'.join(lines).rstrip()


def _render_changes_panel_body_text(event: dict[str, object]) -> object:
    file_count = _coerce_positive_int(event.get('file_count'))
    added = _coerce_positive_int(event.get('added_lines'))
    removed = _coerce_positive_int(event.get('removed_lines'))
    if file_count <= 0:
        return Text('No changes in this conversation.')
    label = 'file' if file_count == 1 else 'files'
    rendered = Text(f'{file_count} {label} changed ')
    _append_colored_change_counts(rendered, added=added, removed=removed)
    rendered.append('\n\n')
    files = event.get('files')
    if isinstance(files, list):
        for file_payload in files:
            if not isinstance(file_payload, dict):
                continue
            _append_changes_panel_file_text(rendered, file_payload)
    truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
    if truncated_file_count:
        rendered.append(f'... {truncated_file_count} more changed file(s)\n')
    rendered.rstrip()
    return rendered


def _render_changes_detail_panel_body_text(event: dict[str, object]) -> object:
    rendered = Text()
    files = event.get('files')
    has_content = False
    if isinstance(files, list):
        for file_payload in files:
            if not isinstance(file_payload, dict):
                continue
            has_content = (
                _append_changes_detail_file_text(rendered, file_payload)
                or has_content
            )
    truncated_file_count = _coerce_positive_int(event.get('truncated_file_count'))
    if truncated_file_count:
        if has_content:
            rendered.append('\n')
        rendered.append(f'... {truncated_file_count} more changed file(s)')
        has_content = True
    if not has_content:
        rendered.append('No added or removed lines available.')
    rendered.rstrip()
    return rendered


def _render_changes_panel_file(file_payload: dict[str, object]) -> str:
    path = _preview_value(file_payload.get('path'), max_chars=160)
    if not path:
        return ''
    added = _coerce_positive_int(file_payload.get('added_lines'))
    removed = _coerce_positive_int(file_payload.get('removed_lines'))
    status = _preview_value(file_payload.get('status')) or 'modified'
    suffix = '' if status == 'modified' else f' {status}'
    return f'- {path} +{added} -{removed}{suffix}'


def _render_changes_detail_file(file_payload: dict[str, object]) -> list[str]:
    path = _preview_value(file_payload.get('path'), max_chars=160)
    if not path:
        return []
    added_lines, removed_lines, truncated = _extract_diff_added_removed_lines(
        file_payload.get('diff')
    )
    if not added_lines and not removed_lines:
        if file_payload.get('binary'):
            return [f'{path}', 'binary or non-text file']
        if file_payload.get('content_truncated'):
            return [f'{path}', 'large file; diff omitted']
        return []
    added = _coerce_positive_int(file_payload.get('added_lines'))
    removed = _coerce_positive_int(file_payload.get('removed_lines'))
    lines = [f'{path} +{added} -{removed}']
    if added_lines:
        lines.append('Added')
        lines.extend(added_lines)
    if removed_lines:
        lines.append('Removed')
        lines.extend(removed_lines)
    if truncated or file_payload.get('diff_truncated'):
        lines.append('...[diff truncated]...')
    return lines


def _append_changes_panel_file_text(
    rendered: object,
    file_payload: dict[str, object],
) -> None:
    if Text is None or not isinstance(rendered, Text):
        return
    path = _preview_value(file_payload.get('path'), max_chars=160)
    if not path:
        return
    added = _coerce_positive_int(file_payload.get('added_lines'))
    removed = _coerce_positive_int(file_payload.get('removed_lines'))
    status = _preview_value(file_payload.get('status')) or 'modified'
    path_style = _change_status_style(status)
    rendered.append('- ')
    rendered.append(path, style=path_style)
    rendered.append(' ')
    _append_colored_change_counts(rendered, added=added, removed=removed)
    if status != 'modified':
        rendered.append(' ')
        rendered.append(status, style=path_style)
    rendered.append('\n')


def _append_changes_detail_file_text(
    rendered: object,
    file_payload: dict[str, object],
) -> bool:
    if Text is None or not isinstance(rendered, Text):
        return False
    path = _preview_value(file_payload.get('path'), max_chars=160)
    if not path:
        return False
    added_lines, removed_lines, truncated = _extract_diff_added_removed_lines(
        file_payload.get('diff')
    )
    if not added_lines and not removed_lines:
        if file_payload.get('binary'):
            _append_changes_detail_message(rendered, path, 'binary or non-text file')
            return True
        if file_payload.get('content_truncated'):
            _append_changes_detail_message(rendered, path, 'large file; diff omitted')
            return True
        return False
    if len(rendered):
        rendered.append('\n\n')
    status = _preview_value(file_payload.get('status')) or 'modified'
    rendered.append(path, style=_change_status_style(status))
    rendered.append(' ')
    _append_colored_change_counts(
        rendered,
        added=_coerce_positive_int(file_payload.get('added_lines')),
        removed=_coerce_positive_int(file_payload.get('removed_lines')),
    )
    if status != 'modified':
        rendered.append(' ')
        rendered.append(status, style=_change_status_style(status))
    rendered.append('\n')
    if added_lines:
        rendered.append('Added\n', style='bold green')
        for line in added_lines:
            rendered.append(line, style='green')
            rendered.append('\n')
    if removed_lines:
        rendered.append('Removed\n', style='bold red')
        for line in removed_lines:
            rendered.append(line, style='red')
            rendered.append('\n')
    if truncated or file_payload.get('diff_truncated'):
        rendered.append('...[diff truncated]...', style='yellow')
    rendered.rstrip()
    return True


def _append_changes_detail_message(
    rendered: object,
    path: str,
    message: str,
) -> None:
    if Text is None or not isinstance(rendered, Text):
        return
    if len(rendered):
        rendered.append('\n\n')
    rendered.append(path)
    rendered.append('\n')
    rendered.append(message)


def _append_colored_change_counts(
    rendered: object,
    *,
    added: int,
    removed: int,
) -> None:
    if Text is None or not isinstance(rendered, Text):
        return
    rendered.append(f'+{added}', style='green')
    rendered.append(' ')
    rendered.append(f'-{removed}', style='red')


def _change_status_style(status: str) -> str:
    if status == 'added':
        return 'green'
    if status == 'deleted':
        return 'red'
    return ''


def _extract_diff_added_removed_lines(diff: object) -> tuple[list[str], list[str], bool]:
    if not isinstance(diff, str) or not diff.strip():
        return [], [], False
    added: list[str] = []
    removed: list[str] = []
    truncated = False
    for raw_line in diff.splitlines():
        if raw_line == '...[diff truncated]...':
            truncated = True
            continue
        if raw_line.startswith('+++') or raw_line.startswith('---'):
            continue
        if raw_line.startswith('+'):
            added.append(_preview_value(raw_line, max_chars=240) or '+')
            continue
        if raw_line.startswith('-'):
            removed.append(_preview_value(raw_line, max_chars=240) or '-')
    return added, removed, truncated


def _coerce_positive_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


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
    if entry.kind == 'workspace_change':
        return 'Workspace Changes'
    if entry.kind == 'workspace_change_recap':
        return 'Changed Files'
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


def conversation_scroll_is_at_end(
    scroll_y: object,
    max_scroll_y: object,
    *,
    tolerance: float = 1.0,
) -> bool:
    try:
        current_y = float(scroll_y)
        end_y = float(max_scroll_y)
    except (TypeError, ValueError):
        return True
    return end_y - current_y <= tolerance


def should_follow_conversation_bottom(
    *,
    allow_stick_to_bottom: bool,
    selected_latest_turn: bool,
    was_at_end: bool,
    following_bottom: bool,
) -> bool:
    if not allow_stick_to_bottom or not selected_latest_turn:
        return False
    return was_at_end or following_bottom


def conversation_follow_after_user_scroll(
    *,
    scroll_direction: str,
    was_following: bool,
    at_end: bool,
) -> bool:
    if scroll_direction == 'up':
        return False
    if scroll_direction == 'down':
        return was_following or at_end
    return at_end


def _utc_log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


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
            Markdown as TextualMarkdown,
            OptionList,
            Static,
            TextArea,
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

        def update_state(
            self,
            turn: ConversationTurn,
            *,
            turn_index: int,
            selected: bool,
            active: bool,
            busy: bool,
            phase: str,
            phase_detail: str,
            spinner_index: int,
            collapsed_sections: dict[str, bool],
        ) -> None:
            # update in place via recompose to keep the widget mounted
            # (mount+remove caused the container height to oscillate, which
            # made the scroll-to-bottom polling bounce up/down)
            self._turn = turn
            self._turn_index = turn_index
            self._selected = selected
            self._active = active
            self._busy = busy
            self._phase = phase
            self._phase_detail = phase_detail
            self._spinner_index = spinner_index
            self._collapsed_sections = dict(collapsed_sections)
            if selected:
                self.add_class('selected')
            else:
                self.remove_class('selected')
            try:
                self.refresh(recompose=True)
            except TypeError:
                self.refresh()

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
            # incremental child management: keep TurnCard widgets mounted across
            # refreshes so the scroll position never resets to the top
            self._card_signatures: dict[str, tuple[object, ...]] = {}
            self._cards: dict[str, TurnCard] = {}
            self._empty_placeholder = None

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
            full_recompose = conversation_id != self._conversation_id
            self._render_signature = render_signature
            self._conversation_id = conversation_id
            self._turns = turns
            self._selected_turn_id = selected_turn_id
            self._state_phase = phase
            self._state_phase_detail = phase_detail
            self._state_busy = busy
            self._spinner_index = spinner_index
            self._collapsed_sections = dict(collapsed_sections)
            if full_recompose or not self.is_mounted:
                self._card_signatures.clear()
                self._cards.clear()
                self._empty_placeholder = None
                self.refresh(recompose=True, layout=True)
                return
            self._apply_incremental_update()

        def _apply_incremental_update(self) -> None:
            if self._empty_placeholder is not None:
                try:
                    self._empty_placeholder.remove()
                except Exception:
                    pass
                self._empty_placeholder = None
            if not self._turns:
                for card in list(self._cards.values()):
                    try:
                        card.remove()
                    except Exception:
                        pass
                self._cards.clear()
                self._card_signatures.clear()
                empty = Static('No conversation yet. Submit a prompt to start.')
                empty.add_class('conversation-empty')
                self.mount(empty)
                self._empty_placeholder = empty
                return
            new_turn_ids = {turn.turn_id for turn in self._turns}
            for turn_id in list(self._cards.keys()):
                if turn_id not in new_turn_ids:
                    try:
                        self._cards[turn_id].remove()
                    except Exception:
                        pass
                    del self._cards[turn_id]
                    self._card_signatures.pop(turn_id, None)
            last_turn_id = self._turns[-1].turn_id
            previous_card: TurnCard | None = None
            for index, turn in enumerate(self._turns, start=1):
                signature = self._signature_for(turn, index, last_turn_id)
                existing = self._cards.get(turn.turn_id)
                if existing is not None and self._card_signatures.get(turn.turn_id) == signature:
                    previous_card = existing
                    continue
                if existing is not None:
                    # update existing card in place; avoids the transient
                    # double-height that causes scroll bouncing during tool streams
                    existing.update_state(
                        turn,
                        turn_index=index,
                        selected=(turn.turn_id == self._selected_turn_id),
                        active=(turn.turn_id == last_turn_id),
                        busy=self._state_busy and turn.turn_id == last_turn_id,
                        phase=self._state_phase,
                        phase_detail=self._state_phase_detail,
                        spinner_index=self._spinner_index,
                        collapsed_sections=self._collapsed_sections,
                    )
                    self._card_signatures[turn.turn_id] = signature
                    previous_card = existing
                    continue
                new_card = TurnCard(
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
                if previous_card is not None:
                    self.mount(new_card, after=previous_card)
                else:
                    self.mount(new_card)
                self._cards[turn.turn_id] = new_card
                self._card_signatures[turn.turn_id] = signature
                previous_card = new_card

        def _signature_for(
            self,
            turn: ConversationTurn,
            index: int,
            last_turn_id: str | None,
        ) -> tuple[object, ...]:
            is_last = turn.turn_id == last_turn_id
            return (
                index,
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
                turn.turn_id == self._selected_turn_id,
                is_last,
                self._state_busy and is_last,
                self._state_phase if is_last else '',
                self._state_phase_detail if is_last else '',
                self._spinner_index if (self._state_busy and is_last) else 0,
                tuple(sorted(self._collapsed_sections.items())),
            )

        def _turns_render_signature(
            self,
            turns: tuple[ConversationTurn, ...],
        ) -> tuple[object, ...]:
            return conversation_turns_render_signature(turns)

        def compose(self) -> ComposeResult:
            self._card_signatures.clear()
            self._cards.clear()
            self._empty_placeholder = None
            if not self._turns:
                empty = Static('No conversation yet. Submit a prompt to start.')
                empty.add_class('conversation-empty')
                self._empty_placeholder = empty
                yield empty
                return
            last_turn_id = self._turns[-1].turn_id
            for index, turn in enumerate(self._turns, start=1):
                self._card_signatures[turn.turn_id] = self._signature_for(
                    turn, index, last_turn_id,
                )
                card = TurnCard(
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
                self._cards[turn.turn_id] = card
                yield card

    class PromptInput(TextArea):
        def __init__(self, *args, placeholder: str = '', **kwargs) -> None:
            _ = placeholder
            super().__init__(*args, soft_wrap=True, show_line_numbers=False, **kwargs)

        @property
        def value(self) -> str:
            return self.text

        @value.setter
        def value(self, text: str) -> None:
            self.load_text(text)

        @property
        def cursor_position(self) -> int:
            row, column = self.cursor_location
            lines = self.text.split('\n')
            return sum(len(line) + 1 for line in lines[:row]) + column

        @cursor_position.setter
        def cursor_position(self, offset: int) -> None:
            self.move_cursor(self._location_from_offset(offset))

        def insert_text_at_cursor(self, text: str) -> None:
            result = self.insert(text, maintain_selection_offset=False)
            self.move_cursor(result.end_location)

        async def _on_key(self, event: events.Key) -> None:
            handler = getattr(self.app, '_handle_prompt_editor_key', None)
            if callable(handler) and handler(event):
                event.stop()
                event.prevent_default()
                return
            await super()._on_key(event)

        async def _on_paste(self, event: events.Paste) -> None:
            handler = getattr(self.app, '_handle_prompt_paste', None)
            if callable(handler) and handler(event.text):
                event.stop()
                event.prevent_default()
                return
            await super()._on_paste(event)

        def _location_from_offset(self, offset: int) -> tuple[int, int]:
            remaining = max(0, min(offset, len(self.text)))
            lines = self.text.split('\n')
            for row, line in enumerate(lines):
                line_length = len(line)
                if remaining <= line_length:
                    return row, remaining
                remaining -= line_length + 1
            return len(lines) - 1, len(lines[-1]) if lines else 0

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
            width: 38;
            min-width: 28;
        }

        #actions-toolbar {
            height: auto;
            margin-bottom: 1;
        }

        #actions-toolbar Button {
            width: 1fr;
            min-width: 10;
        }

        #reuse-prompt-button {
            color: #79c0ff;
        }

        #retry-turn-button {
            color: #ffdf5d;
        }

        #delete-conversation-button {
            color: #ffa198;
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

        #details-scroll {
            height: 1fr;
            border: round #2f81f7;
            background: #0d141d;
            overflow-y: auto;
        }

        #details {
            width: 1fr;
            height: auto;
            padding: 1 2 2 2;
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

        #attachment-shelf {
            height: auto;
            margin: 0 1;
            border: round #d29922;
            background: #111923;
            padding: 0 1;
        }

        #changes-panel {
            height: auto;
            margin: 0 1;
            border: round #3b4b5c;
            background: #0f1720;
        }

        #changes-body {
            width: 1fr;
            height: auto;
            max-height: 10;
            padding: 0 1;
            color: #e6edf3;
        }

        #changes-detail-panel {
            height: auto;
            margin: 0 1 1 1;
            border: round #3b4b5c;
            background: #0f1720;
        }

        #changes-detail-scroll {
            width: 1fr;
            height: auto;
            max-height: 18;
        }

        #changes-detail-body {
            width: 1fr;
            height: auto;
            padding: 0 1;
            color: #e6edf3;
        }

        #attachment-summary {
            width: 1fr;
            height: auto;
            color: #e6edf3;
        }

        #clear-attachments-button {
            width: 9;
            min-width: 8;
            margin-left: 1;
        }

        #prompt-row {
            height: auto;
            margin: 0 1 1 1;
        }

        #prompt {
            width: 1fr;
            height: 5;
            min-height: 3;
            margin: 0;
        }

        #stop-run-button {
            width: 10;
            min-width: 8;
            margin-left: 1;
            height: 3;
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
            ('ctrl+c', 'stop_generation', 'Stop'),
            ('ctrl+h', 'focus_history', 'History'),
            ('ctrl+g', 'toggle_gitignored_workspace_paths', 'Ignored'),
            ('ctrl+i', 'attach_clipboard_image', 'ClipImg'),
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
            self._tui_log_path = self._workspace / '.port_sessions' / 'tui-debug.log'
            self._history_store = history_store or ConversationHistoryStore()
            self._state = AgentTuiState.from_agent(runtime_agent)
            self._command_suggestions = build_slash_command_suggestions()
            self._visible_command_suggestions: list[SlashCommandSuggestion] = []
            self._hide_gitignored_workspace_paths = True
            self._workspace_path_suggestions = self._build_workspace_path_suggestions()
            self._visible_workspace_path_suggestions: list[WorkspacePathSuggestion] = []
            self._active_workspace_reference_query = None
            self._picker_mode: str | None = None
            self._prompt_attachments: tuple[PromptAttachment, ...] = ()
            self._attachment_batch_id = ''
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
            self._changes_summary_event: dict[str, object] | None = None
            self._changes_collapsed = True
            self._selected_turn_id: str | None = None
            self._conversation_follow_bottom = True
            self._conversation_pinned_scroll_y: float | None = None
            self._scroll_to_end_pending = False
            self._scroll_to_end_token = 0
            self._scroll_to_end_smooth = False
            self._restore_scroll_pending = False
            self._restore_scroll_y: float | None = None
            self._restore_scroll_token = 0
            self._cancel_requested = Event()
            self._active_worker = None
            self._worker_event_counts: dict[str, int] = {}
            self._spinner_index = 0
            self._collapsed_sections: dict[str, bool] = {}
            # debounced panel refresh while streaming
            self._panels_refresh_pending = False
            # debounced async history persistence
            self._persist_history_dirty = False
            self._persist_history_scheduled = False
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
                    with VerticalScroll(id='details-scroll'):
                        yield Static(id='details')
            with Horizontal(id='command-picker'):
                yield OptionList(id='command-options')
                yield Static(id='command-description')
            with Horizontal(id='attachment-shelf'):
                yield Static(id='attachment-summary')
                yield Button('Clear', id='clear-attachments-button')
            changes_panel = Collapsible(title='Changes', collapsed=True)
            changes_panel.id = 'changes-panel'
            with changes_panel:
                yield Static('No changes in this conversation.', id='changes-body')
                changes_detail_panel = Collapsible(
                    title='Added / Removed',
                    collapsed=True,
                )
                changes_detail_panel.id = 'changes-detail-panel'
                with changes_detail_panel:
                    with VerticalScroll(id='changes-detail-scroll'):
                        yield Static(
                            'No added or removed lines available.',
                            id='changes-detail-body',
                        )
            with Horizontal(id='prompt-row'):
                yield PromptInput(
                    placeholder='Type a task, / command, @ workspace path, or paste/drop file paths',
                    id='prompt',
                )
                yield Button('Stop', id='stop-run-button')
            yield Footer()

        def on_mount(self) -> None:
            self._debug_log(
                'mount',
                session=bool(self._active_session_id),
                conversations=len(self._conversations),
            )
            self.set_interval(0.25, self._tick_spinner)
            self._set_command_picker_visible(False)
            self._refresh_attachment_shelf()
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

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            if event.text_area.id != 'prompt':
                return
            self._refresh_command_picker(
                event.text_area.text,
                cursor_position=getattr(event.text_area, 'cursor_position', None),
            )

        def _submit_prompt_from_editor(self) -> None:
            prompt_editor = self.query_one('#prompt', PromptInput)
            prompt = prompt_editor.value.strip()
            prompt_editor.value = ''
            self._refresh_command_picker('')
            extracted_prompt = self._extract_prompt_path_attachments(prompt)
            if extracted_prompt is None:
                prompt_editor.value = prompt
                prompt_editor.cursor_position = len(prompt)
                prompt_editor.focus()
                return
            prompt = extracted_prompt
            if not prompt and not self._prompt_attachments:
                return
            if prompt in {'/exit', '/quit'} and not self._prompt_attachments:
                self.exit()
                return
            lowered_prompt = prompt.lower()
            if lowered_prompt in {'/new', '/new-chat', '/new-conversation'} and not self._prompt_attachments:
                self.action_new_conversation()
                return
            if lowered_prompt in {'/back', '/prev', '/previous'} and not self._prompt_attachments:
                self.action_previous_conversation()
                return
            if lowered_prompt in {'/next', '/forward'} and not self._prompt_attachments:
                self.action_next_conversation()
                return
            if lowered_prompt in {'/retry', '/rerun'} and not self._prompt_attachments:
                self.action_retry_selected_turn()
                return
            if lowered_prompt in {'/reuse', '/edit-selected'} and not self._prompt_attachments:
                self.action_reuse_selected_prompt()
                return
            if lowered_prompt in {'/delete', '/delete-conversation'} and not self._prompt_attachments:
                self.action_delete_conversation()
                return
            display_prompt = prompt or 'Please inspect the attached file(s).'
            runtime_prompt = self._runtime_prompt_from_display_prompt(display_prompt)
            prompt_blocks = build_prompt_image_blocks(
                self._prompt_attachments,
                self._workspace,
            )
            display_prompt = build_display_prompt_with_attachments(
                display_prompt,
                self._prompt_attachments,
            )
            self._clear_prompt_attachments()
            self._submit_prompt(
                display_prompt,
                runtime_prompt=runtime_prompt,
                prompt_blocks=prompt_blocks,
            )

        def on_key(self, event: events.Key) -> None:
            prompt = self.query_one('#prompt', PromptInput)
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

        def _handle_prompt_editor_key(self, event: events.Key) -> bool:
            prompt = self.query_one('#prompt', PromptInput)
            if event.key == 'shift+enter':
                prompt.insert_text_at_cursor('\n')
                self._refresh_command_picker(
                    prompt.value,
                    cursor_position=prompt.cursor_position,
                )
                return True
            if self._command_picker_visible and event.key == 'down':
                self._move_command_selection(1)
                return True
            if self._command_picker_visible and event.key == 'up':
                self._move_command_selection(-1)
                return True
            if self._command_picker_visible and event.key == 'tab':
                self._apply_highlighted_command()
                return True
            if self._command_picker_visible and event.key == 'escape':
                self._set_command_picker_visible(False)
                return True
            if event.key == 'enter':
                if self._command_picker_visible and self._should_accept_completion_on_enter(
                    prompt.value
                ):
                    self._apply_highlighted_command()
                    return True
                self._submit_prompt_from_editor()
                return True
            return False

        def on_paste(self, event: events.Paste) -> None:
            prompt = self.query_one('#prompt', PromptInput)
            if self.focused is prompt:
                return
            if self._handle_prompt_paste(event.text):
                event.stop()
                event.prevent_default()
                return
            if self._state.busy:
                return
            prompt.focus()
            prompt.insert_text_at_cursor(event.text)
            self._refresh_command_picker(
                prompt.value,
                cursor_position=prompt.cursor_position,
            )
            event.stop()
            event.prevent_default()

        def on_mouse_scroll_up(self, event) -> None:
            if self._event_targets_conversation(event):
                self._handle_conversation_user_scroll('up')

        def on_mouse_scroll_down(self, event) -> None:
            if self._event_targets_conversation(event):
                self._handle_conversation_user_scroll('down')

        def _handle_conversation_scroll_key(self, event: events.Key, prompt: PromptInput) -> bool:
            key = event.key
            if key not in {'pageup', 'pagedown', 'home', 'end'}:
                return False
            if self.focused is prompt and key in {'home', 'end'}:
                return False
            event.stop()
            event.prevent_default()
            if key == 'pageup':
                self._scroll_conversation_by_pages(-1)
                return True
            if key == 'pagedown':
                self._scroll_conversation_by_pages(1)
                return True
            if key == 'home':
                self._scroll_conversation_to_y(0.0)
                return True
            self._stop_pending_scroll_to_end()
            self._cancel_pending_scroll_restore()
            self._conversation_follow_bottom = True
            self._scroll_conversation_to_current_end(smooth=True)
            return True

        def _route_key_to_prompt(self, event: events.Key, prompt: PromptInput) -> bool:
            if not should_route_key_to_prompt(
                character=getattr(event, 'character', None),
                prompt_focused=self.focused is prompt,
                busy=self._state.busy or prompt.disabled,
            ):
                return False
            prompt.focus()
            prompt.value = f'{prompt.value}{event.character}'
            prompt.cursor_position = len(prompt.value)
            self._refresh_command_picker(prompt.value, cursor_position=prompt.cursor_position)
            event.stop()
            event.prevent_default()
            return True

        def on_option_list_option_highlighted(
            self,
            event: OptionList.OptionHighlighted,
        ) -> None:
            if event.option_list.id == 'command-options':
                if self._picker_mode == 'workspace':
                    path_suggestion = self._workspace_suggestion_for_option_id(event.option_id)
                    if path_suggestion is not None:
                        self.query_one('#command-description', Static).update(
                            render_workspace_path_suggestion_detail(path_suggestion)
                        )
                    return
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
                if self._picker_mode == 'workspace':
                    self._apply_workspace_path_suggestion(event.option_id)
                else:
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
            if event.button.id == 'clear-attachments-button':
                self._clear_prompt_attachments()
                event.stop()
                return

        def action_focus_prompt(self) -> None:
            self.query_one('#prompt', PromptInput).focus()

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
            self._workspace_path_suggestions = self._build_workspace_path_suggestions()
            self._refresh_all_panels()

        def action_toggle_gitignored_workspace_paths(self) -> None:
            if self._state.busy:
                return
            self._hide_gitignored_workspace_paths = not self._hide_gitignored_workspace_paths
            self._workspace_path_suggestions = self._build_workspace_path_suggestions()
            prompt = self.query_one('#prompt', PromptInput)
            self._refresh_command_picker(
                prompt.value,
                cursor_position=prompt.cursor_position,
            )
            self._refresh_details_panel()

        def action_attach_clipboard_image(self) -> None:
            if self._state.busy:
                return
            if not self._attachment_batch_id:
                self._attachment_batch_id = f'prompt-{len(self._conversation_turns) + 1}'
            attachment = copy_clipboard_image_attachment(
                self._workspace,
                batch_id=self._attachment_batch_id,
            )
            if attachment is None:
                self._notify_user('No clipboard image was available to attach.')
                return
            self._add_prompt_attachments((attachment,))
            self.query_one('#prompt', PromptInput).focus()

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
            prompt = self.query_one('#prompt', PromptInput)
            prompt.value = turn.user_prompt
            prompt.cursor_position = len(prompt.value)
            self._refresh_command_picker(prompt.value, cursor_position=prompt.cursor_position)
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
            self._reset_conversation_scroll_state()
            self._bridge.restore_history(target.turns, announce_activity=False)
            self._persist_history()
            self._refresh_all_panels()
            self.call_after_refresh(self.action_focus_prompt)

        def on_collapsible_toggled(self, event) -> None:
            collapsible = getattr(event, 'collapsible', None)
            if collapsible is None:
                collapsible = getattr(event, 'control', None)
            widget_id = getattr(collapsible, 'id', None) if collapsible is not None else None
            if widget_id == 'changes-panel':
                collapsed = getattr(collapsible, 'collapsed', None)
                if isinstance(collapsed, bool):
                    self._changes_collapsed = collapsed
                return
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

        def _build_workspace_path_suggestions(self) -> tuple[WorkspacePathSuggestion, ...]:
            return build_workspace_path_suggestions(
                self._workspace,
                hide_gitignored=self._hide_gitignored_workspace_paths,
            )

        def _notify_user(self, message: str) -> None:
            notify = getattr(self, 'notify', None)
            if callable(notify):
                try:
                    notify(message)
                except Exception:
                    return

        def _refresh_attachment_shelf(self) -> None:
            shelf = self.query_one('#attachment-shelf', Horizontal)
            summary = self.query_one('#attachment-summary', Static)
            clear_button = self.query_one('#clear-attachments-button', Button)
            shelf.display = bool(self._prompt_attachments)
            summary.update(render_attachment_summary(self._prompt_attachments))
            clear_button.disabled = not self._prompt_attachments or self._state.busy

        def _clear_prompt_attachments(self) -> None:
            self._prompt_attachments = ()
            self._attachment_batch_id = ''
            self._refresh_attachment_shelf()

        def _add_prompt_attachments(self, attachments: Sequence[PromptAttachment]) -> None:
            if not attachments:
                return
            existing_paths = {attachment.workspace_path for attachment in self._prompt_attachments}
            merged = list(self._prompt_attachments)
            for attachment in attachments:
                if attachment.workspace_path in existing_paths:
                    continue
                existing_paths.add(attachment.workspace_path)
                merged.append(attachment)
            self._prompt_attachments = tuple(merged)
            self._refresh_attachment_shelf()

        def _handle_prompt_paste(self, text: str) -> bool:
            if self._state.busy:
                return False
            remaining_text, paths = extract_prompt_file_paths(text)
            if not paths:
                return False
            prompt = self.query_one('#prompt', PromptInput)
            workspace_mentions, attachments = self._attachments_from_paths(paths)
            insertion = self._prompt_text_with_workspace_mentions(
                remaining_text,
                workspace_mentions,
            )
            if insertion:
                if prompt.value and not prompt.value.endswith((' ', '\n')):
                    insertion = f' {insertion}'
                prompt.insert_text_at_cursor(insertion)
                self._refresh_command_picker(
                    prompt.value,
                    cursor_position=getattr(prompt, 'cursor_position', None),
                )
            self._add_prompt_attachments(attachments)
            return bool(workspace_mentions or attachments)

        def _attachments_from_paths(
            self,
            paths: Sequence[Path],
        ) -> tuple[list[str], list[PromptAttachment]]:
            workspace_mentions: list[str] = []
            attachments: list[PromptAttachment] = []
            if not self._attachment_batch_id:
                self._attachment_batch_id = f'prompt-{len(self._conversation_turns) + 1}'
            for path in paths:
                workspace_reference = workspace_reference_for_pasted_path(path, self._workspace)
                if workspace_reference is not None:
                    if any(character.isspace() for character in workspace_reference):
                        workspace_mentions.append(f'@"{workspace_reference}"')
                    else:
                        workspace_mentions.append(f'@{workspace_reference}')
                    continue
                try:
                    attachment = copy_external_attachment(
                        path,
                        self._workspace,
                        batch_id=self._attachment_batch_id,
                    )
                except OSError:
                    attachment = None
                if attachment is not None:
                    attachments.append(attachment)
            return workspace_mentions, attachments

        def _extract_prompt_path_attachments(self, prompt: str) -> str | None:
            # slash commands are not file paths — bypass path detection
            stripped = prompt.lstrip()
            if stripped.startswith('/') and not self._prompt_attachments:
                parts = stripped[1:].split(None, 1)
                if parts and find_slash_command(parts[0].lower()) is not None:
                    return prompt
            remaining_text, paths = extract_prompt_file_paths(prompt)
            if not paths:
                unresolved = extract_unresolved_prompt_file_path_candidates(prompt)
                if unresolved:
                    self._notify_user(self._attachment_resolution_failure_message(unresolved))
                    return None
                return prompt
            workspace_mentions, attachments = self._attachments_from_paths(paths)
            if not workspace_mentions and not attachments:
                self._notify_user('Could not attach the resolved file path.')
                return None
            self._add_prompt_attachments(attachments)
            return self._prompt_text_with_workspace_mentions(
                remaining_text,
                workspace_mentions,
            )

        def _attachment_resolution_failure_message(
            self,
            unresolved: Sequence[str],
        ) -> str:
            root = unresolved[0] if unresolved else 'that file'
            if len(root) > 54:
                root = f'...{root[-51:]}'
            return (
                f'Could not attach {root}. Check the launch "host attachments" path '
                'covers the file, then restart the TUI.'
            )

        def _prompt_text_with_workspace_mentions(
            self,
            prompt: str,
            workspace_mentions: Sequence[str],
        ) -> str:
            prompt = prompt.strip()
            if not workspace_mentions:
                return prompt
            mention_text = ' '.join(workspace_mentions)
            if not prompt:
                return mention_text
            return f'{mention_text} {prompt}'

        def _prompt_text_as_only_paths(self, text: str) -> tuple[Path, ...]:
            paths = parse_pasted_file_paths(text)
            if not paths:
                return ()
            lines = [line for line in text.splitlines() if line.strip()]
            if len(lines) > 1 and len(paths) != len(lines):
                return ()
            return paths

        def _runtime_prompt_from_display_prompt(self, prompt: str) -> str:
            return build_prompt_with_references(
                prompt,
                attachments=self._prompt_attachments,
                workspace_references=extract_workspace_path_references(
                    prompt,
                    self._workspace,
                ),
            )

        def _make_bridge(self, state: AgentTuiState) -> AgentTuiEventBridge:
            return AgentTuiEventBridge(
                state,
                emit_data=lambda _text: None,
                on_state_change=self._handle_state_change,
                on_turns_change=self._handle_turns_change,
                on_history_change=self._handle_history_change,
                on_activity_change=self._handle_activity_change,
                on_changes_change=self._handle_changes_change,
            )

        def _debug_log(self, event: str, **fields: object) -> None:
            try:
                self._tui_log_path.parent.mkdir(parents=True, exist_ok=True)
                parts = [_utc_log_timestamp(), event]
                for key, value in fields.items():
                    rendered = _preview_value(value, max_chars=180)
                    parts.append(f'{key}={rendered}')
                with self._tui_log_path.open('a', encoding='utf-8') as handle:
                    handle.write(' '.join(parts).rstrip() + '\n')
            except Exception:
                return

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
                    self._reset_conversation_scroll_state()
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
            self._refresh_changes_panel()

        def _handle_state_change(self, state: AgentTuiState) -> None:
            state.refresh_from_agent(self._agent)
            workspace_name = Path(state.workspace).name or state.workspace
            self.sub_title = f'{workspace_name} | {state.status}'
            self._active_conversation().session_id = state.session_id
            prompt = self.query_one('#prompt', PromptInput)
            prompt.disabled = state.busy
            if not state.busy:
                # workspace path suggestions are rebuilt only on explicit refresh
                # to avoid a full filesystem walk after every prompt
                prompt.focus()
            else:
                self._set_command_picker_visible(False)
            self._refresh_attachment_shelf()
            self._refresh_details_panel()
            # while streaming, defer conversation refresh through the debouncer
            # so we don't recompose on every model event
            if state.busy:
                self._schedule_panels_refresh()
            else:
                self._refresh_conversation_view()
            if not state.busy:
                self._persist_history()

        def _handle_turns_change(self, turns: tuple[ConversationTurn, ...]) -> None:
            self._conversation_turns = turns
            self._sync_active_conversation()
            if turns and self._selected_turn_id is None:
                self._selected_turn_id = turns[-1].turn_id
            if self._selected_turn_id not in {turn.turn_id for turn in turns}:
                self._selected_turn_id = turns[-1].turn_id if turns else None
            # batch refreshes while the agent streams events; otherwise refresh now
            if self._state.busy:
                self._schedule_panels_refresh()
            else:
                self._refresh_conversation_view()
                self._refresh_history_list()
                self._refresh_details_panel()

        def _schedule_panels_refresh(self) -> None:
            if self._panels_refresh_pending:
                return
            self._panels_refresh_pending = True
            self.set_timer(0.08, self._flush_panels_refresh)

        def _flush_panels_refresh(self) -> None:
            self._panels_refresh_pending = False
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

        def _handle_changes_change(self, event: dict[str, object] | None) -> None:
            self._changes_summary_event = dict(event) if event is not None else None
            self._refresh_changes_panel()

        def _refresh_changes_panel(self) -> None:
            try:
                panel = self.query_one('#changes-panel', Collapsible)
                body = self.query_one('#changes-body', Static)
                detail_panel = self.query_one('#changes-detail-panel', Collapsible)
                detail_body = self.query_one('#changes-detail-body', Static)
            except Exception:
                return
            has_changes = self._changes_summary_event is not None
            panel.display = has_changes
            detail_panel.display = has_changes
            title = render_changes_panel_title(
                self._changes_summary_event,
                color=True,
            )
            try:
                panel.title = title
            except Exception:
                panel.title = render_changes_panel_title(self._changes_summary_event)
            body.update(
                render_changes_panel_body(
                    self._changes_summary_event,
                    color=True,
                )
            )
            detail_title = render_changes_detail_panel_title(
                self._changes_summary_event,
                color=True,
            )
            try:
                detail_panel.title = detail_title
            except Exception:
                detail_panel.title = render_changes_detail_panel_title(
                    self._changes_summary_event
                )
            detail_body.update(
                render_changes_detail_panel_body(
                    self._changes_summary_event,
                    color=True,
                )
            )

        def _refresh_command_picker(
            self,
            value: str,
            *,
            cursor_position: int | None = None,
        ) -> None:
            option_list = self.query_one('#command-options', OptionList)
            description = self.query_one('#command-description', Static)
            option_list.clear_options()
            self._visible_command_suggestions = []
            self._visible_workspace_path_suggestions = []
            self._active_workspace_reference_query = None
            self._picker_mode = None
            if self._state.busy:
                description.update('')
                self._set_command_picker_visible(False)
                return

            slash_suggestions = filter_slash_command_suggestions(
                value,
                suggestions=self._command_suggestions,
            )
            if slash_suggestions:
                self._visible_command_suggestions = slash_suggestions
                self._picker_mode = 'slash'
                option_list.add_options(
                    Option(suggestion.label, id=suggestion.primary_name)
                    for suggestion in slash_suggestions
                )
                option_list.highlighted = 0
                description.update(render_slash_command_suggestion_detail(slash_suggestions[0]))
                self._set_command_picker_visible(True)
                return

            reference_query = extract_workspace_reference_query(value, cursor_position)
            if reference_query is None:
                description.update('')
                self._set_command_picker_visible(False)
                return
            path_suggestions = filter_workspace_path_suggestions(
                reference_query.query,
                self._workspace_path_suggestions,
            )
            if not path_suggestions:
                description.update('')
                self._set_command_picker_visible(False)
                return
            self._visible_workspace_path_suggestions = path_suggestions
            self._active_workspace_reference_query = reference_query
            self._picker_mode = 'workspace'
            option_list.add_options(
                Option(suggestion.label, id=f'workspace-path-{index}')
                for index, suggestion in enumerate(path_suggestions)
            )
            option_list.highlighted = 0
            description.update(render_workspace_path_suggestion_detail(path_suggestions[0]))
            self._set_command_picker_visible(True)

        def _move_command_selection(self, delta: int) -> None:
            visible_count = len(
                self._visible_workspace_path_suggestions
                if self._picker_mode == 'workspace'
                else self._visible_command_suggestions
            )
            if not visible_count:
                return
            option_list = self.query_one('#command-options', OptionList)
            current = option_list.highlighted
            if current is None:
                current = 0
            next_index = max(0, min(visible_count - 1, current + delta))
            option_list.highlighted = next_index
            option_list.scroll_to_highlight()

        def _should_accept_completion_on_enter(self, value: str) -> bool:
            if self._picker_mode == 'workspace':
                return bool(self._visible_workspace_path_suggestions)
            if not self._visible_command_suggestions:
                return False
            query = extract_slash_command_query(value)
            if query is None:
                return False
            if not query:
                return True
            return find_slash_command(query) is None

        def _apply_highlighted_command(self) -> None:
            if self._picker_mode == 'workspace':
                if not self._visible_workspace_path_suggestions:
                    return
                option_list = self.query_one('#command-options', OptionList)
                highlighted = option_list.highlighted
                if (
                    highlighted is None
                    or highlighted >= len(self._visible_workspace_path_suggestions)
                ):
                    highlighted = 0
                self._apply_workspace_path_suggestion(f'workspace-path-{highlighted}')
                return
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
            prompt = self.query_one('#prompt', PromptInput)
            prompt.value = suggestion.insertion_text
            prompt.cursor_position = len(prompt.value)
            self._refresh_command_picker(prompt.value, cursor_position=prompt.cursor_position)
            prompt.focus()

        def _apply_workspace_path_suggestion(self, option_id: str | None) -> None:
            suggestion = self._workspace_suggestion_for_option_id(option_id)
            reference_query = self._active_workspace_reference_query
            if suggestion is None or reference_query is None:
                return
            prompt = self.query_one('#prompt', PromptInput)
            value = prompt.value
            start = reference_query.start_index
            end = reference_query.end_index
            prompt.value = f'{value[:start]}{suggestion.insertion_text}{value[end:]}'
            prompt.cursor_position = start + len(suggestion.insertion_text)
            self._refresh_command_picker(prompt.value, cursor_position=prompt.cursor_position)
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

        def _workspace_suggestion_for_option_id(
            self,
            option_id: str | None,
        ) -> WorkspacePathSuggestion | None:
            if option_id is None or not option_id.startswith('workspace-path-'):
                return None
            try:
                index = int(option_id.rsplit('-', 1)[1])
            except ValueError:
                return None
            if index < 0 or index >= len(self._visible_workspace_path_suggestions):
                return None
            return self._visible_workspace_path_suggestions[index]

        def _selected_turn(self) -> ConversationTurn | None:
            if not self._conversation_turns:
                return None
            if self._selected_turn_id is None:
                return self._conversation_turns[-1]
            for turn in self._conversation_turns:
                if turn.turn_id == self._selected_turn_id:
                    return turn
            return self._conversation_turns[-1]

        def _refresh_conversation_view(self, *, allow_stick_to_bottom: bool = True) -> None:
            try:
                scroll_container = self.query_one('#conversation-scroll', VerticalScroll)
            except Exception:
                return
            previous_scroll_y = self._conversation_scroll_y(scroll_container)
            was_at_end = self._conversation_at_end()
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
            if should_follow_conversation_bottom(
                allow_stick_to_bottom=allow_stick_to_bottom,
                selected_latest_turn=selected_latest_turn,
                was_at_end=was_at_end,
                following_bottom=self._conversation_follow_bottom,
            ):
                self._schedule_conversation_scroll_to_end(smooth=False)
                return
            self._stop_pending_scroll_to_end()
            self._conversation_follow_bottom = False
            if self._conversation_pinned_scroll_y is None:
                self._conversation_pinned_scroll_y = previous_scroll_y
            self._schedule_conversation_scroll_restore(self._conversation_pinned_scroll_y)

        def _refresh_history_list(self) -> None:
            try:
                option_list = self.query_one('#history-list', OptionList)
            except Exception:
                return
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
            try:
                reuse_button = self.query_one('#reuse-prompt-button', Button)
                retry_button = self.query_one('#retry-turn-button', Button)
                stop_button = self.query_one('#stop-run-button', Button)
                delete_button = self.query_one('#delete-conversation-button', Button)
                details = self.query_one('#details', Static)
            except Exception:
                return
            can_use_turn = turn is not None and not self._state.busy
            reuse_button.disabled = not can_use_turn
            retry_button.disabled = not can_use_turn
            stop_button.disabled = not self._state.busy
            delete_button.disabled = self._state.busy
            details.update(
                render_details_panel(
                    self._state,
                    turn,
                    self._activity_items,
                    hide_gitignored_paths=self._hide_gitignored_workspace_paths,
                    color=True,
                )
            )

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

        def _conversation_scroll_y(self, container=None) -> float:
            if container is None:
                container = self._conversation_scroll_container()
            try:
                return float(getattr(container, 'scroll_y', 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _conversation_at_end(self) -> bool:
            container = self._conversation_scroll_container()
            return conversation_scroll_is_at_end(
                getattr(container, 'scroll_y', 0.0),
                getattr(container, 'max_scroll_y', 0.0),
            )

        def _sync_conversation_follow_to_position(self) -> None:
            self._conversation_follow_bottom = self._conversation_at_end()
            self._conversation_pinned_scroll_y = (
                None
                if self._conversation_follow_bottom
                else self._conversation_scroll_y()
            )

        def _handle_conversation_user_scroll(self, scroll_direction: str) -> None:
            was_following = self._conversation_follow_bottom
            if scroll_direction == 'up':
                self._stop_pending_scroll_to_end()
                self._conversation_follow_bottom = False
            self._cancel_pending_scroll_restore()
            self.call_after_refresh(
                lambda: self._sync_conversation_follow_after_user_scroll(
                    scroll_direction,
                    was_following=was_following,
                )
            )

        def _sync_conversation_follow_after_user_scroll(
            self,
            scroll_direction: str,
            *,
            was_following: bool,
        ) -> None:
            at_end = self._conversation_at_end()
            self._conversation_follow_bottom = conversation_follow_after_user_scroll(
                scroll_direction=scroll_direction,
                was_following=was_following,
                at_end=at_end,
            )
            self._conversation_pinned_scroll_y = (
                None
                if self._conversation_follow_bottom
                else self._conversation_scroll_y()
            )
            if self._conversation_follow_bottom:
                self._schedule_conversation_scroll_to_end(smooth=False)
            elif self._conversation_pinned_scroll_y is not None:
                self._schedule_conversation_scroll_restore(
                    self._conversation_pinned_scroll_y
                )

        def _stop_pending_scroll_to_end(self) -> None:
            self._scroll_to_end_pending = False
            self._scroll_to_end_token += 1
            self._scroll_to_end_smooth = False

        def _cancel_pending_scroll_restore(self) -> None:
            self._restore_scroll_pending = False
            self._restore_scroll_y = None
            self._restore_scroll_token += 1

        def _reset_conversation_scroll_state(self) -> None:
            self._conversation_follow_bottom = True
            self._conversation_pinned_scroll_y = None
            self._restore_scroll_pending = False
            self._restore_scroll_y = None
            self._restore_scroll_token += 1
            self._scroll_to_end_pending = False
            self._scroll_to_end_token += 1
            self._scroll_to_end_smooth = False

        def _scroll_conversation_by_pages(self, pages: int) -> None:
            container = self._conversation_scroll_container()
            size = getattr(container, 'size', None)
            page_height = max(int(getattr(size, 'height', 20)) - 2, 1)
            current_y = float(getattr(container, 'scroll_y', 0.0))
            max_y = float(getattr(container, 'max_scroll_y', 0.0))
            target_y = max(0.0, min(max_y, current_y + (pages * page_height)))
            self._scroll_conversation_to_y(target_y)

        def _scroll_conversation_to_y(self, y: float, *, smooth: bool = False) -> None:
            self._stop_pending_scroll_to_end()
            self._cancel_pending_scroll_restore()
            container = self._conversation_scroll_container()
            try:
                container.scroll_to(y=y, animate=smooth, immediate=not smooth)
            except TypeError:
                try:
                    container.scroll_to(y=y, animate=smooth)
                except TypeError:
                    try:
                        container.scroll_to(y=y)
                    except Exception:
                        return
                except Exception:
                    return
            except Exception:
                return
            self._sync_conversation_follow_to_position()
            if self._conversation_follow_bottom:
                self._schedule_conversation_scroll_to_end(smooth=False)

        def _scroll_conversation_to_current_end(self, *, smooth: bool = False) -> None:
            try:
                container = self.query_one('#conversation-scroll', VerticalScroll)
            except Exception:
                return
            try:
                container.scroll_end(animate=smooth, immediate=not smooth)
            except TypeError:
                try:
                    container.scroll_end(animate=smooth)
                except TypeError:
                    try:
                        container.scroll_end()
                    except AttributeError:
                        return
                    except Exception:
                        return
                except AttributeError:
                    return
                except Exception:
                    return
            except AttributeError:
                return
            except Exception:
                return
            self._conversation_follow_bottom = True
            self._conversation_pinned_scroll_y = None

        def _scroll_conversation_to_end(self, token: int) -> None:
            if not self._scroll_to_end_pending or token != self._scroll_to_end_token:
                return
            self._finish_scroll_conversation_to_end(token, 0, -1.0, 0)

        # safety cap and interval for scroll-to-end polling: keep snapping to
        # the bottom as long as content height is still growing (markdown
        # widgets lay out progressively after a conversation switch). While a
        # run is active and follow mode is linked, keep a slower heartbeat so
        # delayed layout growth cannot leave the viewport above the live tail.
        _SCROLL_TO_END_MAX_ATTEMPTS = 80
        _SCROLL_TO_END_INTERVAL = 0.01
        _SCROLL_TO_END_FOLLOW_INTERVAL = 0.03
        _SCROLL_TO_END_STABLE_STREAK_REQUIRED = 3
        _SCROLL_RESTORE_MAX_ATTEMPTS = 80
        _SCROLL_RESTORE_INTERVAL = 0.01
        _SCROLL_RESTORE_FOLLOW_INTERVAL = 0.03
        _SCROLL_RESTORE_STABLE_STREAK_REQUIRED = 3

        def _finish_scroll_conversation_to_end(
            self,
            token: int,
            attempt: int,
            previous_max_y: float,
            stable_streak: int,
        ) -> None:
            if not self._scroll_to_end_pending or token != self._scroll_to_end_token:
                return
            self._scroll_conversation_to_current_end(
                smooth=self._scroll_to_end_smooth and attempt == 0
            )
            try:
                container = self._conversation_scroll_container()
            except Exception:
                self._scroll_to_end_pending = False
                self._scroll_to_end_smooth = False
                return
            scroll_y = float(getattr(container, 'scroll_y', 0.0))
            max_y = float(getattr(container, 'max_scroll_y', 0.0))
            at_end = max_y > 0 and (max_y - scroll_y) <= 1.0
            if max_y == previous_max_y:
                stable_streak += 1
            else:
                stable_streak = 0
            if at_end and stable_streak >= self._SCROLL_TO_END_STABLE_STREAK_REQUIRED:
                if self._state.busy and self._conversation_follow_bottom:
                    self.set_timer(
                        self._SCROLL_TO_END_FOLLOW_INTERVAL,
                        lambda: self._finish_scroll_conversation_to_end(
                            token, 0, max_y, stable_streak,
                        ),
                    )
                    return
                self._scroll_to_end_pending = False
                self._scroll_to_end_smooth = False
                return
            should_keep_following = self._state.busy and self._conversation_follow_bottom
            if attempt < self._SCROLL_TO_END_MAX_ATTEMPTS or should_keep_following:
                next_attempt = (
                    attempt + 1
                    if attempt < self._SCROLL_TO_END_MAX_ATTEMPTS
                    else 0
                )
                self.set_timer(
                    self._SCROLL_TO_END_INTERVAL,
                    lambda: self._finish_scroll_conversation_to_end(
                        token, next_attempt, max_y, stable_streak,
                    ),
                )
                return
            self._scroll_to_end_pending = False
            self._scroll_to_end_smooth = False

        def _schedule_conversation_scroll_to_end(self, *, smooth: bool = False) -> None:
            self._cancel_pending_scroll_restore()
            self._conversation_follow_bottom = True
            self._conversation_pinned_scroll_y = None
            if self._scroll_to_end_pending:
                self._scroll_to_end_smooth = smooth
                return
            self._scroll_to_end_pending = True
            self._scroll_to_end_token += 1
            self._scroll_to_end_smooth = smooth
            token = self._scroll_to_end_token
            self.call_after_refresh(lambda: self._scroll_conversation_to_end(token))

        def _schedule_conversation_scroll_restore(self, scroll_y: float) -> None:
            if self._scroll_to_end_pending:
                return
            self._restore_scroll_y = scroll_y
            if self._restore_scroll_pending:
                return
            self._restore_scroll_pending = True
            self._restore_scroll_token += 1
            token = self._restore_scroll_token
            self.call_after_refresh(
                lambda: self._restore_conversation_scroll(token, 0, -1.0, 0)
            )

        def _restore_conversation_scroll(
            self,
            token: int,
            attempt: int,
            previous_max_y: float,
            stable_streak: int,
        ) -> None:
            if (
                not self._restore_scroll_pending
                or token != self._restore_scroll_token
                or self._scroll_to_end_pending
                or self._conversation_follow_bottom
            ):
                return
            scroll_y = self._restore_scroll_y
            if scroll_y is None:
                self._restore_scroll_pending = False
                return
            try:
                container = self.query_one('#conversation-scroll', VerticalScroll)
            except Exception:
                self._restore_scroll_pending = False
                return
            max_y = float(getattr(container, 'max_scroll_y', 0.0))
            target_y = max(0.0, min(max_y, scroll_y))
            try:
                container.scroll_to(y=target_y, animate=False, immediate=True)
            except TypeError:
                try:
                    container.scroll_to(y=target_y, animate=False)
                except Exception:
                    self._restore_scroll_pending = False
                    return
            except Exception:
                self._restore_scroll_pending = False
                return
            self._conversation_follow_bottom = False
            current_y = self._conversation_scroll_y(container)
            max_y = float(getattr(container, 'max_scroll_y', 0.0))
            at_target = abs(current_y - target_y) <= 1.0
            if max_y == previous_max_y:
                stable_streak += 1
            else:
                stable_streak = 0
            if at_target and stable_streak >= self._SCROLL_RESTORE_STABLE_STREAK_REQUIRED:
                if self._state.busy and not self._conversation_follow_bottom:
                    self.set_timer(
                        self._SCROLL_RESTORE_FOLLOW_INTERVAL,
                        lambda: self._restore_conversation_scroll(
                            token, 0, max_y, stable_streak,
                        ),
                    )
                    return
                self._restore_scroll_pending = False
                self._restore_scroll_y = None
                return
            should_keep_restoring = self._state.busy and not self._conversation_follow_bottom
            if attempt < self._SCROLL_RESTORE_MAX_ATTEMPTS or should_keep_restoring:
                next_attempt = (
                    attempt + 1
                    if attempt < self._SCROLL_RESTORE_MAX_ATTEMPTS
                    else 0
                )
                self.set_timer(
                    self._SCROLL_RESTORE_INTERVAL,
                    lambda: self._restore_conversation_scroll(
                        token, next_attempt, max_y, stable_streak,
                    ),
                )
                return
            self._restore_scroll_pending = False
            self._restore_scroll_y = None

        def _tick_spinner(self) -> None:
            if not self._state.busy:
                return
            self._spinner_index = (self._spinner_index + 1) % 4
            self._refresh_details_panel()

        def _submit_prompt(
            self,
            prompt: str,
            *,
            runtime_prompt: str | None = None,
            prompt_blocks: tuple[dict[str, object], ...] = (),
        ) -> None:
            if self._state.busy:
                return
            effective_prompt = runtime_prompt or prompt
            self._cancel_requested.clear()
            self._worker_event_counts.clear()
            self._debug_log(
                'submit_prompt',
                turns=len(self._conversation_turns),
                session=bool(self._active_session_id),
                attachments=len(prompt_blocks),
            )
            self._schedule_conversation_scroll_to_end(smooth=True)
            self._bridge.begin_prompt(prompt, session_id=self._active_session_id)
            self._active_worker = self._run_prompt(effective_prompt, prompt_blocks)

        @work(thread=True, exclusive=True)
        def _run_prompt(
            self,
            prompt: str,
            prompt_blocks: tuple[dict[str, object], ...] = (),
        ) -> None:
            self._debug_log(
                'worker_start',
                session=bool(self._active_session_id),
                attachments=len(prompt_blocks),
            )
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
                        prompt_blocks=prompt_blocks,
                        event_handler=self._handle_event_from_worker,
                    )
                else:
                    result = self._agent.resume(
                        prompt,
                        stored_session,
                        prompt_blocks=prompt_blocks,
                        event_handler=self._handle_event_from_worker,
                    )
                if self._cancel_requested.is_set():
                    raise _TuiRunCancelled()
            except _TuiRunCancelled:
                self._debug_log('worker_cancelled')
                self.call_from_thread(self._finish_cancelled_prompt)
                return
            except BaseException as exc:
                self._debug_log('worker_failed', error=repr(exc))
                self.call_from_thread(self._finish_failed_prompt, exc)
                return
            self._debug_log(
                'worker_finished',
                turns=result.turns,
                tools=result.tool_calls,
                stop=result.stop_reason,
            )
            self.call_from_thread(self._finish_prompt, result)

        def _handle_event_from_worker(self, event: dict[str, object]) -> None:
            if self._cancel_requested.is_set():
                raise _TuiRunCancelled()
            event_type = str(event.get('type') or 'unknown')
            count = self._worker_event_counts.get(event_type, 0) + 1
            self._worker_event_counts[event_type] = count
            if event_type != 'content_delta' or count in {1, 25, 100} or count % 250 == 0:
                self._debug_log('worker_event', type=event_type, count=count)
            self.call_from_thread(self._bridge.handle_event, dict(event))

        def _finish_cancelled_prompt(self) -> None:
            self._debug_log('finish_cancelled_start')
            self._active_worker = None
            self._cancel_requested.clear()
            self._bridge.cancel('Stopped by user')
            self._persist_history()
            self._debug_log('finish_cancelled_done')

        def _finish_failed_prompt(self, error: BaseException) -> None:
            self._debug_log('finish_failed_start', error=repr(error))
            self._active_worker = None
            self._cancel_requested.clear()
            self._bridge.fail(error)
            self._persist_history()
            self._debug_log('finish_failed_done')

        def _finish_prompt(self, result: AgentRunResult) -> None:
            self._debug_log(
                'finish_prompt_start',
                turns=result.turns,
                tools=result.tool_calls,
                stop=result.stop_reason,
            )
            self._active_worker = None
            if self._cancel_requested.is_set():
                self._finish_cancelled_prompt()
                return
            self._active_session_id = result.session_id or self._active_session_id
            self._bridge.complete(result)
            self._debug_log(
                'finish_prompt_bridge_complete',
                busy=self._state.busy,
                turns=len(self._conversation_turns),
            )
            self._persist_history()
            self._debug_log('finish_prompt_done')

        def _persist_history(self) -> None:
            # mark dirty and schedule a single async write; coalesces bursts
            # of save calls (every turn event would otherwise hit the disk
            # on the UI thread)
            self._persist_history_dirty = True
            if self._persist_history_scheduled:
                return
            self._persist_history_scheduled = True
            self.set_timer(0.3, self._flush_persist_history)

        def _flush_persist_history(self) -> None:
            self._persist_history_scheduled = False
            if not self._persist_history_dirty:
                return
            self._persist_history_dirty = False
            # snapshot on the UI thread (conversations are mutated from here)
            snapshot = tuple(self._conversations)
            active_id = self._active_conversation_id
            self._write_history_snapshot(snapshot, active_id)

        @work(thread=True, exclusive=True, group='persist-history')
        def _write_history_snapshot(
            self,
            conversations: tuple[ConversationThread, ...],
            active_conversation_id: str | None,
        ) -> None:
            try:
                if not any(
                    conversation.turns or conversation.session_id
                    for conversation in conversations
                ):
                    try:
                        self._history_store.workspace_path(self._workspace).unlink()
                    except FileNotFoundError:
                        pass
                    return
                self._history_store.save_workspace_conversations(
                    self._workspace,
                    conversations,
                    active_conversation_id=active_conversation_id,
                )
            except OSError:
                self._debug_log('persist_history_failed')
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
