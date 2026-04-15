from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .agent_slash_commands import find_slash_command, get_slash_command_specs
from .agent_runtime import LocalCodingAgent
from .agent_types import AgentRunResult
from .session_store import load_agent_session


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


@dataclass
class AgentTuiState:
    workspace: str
    model: str
    permissions: str
    status: str = 'Idle'
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
    busy: bool = False

    @classmethod
    def from_agent(cls, agent: LocalCodingAgent) -> 'AgentTuiState':
        return cls(
            workspace=str(agent.runtime_config.cwd),
            model=agent.model_config.model,
            permissions=_render_permissions(agent),
        )

    def render(self) -> str:
        lines = [
            'Session',
            '',
            f'status={self.status}',
            f'workspace={self.workspace}',
            f'model={self.model}',
            f'permissions={self.permissions}',
            f'prompt_count={self.prompt_count}',
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
    ) -> None:
        self.state = state
        self._emit_data = emit_data
        self._on_state_change = on_state_change
        self._assistant_open = False
        self._assistant_ends_with_newline = True
        self._tool_stream_key: tuple[str | None, str] | None = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True
        self._announced_tool_plans: set[str] = set()
        self._streamed_assistant_output = False

    def begin_prompt(self, prompt: str, *, session_id: str | None = None) -> None:
        self._close_open_blocks()
        self._announced_tool_plans.clear()
        self.state.busy = True
        self.state.status = 'Running'
        self.state.prompt_count += 1
        if session_id:
            self.state.session_id = session_id
        self._streamed_assistant_output = False
        self._emit_data(f'[user] {prompt}\n')
        self._publish_state()

    def handle_event(self, event: dict[str, object]) -> None:
        event_type = event.get('type')
        if event_type == 'message_start':
            self.state.status = 'Querying model'
            self._write_status('[thinking] model call started')
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
                self._write_status('[thinking] dispatching tool call')
            elif finish_reason == 'length':
                self.state.status = 'Continuation requested'
                self._write_status('[thinking] model output hit the length limit')
            else:
                self._close_open_blocks()
            self._publish_state()
            return
        if event_type == 'tool_start':
            tool_name = event.get('tool_name')
            if isinstance(tool_name, str) and tool_name:
                self.state.last_tool = tool_name
                self.state.status = f'Tool: {tool_name}'
            self._write_status(self._render_tool_start(event))
            self._publish_state()
            return
        if event_type == 'tool_delta':
            self._render_tool_delta(event)
            return
        if event_type == 'tool_result':
            self.state.status = 'Processing tool result'
            self._write_status(self._render_tool_result(event))
            self._publish_state()
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
            return
        if event_type == 'continuation_request':
            self.state.status = 'Continuation requested'
            self._write_status('[thinking] requesting continuation')
            self._publish_state()
            return
        if event_type == 'tool_permission_denial':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self.state.status = 'Permission denied'
            self._write_status(f'[tool] permission denied: {reason}')
            self._publish_state()
            return
        if event_type == 'task_budget_exceeded':
            reason = _preview_value(event.get('reason'), max_chars=220)
            self.state.status = 'Budget exceeded'
            self._write_status(f'[budget] {reason}')
            self._publish_state()
            return
        if event_type == 'plugin_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self.state.status = 'Plugin blocked tool'
            self._write_status(f'[plugin] blocked tool: {message}')
            self._publish_state()
            return
        if event_type == 'hook_policy_tool_block':
            message = _preview_value(event.get('message'), max_chars=220)
            self.state.status = 'Policy blocked tool'
            self._write_status(f'[policy] blocked tool: {message}')
            self._publish_state()
            return

    def complete(self, result: AgentRunResult) -> None:
        self._close_open_blocks()
        if not self._streamed_assistant_output and result.final_output:
            self._emit_data(f'[assistant] {result.final_output}\n')
        self.state.busy = False
        self.state.status = 'Ready'
        self.state.session_id = result.session_id or self.state.session_id
        self.state.last_turns = result.turns
        self.state.last_tool_calls = result.tool_calls
        self.state.total_tokens = result.usage.total_tokens
        self.state.input_tokens = result.usage.input_tokens
        self.state.output_tokens = result.usage.output_tokens
        self.state.total_cost_usd = result.total_cost_usd
        self.state.last_stop_reason = result.stop_reason
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
        self._publish_state()

    def fail(self, error: BaseException) -> None:
        self._close_open_blocks()
        self.state.busy = False
        self.state.status = 'Error'
        self.state.last_stop_reason = error.__class__.__name__
        self._emit_data(f'[error] {error}\n')
        self._publish_state()

    def reset_display(self) -> None:
        self._assistant_open = False
        self._assistant_ends_with_newline = True
        self._tool_stream_key = None
        self._tool_stream_open = False
        self._tool_stream_ends_with_newline = True
        self._streamed_assistant_output = False

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

    def _render_tool_plan(self, event: dict[str, object]) -> None:
        tool_name = event.get('tool_name')
        if not isinstance(tool_name, str) or not tool_name:
            return
        key = str(event.get('tool_call_id') or f"index:{event.get('tool_call_index', 0)}")
        if key in self._announced_tool_plans:
            return
        self._announced_tool_plans.add(key)
        self._write_status(f'[thinking] planning tool call: {tool_name}')

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
        from textual.widgets import Footer, Header, Input, Log, OptionList, Static
        from textual.widgets.option_list import Option
    except ImportError as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(
            'Textual is not installed. Install it with `pip install -e .[tui]` '
            'or `pip install textual` before using `agent-tui`.'
        ) from exc

    class AgentTuiApp(App[None]):
        TITLE = 'Claw Code Agent'
        CSS = """
        Screen {
            layout: vertical;
        }

        #body {
            height: 1fr;
        }

        #chat-log {
            width: 1fr;
            border: round $accent;
        }

        #sidebar {
            width: 36;
            min-width: 28;
        }

        #summary {
            height: 1fr;
            border: round $primary;
            padding: 1 2;
        }

        #help {
            height: auto;
            border: round $surface;
            padding: 1 2;
            margin-top: 1;
        }

        #prompt {
            margin: 1 1 0 1;
        }

        #command-picker {
            height: 12;
            margin: 0 1;
        }

        #command-options {
            width: 36;
            min-width: 28;
            border: round $accent;
        }

        #command-description {
            width: 1fr;
            border: round $surface;
            padding: 1 2;
        }
        """
        BINDINGS = [
            ('q', 'quit', 'Quit'),
            ('ctrl+l', 'clear_log', 'Clear Log'),
            ('ctrl+j', 'focus_prompt', 'Focus Prompt'),
            ('ctrl+r', 'refresh_sidebar', 'Refresh'),
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
            if resumed_session_id:
                self._state.session_id = resumed_session_id
                self._state.status = 'Resumed'
            self._bridge = AgentTuiEventBridge(
                self._state,
                emit_data=self._append_log,
                on_state_change=self._update_sidebar,
            )

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id='body'):
                yield Log(id='chat-log', auto_scroll=True, highlight=False)
                with Vertical(id='sidebar'):
                    yield Static(id='summary')
                    yield Static(id='help')
            with Horizontal(id='command-picker'):
                yield OptionList(id='command-options')
                yield Static(id='command-description')
            yield Input(
                placeholder='Type a prompt or slash command and press Enter',
                id='prompt',
            )
            yield Footer()

        def on_mount(self) -> None:
            self.query_one('#help', Static).update(
                '\n'.join(
                    [
                        'Controls',
                        '',
                        'Enter: submit prompt',
                        'q: quit',
                        'ctrl+l: clear log',
                        'ctrl+j: focus prompt',
                        'ctrl+r: refresh sidebar',
                        'when typing /: Up/Down browse commands',
                        'Tab or click: insert selected command',
                        '',
                        'Slash commands work here too.',
                    ]
                )
            )
            self._append_log('[system] Textual UI ready.\n')
            if self._active_session_id:
                self._append_log(f'[system] resuming_session_id={self._active_session_id}\n')
            self._set_command_picker_visible(False)
            self._update_sidebar(self._state)
            self.action_focus_prompt()
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
            self._submit_prompt(prompt)

        def on_key(self, event: events.Key) -> None:
            prompt = self.query_one('#prompt', Input)
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

        def on_option_list_option_highlighted(
            self,
            event: OptionList.OptionHighlighted,
        ) -> None:
            if event.option_list.id != 'command-options':
                return
            suggestion = self._suggestion_for_option_id(event.option_id)
            if suggestion is not None:
                self.query_one('#command-description', Static).update(
                    render_slash_command_suggestion_detail(suggestion)
                )

        def on_option_list_option_selected(
            self,
            event: OptionList.OptionSelected,
        ) -> None:
            if event.option_list.id != 'command-options':
                return
            self._apply_command_suggestion(event.option_id)
            event.stop()

        def action_clear_log(self) -> None:
            self.query_one('#chat-log', Log).clear()
            self._bridge.reset_display()
            self._append_log('[system] Log cleared.\n')

        def action_focus_prompt(self) -> None:
            self.query_one('#prompt', Input).focus()

        def action_refresh_sidebar(self) -> None:
            self._update_sidebar(self._state)

        def _append_log(self, text: str) -> None:
            self.query_one('#chat-log', Log).write(text)

        @property
        def _command_picker_visible(self) -> bool:
            return bool(self.query_one('#command-picker', Horizontal).display)

        def _set_command_picker_visible(self, visible: bool) -> None:
            self.query_one('#command-picker', Horizontal).display = visible

        def _update_sidebar(self, state: AgentTuiState) -> None:
            self.query_one('#summary', Static).update(state.render())
            workspace_name = Path(state.workspace).name or state.workspace
            self.sub_title = f'{workspace_name} | {state.status}'
            prompt = self.query_one('#prompt', Input)
            prompt.disabled = state.busy
            if not state.busy:
                prompt.focus()
            else:
                self._set_command_picker_visible(False)

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

        def _submit_prompt(self, prompt: str) -> None:
            if self._state.busy:
                self._append_log('[system] Agent is still processing the previous prompt.\n')
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
