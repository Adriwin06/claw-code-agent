from __future__ import annotations

from dataclasses import dataclass

from src.agent.runtime.agent import LocalCodingAgent
from src.session.session_store import StoredAgentSession, usage_from_payload


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
    max_turns: int | None = 0
    command_timeout_seconds: float = 0.0
    conversation_turns: int = 0
    activity_events: int = 0
    workspace_change_events: int = 0
    workspace_changed_files: int = 0
    workspace_added_lines: int = 0
    workspace_removed_lines: int = 0
    search_enabled: bool = True
    search_context_size: str = 'medium'
    search_default_max_results: int = 5
    search_provider_count: int = 0
    search_manifest_count: int = 0
    search_active_provider: str = 'none'
    busy: bool = False

    @classmethod
    def from_agent(cls, agent: LocalCodingAgent) -> 'AgentTuiState':
        state = cls(
            workspace=str(agent.runtime_config.cwd),
            model=agent.model_config.model,
            permissions=_render_permissions(agent),
            streaming_enabled=agent.runtime_config.stream_model_responses,
            max_turns=agent.runtime_config.max_turns,
            command_timeout_seconds=agent.runtime_config.command_timeout_seconds,
        )
        state.refresh_from_agent(agent)
        return state

    def refresh_from_agent(self, agent: LocalCodingAgent) -> None:
        self.workspace = str(agent.runtime_config.cwd)
        self.model = agent.model_config.model
        self.permissions = _render_permissions(agent)
        self.streaming_enabled = agent.runtime_config.stream_model_responses
        self.max_turns = agent.runtime_config.max_turns
        self.command_timeout_seconds = agent.runtime_config.command_timeout_seconds

        search_runtime = agent.search_runtime
        if search_runtime is None:
            self.search_enabled = False
            self.search_context_size = 'medium'
            self.search_default_max_results = 5
            self.search_provider_count = 0
            self.search_manifest_count = 0
            self.search_active_provider = 'none'
            return

        self.search_enabled = search_runtime.web_search_enabled
        self.search_context_size = search_runtime.web_search_context_size
        self.search_default_max_results = search_runtime.default_max_results
        self.search_provider_count = len(search_runtime.providers)
        self.search_manifest_count = len(search_runtime.manifests)
        active_provider = search_runtime.current_provider()
        if active_provider is None:
            self.search_active_provider = 'none'
        else:
            self.search_active_provider = f'{active_provider.name} ({active_provider.provider})'

    def render(self) -> str:
        max_turns_label = 'unlimited' if self.max_turns is None else str(self.max_turns)
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
            f'max_turns={max_turns_label}',
            f'command_timeout_seconds={self.command_timeout_seconds:.1f}',
            f'search_enabled={self.search_enabled}',
            f'search_context_size={self.search_context_size}',
            f'search_default_max_results={self.search_default_max_results}',
            f'search_provider_count={self.search_provider_count}',
            f'search_manifest_count={self.search_manifest_count}',
            f'search_active_provider={self.search_active_provider}',
            f'prompt_count={self.prompt_count}',
            f'conversation_turns={self.conversation_turns}',
            f'activity_events={self.activity_events}',
            f'workspace_change_events={self.workspace_change_events}',
            f'workspace_changed_files={self.workspace_changed_files}',
            f'workspace_added_lines={self.workspace_added_lines}',
            f'workspace_removed_lines={self.workspace_removed_lines}',
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
