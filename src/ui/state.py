from __future__ import annotations

from dataclasses import dataclass

from src.agent.agent_runtime import LocalCodingAgent
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
