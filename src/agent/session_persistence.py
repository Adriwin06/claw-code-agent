from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.agent.agent_session import AgentSessionState
from src.agent.agent_types import AgentRunResult, ModelConfig, AgentRuntimeConfig
from src.session.session_store import (
    StoredAgentSession,
    load_agent_session,
    save_agent_session,
    serialize_model_config,
    serialize_runtime_config,
)


def persist_agent_run(
    *,
    session: AgentSessionState,
    result: AgentRunResult,
    model_config: ModelConfig,
    runtime_config: AgentRuntimeConfig,
    plugin_state: dict[str, Any],
) -> tuple[AgentRunResult, str]:
    if result.session_id is None:
        return result, result.session_path or ''

    previous_turns = 0
    previous_tool_calls = 0
    previous_budget_state: dict[str, object] = {}
    existing_path = runtime_config.session_directory / f'{result.session_id}.json'
    if existing_path.exists():
        try:
            previous = load_agent_session(
                result.session_id,
                directory=runtime_config.session_directory,
            )
        except OSError:
            previous = None
        if previous is not None:
            previous_turns = previous.turns
            previous_tool_calls = previous.tool_calls
            if isinstance(previous.budget_state, dict):
                previous_budget_state = dict(previous.budget_state)
    budget_state = {
        'model_calls': int(previous_budget_state.get('model_calls', 0))
        + max(result.turns, 0),
        'session_turns': previous_turns + result.turns,
        'tool_calls': previous_tool_calls + result.tool_calls,
        'delegated_tasks': sum(
            1
            for entry in result.file_history
            if entry.get('action') in ('delegate_agent', 'Agent')
        ),
    }
    stored = StoredAgentSession(
        session_id=result.session_id,
        model_config=serialize_model_config(model_config),
        runtime_config=serialize_runtime_config(runtime_config),
        system_prompt_parts=session.system_prompt_parts,
        user_context=dict(session.user_context),
        system_context=dict(session.system_context),
        messages=session.transcript(),
        turns=previous_turns + result.turns,
        tool_calls=previous_tool_calls + result.tool_calls,
        usage=result.usage.to_dict(),
        total_cost_usd=result.total_cost_usd,
        file_history=result.file_history,
        budget_state=budget_state,
        plugin_state=plugin_state,
        scratchpad_directory=result.scratchpad_directory,
    )
    path = save_agent_session(
        stored,
        directory=runtime_config.session_directory,
    )
    return (
        replace(
            result,
            session_path=str(path),
            transcript=session.transcript(),
        ),
        str(path),
    )
