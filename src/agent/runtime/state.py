from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.models.session import AgentSessionState
from src.agent.models.types import AgentRunResult, UsageStats


JSONDict = dict[str, object]


@dataclass
class PromptRunState:
    session_id: str
    scratchpad_directory: Path | None
    effective_prompt: str
    session: AgentSessionState
    starting_session_turns: int
    tool_calls: int
    total_usage: UsageStats
    total_cost_usd: float
    file_history: list[JSONDict]
    stream_events: Any
    delegated_tasks: int
    model_calls: int
    assistant_response_segments: list[str] = field(default_factory=list)
    last_content: str = ''
    turn_index: int = 0
    consecutive_pending_work_continuations: int = 0


@dataclass(frozen=True)
class BudgetDecision:
    exceeded: bool
    reason: str | None = None


@dataclass(frozen=True)
class PromptPreflightResult:
    usage_increment: UsageStats = field(default_factory=UsageStats)
    model_calls_increment: int = 0
    stop_reason: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class TurnLoopDirective:
    continue_loop: bool = False
    result: AgentRunResult | None = None


def build_run_result(
    state: PromptRunState,
    *,
    final_output: str,
    turns: int,
    stop_reason: str | None,
) -> AgentRunResult:
    return AgentRunResult(
        final_output=final_output,
        turns=turns,
        tool_calls=state.tool_calls,
        transcript=state.session.transcript(),
        events=tuple(state.stream_events),
        usage=state.total_usage,
        total_cost_usd=state.total_cost_usd,
        stop_reason=stop_reason,
        file_history=tuple(state.file_history),
        session_id=state.session_id,
        scratchpad_directory=(
            str(state.scratchpad_directory)
            if state.scratchpad_directory is not None
            else None
        ),
    )
