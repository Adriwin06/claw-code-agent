from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from src.features.system.account_runtime import AccountRuntime
from src.agent.agent_manager import AgentManager
from src.agent.agent_context import clear_context_caches
from src.agent.agent_context import render_context_report as render_agent_context_report
from src.agent.agent_context_usage import collect_context_usage, format_context_usage
from src.session.compact import compact_conversation
from src.features.collaboration.ask_user_runtime import AskUserRuntime
from src.agent.agent_registry import (
    load_agent_registry,
    render_agent_detail,
    render_agents_report,
)
from src.features.system.config_runtime import ConfigRuntime
from src.features.system.hook_policy import HookPolicyRuntime
from src.features.integration.lsp_runtime import LSPRuntime
from src.features.integration.mcp_runtime import MCPRuntime
from src.agent.agent_prompting import (
    build_prompt_context,
    build_system_prompt_parts,
    render_system_prompt,
)
from src.agent.model_turn_runner import (
    normalize_finish_reason,
    query_model_turn,
)
from src.agent.prompt_pressure import (
    build_prompt_length_error,
    can_auto_compact_with_summary,
    check_token_budget,
    compact_session_pass,
    preflight_prompt_length,
    reactive_compact_session,
    reduce_context_pressure,
    snip_session_pass,
)
from src.agent.run_state import (
    BudgetDecision,
    PromptPreflightResult,
    PromptRunState,
    TurnLoopDirective,
    build_run_result,
)
from src.agent.delegate_orchestrator import (
    delegated_task_units,
    execute_delegate_agent,
)
from src.agent.tool_call_runner import (
    ToolCallExecutionHooks,
    execute_runtime_tool_call,
)
from src.agent.agent_session import AgentSessionState
from src.agent.agent_slash_commands import preprocess_slash_command
from src.agent.agent_tools import (
    AgentTool,
    build_tool_context,
    default_tool_registry,
)
from src.agent.agent_types import (
    AgentRunResult,
    AgentRuntimeConfig,
    AssistantTurn,
    BudgetConfig,
    ModelConfig,
    StreamEvent,
    ToolCall,
    ToolExecutionResult,
    UsageStats,
)
from src.llm import build_llm_client, resolve_llm_backend
from src.llm.parsers import LLMBackendError
from src.features.orchestration.plan_runtime import PlanRuntime
from src.features.integration.plugin_runtime import PluginRuntime
from src.features.integration.remote_runtime import RemoteRuntime
from src.features.integration.remote_trigger_runtime import RemoteTriggerRuntime
from src.features.integration.search_runtime import SearchRuntime
from src.features.orchestration.task_runtime import TaskRuntime
from src.features.collaboration.team_runtime import TeamRuntime
from src.features.system.tokenizer_runtime import describe_token_counter
from src.features.orchestration.workflow_runtime import WorkflowRuntime
from src.features.orchestration.worktree_runtime import WorktreeRuntime
from src.agent.runtime_dependencies import AgentRuntimeDependencies
from src.agent.session_persistence import persist_agent_run
from src.session.session_store import (
    StoredAgentSession,
    load_agent_session,
    usage_from_payload,
)
from src.core.governance.token_budget import calculate_token_budget, format_token_budget
from src.agent.builtin_agents import AgentDefinition
from src.session.microcompact import microcompact_messages as _microcompact_messages


RuntimeEventHandler = Callable[[dict[str, object]], None]

_COMPACT_OLLAMA_TOOL_SCHEMA_CHAR_LIMIT = 8_000
_COMPACT_OLLAMA_TOOL_NAMES: tuple[str, ...] = (
    'list_available_tools',
    'tool_search',
    'list_dir',
    'read_file',
    'write_file',
    'edit_file',
    'glob_search',
    'grep_search',
    'bash',
    'LSP',
    'web_fetch',
    'Agent',
    'delegate_agent',
    'Skill',
)


class _RuntimeEventRecorder:
    def __init__(self, handler: RuntimeEventHandler | None = None) -> None:
        self._events: list[dict[str, object]] = []
        self._handler = handler

    def append(self, event: dict[str, object] | StreamEvent) -> None:
        if isinstance(event, StreamEvent):
            payload = event.to_dict()
        else:
            payload = dict(event)
        self._events.append(payload)
        if self._handler is not None:
            self._handler(payload)

    def extend(self, events: Iterable[dict[str, object] | StreamEvent]) -> None:
        for event in events:
            self.append(event)

    def __iter__(self):
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


@dataclass
class LocalCodingAgent:
    model_config: ModelConfig
    runtime_config: AgentRuntimeConfig
    llm_backend: str = 'litellm'
    custom_system_prompt: str | None = None
    append_system_prompt: str | None = None
    override_system_prompt: str | None = None
    tool_registry: dict[str, AgentTool] | None = None
    agent_manager: AgentManager | None = None
    parent_agent_id: str | None = None
    managed_group_id: str | None = None
    managed_child_index: int | None = None
    managed_label: str | None = None
    dependencies: AgentRuntimeDependencies | None = None
    plugin_runtime: PluginRuntime | None = None
    hook_policy_runtime: HookPolicyRuntime | None = None
    mcp_runtime: MCPRuntime | None = None
    remote_runtime: RemoteRuntime | None = None
    remote_trigger_runtime: RemoteTriggerRuntime | None = None
    search_runtime: SearchRuntime | None = None
    account_runtime: AccountRuntime | None = None
    ask_user_runtime: AskUserRuntime | None = None
    config_runtime: ConfigRuntime | None = None
    lsp_runtime: LSPRuntime | None = None
    plan_runtime: PlanRuntime | None = None
    task_runtime: TaskRuntime | None = None
    team_runtime: TeamRuntime | None = None
    workflow_runtime: WorkflowRuntime | None = None
    worktree_runtime: WorktreeRuntime | None = None
    last_session: AgentSessionState | None = field(default=None, init=False, repr=False)
    last_run_result: AgentRunResult | None = field(default=None, init=False, repr=False)
    cumulative_usage: UsageStats = field(default_factory=UsageStats, init=False, repr=False)
    cumulative_cost_usd: float = field(default=0.0, init=False, repr=False)
    _compact_consecutive_failures: int = field(default=0, init=False, repr=False)
    active_session_id: str | None = field(default=None, init=False, repr=False)
    last_session_path: str | None = field(default=None, init=False, repr=False)
    managed_agent_id: str | None = field(default=None, init=False, repr=False)
    resume_source_session_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_registry is None:
            self.tool_registry = default_tool_registry()
        if self.agent_manager is None:
            self.agent_manager = AgentManager()
        dependencies = self.dependencies or AgentRuntimeDependencies.from_runtime_config(
            self.runtime_config
        )
        dependencies = dependencies.with_overrides(
            plugin_runtime=self.plugin_runtime,
            hook_policy_runtime=self.hook_policy_runtime,
            mcp_runtime=self.mcp_runtime,
            remote_runtime=self.remote_runtime,
            remote_trigger_runtime=self.remote_trigger_runtime,
            search_runtime=self.search_runtime,
            account_runtime=self.account_runtime,
            ask_user_runtime=self.ask_user_runtime,
            config_runtime=self.config_runtime,
            lsp_runtime=self.lsp_runtime,
            plan_runtime=self.plan_runtime,
            task_runtime=self.task_runtime,
            team_runtime=self.team_runtime,
            workflow_runtime=self.workflow_runtime,
            worktree_runtime=self.worktree_runtime,
        )
        self.dependencies = dependencies
        self.plugin_runtime = dependencies.plugin_runtime
        self.hook_policy_runtime = dependencies.hook_policy_runtime
        self.mcp_runtime = dependencies.mcp_runtime
        self.remote_runtime = dependencies.remote_runtime
        self.remote_trigger_runtime = dependencies.remote_trigger_runtime
        self.search_runtime = dependencies.search_runtime
        self.account_runtime = dependencies.account_runtime
        self.ask_user_runtime = dependencies.ask_user_runtime
        self.config_runtime = dependencies.config_runtime
        self.lsp_runtime = dependencies.lsp_runtime
        self.plan_runtime = dependencies.plan_runtime
        self.task_runtime = dependencies.task_runtime
        self.team_runtime = dependencies.team_runtime
        self.workflow_runtime = dependencies.workflow_runtime
        self.worktree_runtime = dependencies.worktree_runtime
        self.runtime_config = self._apply_hook_policy_budget_overrides(self.runtime_config)
        registry = dict(self.tool_registry)
        plugin_tools = self.plugin_runtime.register_tool_aliases(registry)
        if plugin_tools:
            registry = {**registry, **plugin_tools}
        virtual_tools = self.plugin_runtime.register_virtual_tools(registry)
        if virtual_tools:
            registry = {**registry, **virtual_tools}
        self.tool_registry = registry
        self.llm_backend = resolve_llm_backend(self.llm_backend or self.model_config.llm_backend)
        self.client = build_llm_client(self.model_config, backend=self.llm_backend)
        self.tool_context = build_tool_context(
            self.runtime_config,
            dependencies=self.dependencies,
            tool_registry=self.tool_registry,
            extra_env=(
                self.hook_policy_runtime.safe_env()
                if self.hook_policy_runtime is not None
                else None
            ),
        )

    def set_model(self, model: str) -> None:
        self.model_config = replace(self.model_config, model=model)
        self.client = build_llm_client(self.model_config, backend=self.llm_backend)

    def clear_runtime_state(self) -> None:
        self.last_session = None
        self.last_run_result = None
        self.active_session_id = None
        self.last_session_path = None
        self.resume_source_session_id = None
        if self.plugin_runtime is not None:
            self.plugin_runtime.restore_session_state({})

    def build_prompt_context(self, scratchpad_directory: Path | None = None):
        return build_prompt_context(
            self.runtime_config,
            self.model_config,
            scratchpad_directory=scratchpad_directory,
            dependencies=self.dependencies,
        )

    def build_system_prompt_parts(
        self,
        prompt_context=None,
        *,
        tools: dict[str, AgentTool] | None = None,
    ) -> list[str]:
        if prompt_context is None:
            prompt_context = self.build_prompt_context()
        return build_system_prompt_parts(
            prompt_context=prompt_context,
            runtime_config=self.runtime_config,
            tools=tools or self._tool_registry_for_prompt(),
            available_agents=self.available_agents(),
            custom_system_prompt=self.custom_system_prompt,
            append_system_prompt=self.append_system_prompt,
            override_system_prompt=self.override_system_prompt,
        )

    def load_agent_registry(self):
        return load_agent_registry(self.runtime_config.cwd)

    def available_agents(self) -> tuple[AgentDefinition, ...]:
        return self.load_agent_registry().active_agents

    def build_session(
        self,
        user_prompt: str | None = None,
        *,
        scratchpad_directory: Path | None = None,
    ) -> AgentSessionState:
        prompt_context = self.build_prompt_context(scratchpad_directory)
        system_prompt_parts = self.build_system_prompt_parts(
            prompt_context,
            tools=self._tool_registry_for_prompt(),
        )
        return AgentSessionState.create(
            system_prompt_parts,
            user_prompt,
            user_context=prompt_context.user_context,
            system_context=prompt_context.system_context,
        )

    def _tool_registry_for_prompt(self) -> dict[str, AgentTool]:
        tool_specs = [tool.to_openai_tool() for tool in self.tool_registry.values()]
        if not self._should_compact_ollama_tool_schema(tool_specs):
            return dict(self.tool_registry)
        allowed_names = self._compact_ollama_tool_names()
        compact_registry = {
            name: tool
            for name, tool in self.tool_registry.items()
            if name in allowed_names
        }
        return compact_registry or dict(self.tool_registry)

    def _build_tool_specs_for_session(
        self,
        session: AgentSessionState,
    ) -> list[dict[str, object]]:
        tool_specs = [tool.to_openai_tool() for tool in self.tool_registry.values()]
        if not self._should_compact_ollama_tool_schema(tool_specs):
            return tool_specs
        allowed_names = self._compact_ollama_tool_names(session)
        compact_specs = [
            tool.to_openai_tool()
            for name, tool in self.tool_registry.items()
            if name in allowed_names
        ]
        return compact_specs or tool_specs

    def _should_compact_ollama_tool_schema(
        self,
        tool_specs: list[dict[str, object]],
    ) -> bool:
        base_url = self.model_config.base_url.strip().lower()
        if ':11434' not in base_url and 'ollama' not in base_url:
            return False
        serialized = json.dumps(tool_specs, ensure_ascii=True)
        return len(serialized) > _COMPACT_OLLAMA_TOOL_SCHEMA_CHAR_LIMIT

    def _compact_ollama_tool_names(
        self,
        session: AgentSessionState | None = None,
    ) -> set[str]:
        _ = session
        return {
            name for name in _COMPACT_OLLAMA_TOOL_NAMES if name in self.tool_registry
        }

    def _apply_hook_policy_budget_overrides(
        self,
        runtime_config: AgentRuntimeConfig,
    ) -> AgentRuntimeConfig:
        if self.hook_policy_runtime is None or not self.hook_policy_runtime.manifests:
            return runtime_config
        overrides = self.hook_policy_runtime.budget_overrides()
        if not overrides:
            return runtime_config
        budget = runtime_config.budget_config
        return replace(
            runtime_config,
            budget_config=BudgetConfig(
                max_total_tokens=(
                    budget.max_total_tokens
                    if budget.max_total_tokens is not None
                    else _optional_policy_int(overrides.get('max_total_tokens'))
                ),
                max_input_tokens=(
                    budget.max_input_tokens
                    if budget.max_input_tokens is not None
                    else _optional_policy_int(overrides.get('max_input_tokens'))
                ),
                max_output_tokens=(
                    budget.max_output_tokens
                    if budget.max_output_tokens is not None
                    else _optional_policy_int(overrides.get('max_output_tokens'))
                ),
                max_reasoning_tokens=(
                    budget.max_reasoning_tokens
                    if budget.max_reasoning_tokens is not None
                    else _optional_policy_int(overrides.get('max_reasoning_tokens'))
                ),
                max_total_cost_usd=(
                    budget.max_total_cost_usd
                    if budget.max_total_cost_usd is not None
                    else _optional_policy_float(overrides.get('max_total_cost_usd'))
                ),
                max_tool_calls=(
                    budget.max_tool_calls
                    if budget.max_tool_calls is not None
                    else _optional_policy_int(overrides.get('max_tool_calls'))
                ),
                max_delegated_tasks=(
                    budget.max_delegated_tasks
                    if budget.max_delegated_tasks is not None
                    else _optional_policy_int(overrides.get('max_delegated_tasks'))
                ),
                max_model_calls=(
                    budget.max_model_calls
                    if budget.max_model_calls is not None
                    else _optional_policy_int(overrides.get('max_model_calls'))
                ),
                max_session_turns=(
                    budget.max_session_turns
                    if budget.max_session_turns is not None
                    else _optional_policy_int(overrides.get('max_session_turns'))
                ),
            ),
        )

    def run(
        self,
        prompt: str,
        *,
        event_handler: RuntimeEventHandler | None = None,
    ) -> AgentRunResult:
        self.managed_agent_id = None
        self.resume_source_session_id = None
        if self.plugin_runtime is not None:
            self.plugin_runtime.restore_session_state({})
        session_id = uuid4().hex
        scratchpad_directory = self._ensure_scratchpad_directory(session_id)
        result = self._run_prompt(
            prompt,
            base_session=None,
            session_id=session_id,
            scratchpad_directory=scratchpad_directory,
            existing_file_history=(),
            event_handler=event_handler,
        )
        self._accumulate_usage(result)
        self._finalize_managed_agent(result)
        return result

    def resume(
        self,
        prompt: str,
        stored_session: StoredAgentSession,
        *,
        event_handler: RuntimeEventHandler | None = None,
    ) -> AgentRunResult:
        self.managed_agent_id = None
        self.resume_source_session_id = stored_session.session_id
        session = AgentSessionState.from_persisted(
            system_prompt_parts=stored_session.system_prompt_parts,
            user_context=stored_session.user_context,
            system_context=stored_session.system_context,
            messages=stored_session.messages,
        )
        self._append_file_history_replay_if_needed(
            session,
            stored_session.file_history,
        )
        self._append_compaction_replay_if_needed(session)
        self.active_session_id = stored_session.session_id
        self.last_session = session
        self.last_session_path = str(
            self.runtime_config.session_directory / f'{stored_session.session_id}.json'
        )
        if self.plugin_runtime is not None:
            self.plugin_runtime.restore_session_state(stored_session.plugin_state)
        scratchpad_directory = (
            Path(stored_session.scratchpad_directory)
            if stored_session.scratchpad_directory
            else self._ensure_scratchpad_directory(stored_session.session_id)
        )
        result = self._run_prompt(
            prompt,
            base_session=session,
            session_id=stored_session.session_id,
            scratchpad_directory=scratchpad_directory,
            existing_file_history=stored_session.file_history,
            event_handler=event_handler,
        )
        self._accumulate_usage(result)
        self._finalize_managed_agent(result)
        return result

    def _run_prompt(
        self,
        prompt: str,
        *,
        base_session: AgentSessionState | None,
        session_id: str,
        scratchpad_directory: Path | None,
        existing_file_history: tuple[dict[str, object], ...],
        event_handler: RuntimeEventHandler | None,
    ) -> AgentRunResult:
        slash_result = preprocess_slash_command(self, prompt)
        if slash_result.handled and not slash_result.should_query:
            return AgentRunResult(
                final_output=slash_result.output,
                turns=0,
                tool_calls=0,
                transcript=slash_result.transcript,
                session_id=self.active_session_id,
                session_path=self.last_session_path,
                scratchpad_directory=(
                    str(scratchpad_directory) if scratchpad_directory is not None else None
                ),
            )

        effective_prompt = self._apply_hook_policy_before_prompt_hooks(
            slash_result.prompt or prompt
        )
        effective_prompt = self._apply_plugin_before_prompt_hooks(effective_prompt)
        effective_prompt = self._apply_plugin_resume_hooks(
            effective_prompt,
            resumed=base_session is not None,
        )
        self.managed_agent_id = self.agent_manager.start_agent(
            prompt=effective_prompt,
            parent_agent_id=self.parent_agent_id,
            group_id=self.managed_group_id,
            child_index=self.managed_child_index,
            label=self.managed_label or ('root' if base_session is None else 'resume'),
            resumed_from_session_id=self.resume_source_session_id,
        )
        session = (
            base_session
            if base_session is not None
            else self.build_session(
                None,
                scratchpad_directory=scratchpad_directory,
            )
        )
        session.append_user(effective_prompt)
        self.last_session = session
        self.active_session_id = session_id
        state = self._build_prompt_run_state(
            session_id=session_id,
            scratchpad_directory=scratchpad_directory,
            effective_prompt=effective_prompt,
            session=session,
            existing_file_history=existing_file_history,
            base_session=base_session,
            event_handler=event_handler,
        )
        initial_budget = self._check_budget(
            state.total_usage,
            state.total_cost_usd,
            tool_calls=state.tool_calls,
            delegated_tasks=state.delegated_tasks,
            model_calls=state.model_calls,
            session_turns=state.starting_session_turns,
        )
        if initial_budget.exceeded:
            return self._finalize_run_state_result(
                state,
                final_output=initial_budget.reason or 'Stopped before the first model call.',
                turns=0,
                stop_reason='budget_exceeded',
            )

        max_turns = self.runtime_config.max_turns
        while max_turns is None or state.turn_index < max_turns:
            state.turn_index += 1
            self._microcompact_session_if_needed(
                state.session,
                state.stream_events,
                turn_index=state.turn_index,
            )
            self._snip_session_if_needed(
                state.session,
                state.stream_events,
                turn_index=state.turn_index,
            )
            self._compact_session_if_needed(
                state.session,
                state.stream_events,
                turn_index=state.turn_index,
            )
            preflight = self._preflight_prompt_length(
                state.session,
                state.stream_events,
                turn_index=state.turn_index,
            )
            if preflight.usage_increment.total_tokens or preflight.model_calls_increment:
                state.total_usage = state.total_usage + preflight.usage_increment
                state.total_cost_usd = self.model_config.pricing.estimate_cost_usd(
                    state.total_usage
                )
                state.model_calls += preflight.model_calls_increment
                budget_after_preflight = self._check_budget(
                    state.total_usage,
                    state.total_cost_usd,
                    tool_calls=state.tool_calls,
                    delegated_tasks=state.delegated_tasks,
                    model_calls=state.model_calls,
                    session_turns=state.starting_session_turns + state.turn_index,
                )
                if budget_after_preflight.exceeded:
                    return self._finalize_run_state_result(
                        state,
                        final_output=(
                            budget_after_preflight.reason
                            or 'Stopped because the runtime budget was exceeded.'
                        ),
                        turns=state.turn_index,
                        stop_reason='budget_exceeded',
                    )
            if preflight.stop_reason is not None:
                return self._finalize_run_state_result(
                    state,
                    final_output=preflight.reason or 'Stopped before the next model call.',
                    turns=max(state.turn_index - 1, 0),
                    stop_reason=preflight.stop_reason,
                    append_after_turn=True,
                    after_turn_index=max(state.turn_index - 1, 0),
                )
            try:
                tool_specs = self._build_tool_specs_for_session(state.session)
                turn = self._query_turn_with_recovery(
                    state.session,
                    state.stream_events,
                    turn_index=state.turn_index,
                    tool_specs=tool_specs,
                )
            except LLMBackendError as exc:
                return self._finalize_run_state_result(
                    state,
                    final_output=str(exc),
                    turns=max(state.turn_index - 1, 0),
                    stop_reason='backend_error',
                    append_after_turn=True,
                    after_turn_index=state.turn_index,
                )
            directive = self._process_model_turn(state, turn)
            if directive.result is not None:
                return directive.result
            if directive.continue_loop:
                continue

            for tool_call in turn.tool_calls:
                state.assistant_response_segments.clear()
                state.tool_calls += 1
                if tool_call.name in ('Agent', 'delegate_agent'):
                    state.delegated_tasks += delegated_task_units(
                        tool_call.arguments
                    )
                budget_after_tool_request = self._check_budget(
                    state.total_usage,
                    state.total_cost_usd,
                    tool_calls=state.tool_calls,
                    delegated_tasks=state.delegated_tasks,
                    model_calls=state.model_calls,
                    session_turns=state.starting_session_turns + state.turn_index,
                )
                if budget_after_tool_request.exceeded:
                    state.stream_events.append(
                        {
                            'type': 'task_budget_exceeded',
                            'turn_index': state.turn_index,
                            'tool_name': tool_call.name,
                            'tool_call_id': tool_call.id,
                            'reason': budget_after_tool_request.reason,
                        }
                    )
                    return self._finalize_run_state_result(
                        state,
                        final_output=(
                            budget_after_tool_request.reason
                            or 'Stopped because the runtime budget was exceeded.'
                        ),
                        turns=state.turn_index,
                        stop_reason='budget_exceeded',
                    )
                tool_outcome = execute_runtime_tool_call(
                    tool_call=tool_call,
                    turn_index=state.turn_index,
                    session=state.session,
                    tool_registry=self.tool_registry,
                    tool_context=self.tool_context,
                    stream_events=state.stream_events,
                    plugin_runtime=self.plugin_runtime,
                    hooks=self._build_tool_call_execution_hooks(),
                )
                if tool_outcome.history_entry is not None:
                    state.file_history.append(tool_outcome.history_entry)

        return self._finalize_run_state_result(
            state,
            final_output=(
                state.last_content
                or 'Stopped: max turns reached before the model produced a final answer.'
            ),
            turns=state.turn_index,
            stop_reason='max_turns',
        )

    def _build_prompt_run_state(
        self,
        *,
        session_id: str,
        scratchpad_directory: Path | None,
        effective_prompt: str,
        session: AgentSessionState,
        existing_file_history: tuple[dict[str, object], ...],
        base_session: AgentSessionState | None,
        event_handler: RuntimeEventHandler | None,
    ) -> PromptRunState:
        starting_usage = UsageStats()
        starting_cost_usd = 0.0
        starting_tool_calls = 0
        starting_session_turns = 0
        starting_model_calls = 0
        if base_session is not None and self.resume_source_session_id:
            try:
                stored_resume_state = load_agent_session(
                    self.resume_source_session_id,
                    directory=self.runtime_config.session_directory,
                )
            except OSError:
                stored_resume_state = None
            if stored_resume_state is not None:
                starting_usage = usage_from_payload(stored_resume_state.usage)
                starting_cost_usd = stored_resume_state.total_cost_usd
                starting_tool_calls = stored_resume_state.tool_calls
                starting_session_turns = stored_resume_state.turns
                budget_state = (
                    stored_resume_state.budget_state
                    if isinstance(stored_resume_state.budget_state, dict)
                    else {}
                )
                model_calls = budget_state.get('model_calls', 0)
                if isinstance(model_calls, int) and not isinstance(model_calls, bool):
                    starting_model_calls = model_calls
        file_history = list(existing_file_history)
        return PromptRunState(
            session_id=session_id,
            scratchpad_directory=scratchpad_directory,
            effective_prompt=effective_prompt,
            session=session,
            starting_session_turns=starting_session_turns,
            tool_calls=starting_tool_calls,
            total_usage=starting_usage,
            total_cost_usd=starting_cost_usd,
            file_history=file_history,
            stream_events=_RuntimeEventRecorder(event_handler),
            delegated_tasks=sum(
                1
                for entry in file_history
                if entry.get('action') in ('delegate_agent', 'Agent')
            ),
            model_calls=starting_model_calls,
        )

    def _finalize_run_state_result(
        self,
        state: PromptRunState,
        *,
        final_output: str,
        turns: int,
        stop_reason: str | None,
        append_after_turn: bool = False,
        after_turn_index: int | None = None,
    ) -> AgentRunResult:
        result = build_run_result(
            state,
            final_output=final_output,
            turns=turns,
            stop_reason=stop_reason,
        )
        if append_after_turn:
            result = self._append_runtime_after_turn_events(
                result,
                prompt=state.effective_prompt,
                turn_index=after_turn_index if after_turn_index is not None else state.turn_index,
            )
        result = self._persist_session(state.session, result)
        self.last_run_result = result
        return result

    def _query_turn_with_recovery(
        self,
        session: AgentSessionState,
        stream_events: _RuntimeEventRecorder,
        *,
        turn_index: int,
        tool_specs: list[dict[str, object]],
    ) -> AssistantTurn:
        try:
            return self._query_model(session, tool_specs, stream_events)
        except LLMBackendError as exc:
            if not self._is_prompt_too_long_error(exc):
                raise
            if not self._reactive_compact_session(
                session,
                stream_events,
                turn_index=turn_index,
            ):
                raise
            retry_specs = self._build_tool_specs_for_session(session)
            turn = self._query_model(session, retry_specs, stream_events)
            stream_events.extend(
                {
                    'type': 'reactive_compact_retry',
                    'turn_index': turn_index,
                }
                for _ in [0]
            )
            return turn

    def _process_model_turn(
        self,
        state: PromptRunState,
        turn: AssistantTurn,
    ) -> TurnLoopDirective:
        state.model_calls += 1
        state.total_usage = state.total_usage + turn.usage
        state.total_cost_usd = self.model_config.pricing.estimate_cost_usd(
            state.total_usage
        )
        state.last_content = turn.content

        budget_after_model = self._check_budget(
            state.total_usage,
            state.total_cost_usd,
            tool_calls=state.tool_calls,
            delegated_tasks=state.delegated_tasks,
            model_calls=state.model_calls,
            session_turns=state.starting_session_turns + state.turn_index,
        )
        if budget_after_model.exceeded:
            return TurnLoopDirective(
                result=self._finalize_run_state_result(
                    state,
                    final_output=(
                        budget_after_model.reason
                        or 'Stopped because the runtime budget was exceeded.'
                    ),
                    turns=state.turn_index,
                    stop_reason='budget_exceeded',
                )
            )
        if turn.tool_calls:
            return TurnLoopDirective()
        return self._handle_assistant_only_turn(state, turn)

    def _handle_assistant_only_turn(
        self,
        state: PromptRunState,
        turn: AssistantTurn,
    ) -> TurnLoopDirective:
        state.assistant_response_segments.append(turn.content)
        truncated_response = self._is_truncated_response(turn)
        if self._should_continue_response(
            turn,
            prompt=state.effective_prompt,
            tool_calls_so_far=state.tool_calls,
            current_response=''.join(state.assistant_response_segments),
            continuation_count=len(state.assistant_response_segments),
        ):
            if not truncated_response and state.assistant_response_segments:
                state.assistant_response_segments[-1] = (
                    state.assistant_response_segments[-1].rstrip() + '\n\n'
                )
            state.session.append_user(
                self._build_continuation_prompt(
                    truncated=truncated_response,
                ),
                metadata={
                    'kind': 'continuation_request',
                    'continuation_index': len(state.assistant_response_segments),
                },
                message_id=f'continuation_{state.turn_index}',
            )
            state.stream_events.append(
                {
                    'type': 'continuation_request',
                    'reason': turn.finish_reason,
                    'continuation_index': len(state.assistant_response_segments),
                }
            )
            state.last_content = ''.join(state.assistant_response_segments)
            return TurnLoopDirective(continue_loop=True)
        return TurnLoopDirective(
            result=self._finalize_run_state_result(
                state,
                final_output=''.join(state.assistant_response_segments),
                turns=state.turn_index,
                stop_reason=turn.finish_reason,
                append_after_turn=True,
                after_turn_index=state.turn_index,
            )
        )

    def _query_model(
        self,
        session: AgentSessionState,
        tool_specs: list[dict[str, object]],
        stream_events: _RuntimeEventRecorder | None = None,
    ) -> AssistantTurn:
        return query_model_turn(
            client=self.client,
            session=session,
            tool_specs=tool_specs,
            stream_model_responses=self.runtime_config.stream_model_responses,
            output_schema=self.runtime_config.output_schema,
            stream_event_append=stream_events.append if stream_events is not None else None,
        )

    def _build_tool_call_execution_hooks(self) -> ToolCallExecutionHooks:
        return ToolCallExecutionHooks(
            plugin_tool_preflight_messages=self._plugin_tool_preflight_messages,
            hook_policy_tool_preflight_messages=self._hook_policy_tool_preflight_messages,
            plugin_block_message=self._plugin_block_message,
            hook_policy_block_message=self._hook_policy_block_message,
            execute_delegate_agent=lambda arguments, tool_name: self._execute_delegate_agent(
                arguments,
                tool_name=tool_name,
            ),
            execute_skill=self._execute_skill,
            plugin_tool_result_messages=self._plugin_tool_result_messages,
            hook_policy_tool_result_messages=self._hook_policy_tool_result_messages,
            append_runtime_tool_followup_events=lambda events, current_tool_call, tool_result: self._append_runtime_tool_followup_events(
                events,
                tool_call=current_tool_call,
                tool_result=tool_result,
            ),
            build_plugin_tool_runtime_message=self._build_plugin_tool_runtime_message,
            refresh_runtime_views_for_tool_result=self._refresh_runtime_views_for_tool_result,
            build_file_history_entry=lambda current_tool_call, tool_result, current_turn_index: self._build_file_history_entry(
                tool_call=current_tool_call,
                tool_result=tool_result,
                turn_index=current_turn_index,
            ),
        )

    def _should_continue_response(
        self,
        turn: AssistantTurn,
        *,
        prompt: str = '',
        tool_calls_so_far: int = 0,
        current_response: str = '',
        continuation_count: int = 0,
    ) -> bool:
        if self._is_truncated_response(turn):
            return True
        if (
            normalize_finish_reason(
                turn.finish_reason,
                has_tool_calls=bool(turn.tool_calls),
                content=turn.content,
            )
            != 'stop'
        ):
            return False
        if tool_calls_so_far <= 0 or continuation_count != 1:
            return False
        prompt_text = prompt.lower()
        if not any(
            keyword in prompt_text
            for keyword in (
                'analy',
                'review',
                'audit',
                'architect',
                'improv',
                'investigat',
                'debug',
                'refactor',
            )
        ):
            return False
        response_text = (current_response or turn.content or '').strip()
        if len(response_text) >= 900:
            return False
        lowered_response = response_text.lower()
        if any(
            marker in lowered_response
            for marker in (
                'summary of recommended action',
                'next steps',
                'recommendations',
                'recommended action',
                'here are',
                'to improve',
            )
        ):
            return False
        if any(marker in response_text for marker in ('\n1.', '\n2.', '\n- ', '\n## ')):
            return False
        if response_text.endswith('?'):
            return False
        return True

    def _is_truncated_response(self, turn: AssistantTurn) -> bool:
        return turn.finish_reason in {'length', 'max_tokens'}

    def _build_continuation_prompt(self, *, truncated: bool) -> str:
        if truncated:
            return (
                '<system-reminder>\n'
                'Your previous answer was truncated because the model stopped early. '
                'Continue exactly where you left off. Do not repeat completed text.\n'
                '</system-reminder>'
            )
        return (
            '<system-reminder>\n'
            'Your previous answer appears to stop at an intermediate summary. '
            'Continue working on the user\'s request until you either provide a complete answer '
            'or determine that you need to use another tool. Do not stop after a brief partial analysis.\n'
            '</system-reminder>'
        )

    def _check_budget(
        self,
        usage: UsageStats,
        total_cost_usd: float,
        *,
        tool_calls: int,
        delegated_tasks: int,
        model_calls: int,
        session_turns: int,
    ) -> BudgetDecision:
        budget = self.runtime_config.budget_config
        token_reason = self._check_token_budget(usage, budget)
        if token_reason is not None:
            return BudgetDecision(exceeded=True, reason=token_reason)
        if (
            budget.max_total_cost_usd is not None
            and total_cost_usd > budget.max_total_cost_usd
        ):
            return BudgetDecision(
                exceeded=True,
                reason=(
                    'Stopped because the total estimated cost '
                    f'(${total_cost_usd:.6f}) exceeded the configured budget '
                    f'(${budget.max_total_cost_usd:.6f}).'
                ),
            )
        if (
            budget.max_tool_calls is not None
            and tool_calls > budget.max_tool_calls
        ):
            return BudgetDecision(
                exceeded=True,
                reason=(
                    'Stopped because the tool-call budget was exceeded '
                    f'({tool_calls} > {budget.max_tool_calls}).'
                ),
            )
        if (
            budget.max_delegated_tasks is not None
            and delegated_tasks > budget.max_delegated_tasks
        ):
            return BudgetDecision(
                exceeded=True,
                reason=(
                    'Stopped because the delegated-task budget was exceeded '
                    f'({delegated_tasks} > {budget.max_delegated_tasks}).'
                ),
            )
        if (
            budget.max_model_calls is not None
            and model_calls > budget.max_model_calls
        ):
            return BudgetDecision(
                exceeded=True,
                reason=(
                    'Stopped because the model-call budget was exceeded '
                    f'({model_calls} > {budget.max_model_calls}).'
                ),
            )
        if (
            budget.max_session_turns is not None
            and session_turns > budget.max_session_turns
        ):
            return BudgetDecision(
                exceeded=True,
                reason=(
                    'Stopped because the session-turn budget was exceeded '
                    f'({session_turns} > {budget.max_session_turns}).'
                ),
            )
        return BudgetDecision(exceeded=False)

    def _preflight_prompt_length(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
    ) -> PromptPreflightResult:
        return preflight_prompt_length(
            self,
            session,
            stream_events,
            turn_index=turn_index,
            calculate_token_budget_fn=calculate_token_budget,
            compact_conversation_fn=compact_conversation,
            reduce_context_pressure_fn=self._reduce_context_pressure,
        )

    def _can_auto_compact_with_summary(self, session: AgentSessionState) -> bool:
        return can_auto_compact_with_summary(self, session)

    def _build_prompt_length_error(self, snapshot) -> str:
        return build_prompt_length_error(snapshot)

    def _microcompact_session_if_needed(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
    ) -> None:
        """Run time-based microcompaction to clear old tool results.

        Fires when the gap since the last assistant message exceeds the
        threshold (default 60 minutes), indicating the server-side cache
        has expired and the full prefix will be rewritten anyway.
        """
        if not session.messages:
            return
        result = _microcompact_messages(
            session.messages,
            model=self.model_config.model,
        )
        if not result.triggered:
            return
        session.messages = result.messages
        stream_events.append(
            {
                'type': 'microcompact',
                'turn_index': turn_index,
                'cleared_tool_count': result.cleared_tool_count,
                'kept_tool_count': result.kept_tool_count,
                'estimated_tokens_saved': result.estimated_tokens_saved,
                'gap_minutes': round(result.gap_minutes, 1),
            }
        )

    def _snip_session_if_needed(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
    ) -> None:
        threshold = self.runtime_config.auto_snip_threshold_tokens
        if threshold is None or threshold <= 0:
            return
        self._reduce_context_pressure(
            session,
            stream_events,
            turn_index=turn_index,
            target_tokens=threshold,
            allow_compaction=False,
        )

    def _compact_session_if_needed(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
    ) -> None:
        threshold = self.runtime_config.auto_compact_threshold_tokens
        if threshold is None or threshold <= 0:
            return
        self._reduce_context_pressure(
            session,
            stream_events,
            turn_index=turn_index,
            target_tokens=threshold,
            allow_compaction=True,
        )

    def _reactive_compact_session(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
    ) -> bool:
        return reactive_compact_session(
            self,
            session,
            stream_events,
            turn_index=turn_index,
        )

    def _reduce_context_pressure(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
        target_tokens: int,
        allow_compaction: bool,
        reactive: bool = False,
    ) -> bool:
        return reduce_context_pressure(
            self,
            session,
            stream_events,
            turn_index=turn_index,
            target_tokens=target_tokens,
            allow_compaction=allow_compaction,
            reactive=reactive,
        )

    def _snip_session_pass(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
        target_tokens: int,
        current_total: int,
        reactive: bool,
    ) -> bool:
        return snip_session_pass(
            self,
            session,
            stream_events,
            turn_index=turn_index,
            target_tokens=target_tokens,
            current_total=current_total,
            reactive=reactive,
        )

    def _compact_session_pass(
        self,
        session: AgentSessionState,
        stream_events: list[dict[str, object]],
        *,
        turn_index: int,
        usage_total: int,
        reactive: bool,
    ) -> bool:
        return compact_session_pass(
            self,
            session,
            stream_events,
            turn_index=turn_index,
            usage_total=usage_total,
            reactive=reactive,
        )

    def _check_token_budget(
        self,
        usage: UsageStats,
        budget: BudgetConfig,
    ) -> str | None:
        return check_token_budget(usage, budget)

    def _build_file_history_entry(
        self,
        *,
        tool_call: ToolCall,
        tool_result,
        turn_index: int,
    ) -> dict[str, object] | None:
        if not tool_result.metadata:
            return None
        if (
            'path' not in tool_result.metadata
            and 'command' not in tool_result.metadata
            and tool_result.metadata.get('action') not in ('delegate_agent', 'Agent')
        ):
            return None
        metadata = dict(tool_result.metadata)
        entry: dict[str, object] = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'turn_index': turn_index,
            'tool_call_id': tool_call.id,
            'tool_name': tool_call.name,
            'ok': tool_result.ok,
            'history_entry_id': f'{turn_index}:{tool_call.id}:{tool_call.name}',
            'result_preview': self._preview_text(tool_result.content, 220),
            **metadata,
        }
        action = metadata.get('action')
        path = metadata.get('path')
        if isinstance(path, str) and path:
            entry['history_kind'] = 'file_change'
            entry['changed_paths'] = [path]
            before_sha256 = metadata.get('before_sha256')
            if isinstance(before_sha256, str) and before_sha256:
                entry['before_snapshot_id'] = f'{path}:{before_sha256[:12]}'
            after_sha256 = metadata.get('after_sha256')
            if isinstance(after_sha256, str) and after_sha256:
                entry['after_snapshot_id'] = f'{path}:{after_sha256[:12]}'
        elif isinstance(metadata.get('command'), str):
            entry['history_kind'] = 'shell'
        elif action in ('delegate_agent', 'Agent'):
            entry['history_kind'] = 'delegation'
            delegate_batches = metadata.get('delegate_batches')
            if isinstance(delegate_batches, list):
                entry['delegate_batch_count'] = len(delegate_batches)
            dependency_skips = metadata.get('dependency_skips')
            if isinstance(dependency_skips, int) and not isinstance(dependency_skips, bool):
                entry['dependency_skips'] = dependency_skips
        else:
            entry['history_kind'] = 'tool'
        return entry

    def _compact_prefix_count(self, session: AgentSessionState) -> int:
        prefix_count = 0
        for message in session.messages:
            if prefix_count == 0 and message.role == 'system':
                prefix_count += 1
                continue
            if (
                prefix_count == 1
                and message.role == 'user'
                and message.content.startswith('<system-reminder>')
            ):
                prefix_count += 1
                continue
            break
        return prefix_count

    def _message_can_be_snipped(self, message) -> bool:
        if message.metadata.get('kind') in {
            'compact_boundary',
            'snipped_message',
            'file_history_replay',
        }:
            return False
        if message.role == 'tool':
            return True
        if message.role == 'assistant' and (message.tool_calls or len(message.content) > 600):
            return True
        if (
            message.role == 'user'
            and message.metadata.get('kind') in {'continuation_request', 'file_history_replay'}
        ):
            return True
        return False

    def _build_snipped_message_content(self, message) -> str:
        preview = ' '.join(message.content.split())
        if len(preview) > 120:
            preview = preview[:117] + '...'
        if message.role == 'tool':
            label = f'tool result ({message.name or "tool"})'
        elif message.role == 'assistant':
            label = 'assistant message with tool calls'
        else:
            label = message.role
        return (
            '<system-reminder>\n'
            f'Older {label} was snipped to save context.\n'
            f'Message id: {message.message_id or "(none)"}\n'
            f'Preview: {preview or "(empty)"}\n'
            '</system-reminder>'
        )

    def _build_compact_boundary_message(
        self,
        messages,
        *,
        turn_index: int,
        estimated_tokens_before: int,
        estimated_tokens_removed: int,
        preserved_tail_count: int,
        preserved_tail,
    ):
        summary_lines = [
            '<system-reminder>',
            'Earlier conversation history was compacted to keep the session within the context budget.',
            '',
            'Compacted summary:',
        ]
        remaining = 24
        for message in messages:
            if remaining <= 0:
                break
            label = message.role
            if message.role == 'tool' and message.name:
                label = f'tool:{message.name}'
            snippet = ' '.join(message.content.split())
            if len(snippet) > 160:
                snippet = snippet[:157] + '...'
            if not snippet:
                snippet = '(empty)'
            summary_lines.append(f'- {label}: {snippet}')
            remaining -= 1
        if len(messages) > 24:
            summary_lines.append(f'- ... plus {len(messages) - 24} older messages')
        summary_lines.extend(
            [
                '',
                'Keep using the preserved recent tail as the active working set.',
                '</system-reminder>',
            ]
        )
        from src.agent.agent_session import AgentMessage

        nested_compaction_count = sum(
            1 for message in messages if message.metadata.get('kind') == 'compact_boundary'
        )
        prior_depths = [
            int(message.metadata.get('compaction_depth', 0))
            for message in messages
            if isinstance(message.metadata.get('compaction_depth', 0), int)
        ]
        compaction_depth = (max(prior_depths) if prior_depths else 0) + 1
        compacted_kinds: dict[str, int] = {}
        source_mutation_totals: dict[str, int] = {}
        compacted_lineage_ids: list[str] = []
        preserved_tail_lineage_ids = [
            lineage_id
            for lineage_id in (
                message.metadata.get('lineage_id') for message in preserved_tail
            )
            if isinstance(lineage_id, str) and lineage_id
        ]
        max_source_revision = 0
        max_source_mutation_serial = 0
        compacted_revision_total = 0
        for message in messages:
            kind = message.metadata.get('kind')
            label = str(kind) if isinstance(kind, str) and kind else message.role
            compacted_kinds[label] = compacted_kinds.get(label, 0) + 1
            lineage_id = message.metadata.get('lineage_id')
            if isinstance(lineage_id, str) and lineage_id:
                compacted_lineage_ids.append(lineage_id)
            revision = message.metadata.get('revision')
            if isinstance(revision, int) and not isinstance(revision, bool):
                max_source_revision = max(max_source_revision, revision)
                compacted_revision_total += revision
            max_mutation_serial = message.metadata.get('max_mutation_serial')
            if isinstance(max_mutation_serial, int) and not isinstance(max_mutation_serial, bool):
                max_source_mutation_serial = max(
                    max_source_mutation_serial,
                    max_mutation_serial,
                )
            mutation_totals = message.metadata.get('mutation_totals')
            if isinstance(mutation_totals, dict):
                for mutation_kind, count in mutation_totals.items():
                    if (
                        not isinstance(mutation_kind, str)
                        or not mutation_kind
                        or isinstance(count, bool)
                        or not isinstance(count, int)
                        or count <= 0
                    ):
                        continue
                    source_mutation_totals[mutation_kind] = (
                        source_mutation_totals.get(mutation_kind, 0) + count
                    )

        compact_boundary_id = f'compact_boundary_{turn_index}_{len(messages)}'

        return AgentMessage(
            role='system',
            content='\n'.join(summary_lines),
            message_id=compact_boundary_id,
            metadata={
                'kind': 'compact_boundary',
                'lineage_id': compact_boundary_id,
                'revision': 0,
                'revision_count': 1,
                'message_role': 'system',
                'turn_index': turn_index,
                'compacted_message_count': len(messages),
                'estimated_tokens_before': estimated_tokens_before,
                'estimated_tokens_removed': estimated_tokens_removed,
                'preserved_tail_count': preserved_tail_count,
                'preserved_tail_ids': [
                    message.message_id for message in preserved_tail if message.message_id
                ],
                'compaction_depth': compaction_depth,
                'nested_compaction_count': nested_compaction_count,
                'compacted_kinds': compacted_kinds,
                'compacted_lineage_ids': compacted_lineage_ids,
                'preserved_tail_lineage_ids': preserved_tail_lineage_ids,
                'max_source_revision': max_source_revision,
                'max_source_mutation_serial': max_source_mutation_serial,
                'source_mutation_totals': source_mutation_totals,
                'compacted_revision_total': compacted_revision_total,
                'compacted_message_ids': [
                    message.message_id for message in messages if message.message_id
                ],
            },
        )

    def _is_prompt_too_long_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        patterns = (
            'prompt is too long',
            'maximum context length',
            'context length exceeded',
            'too many tokens',
            'input too long',
            'context window',
        )
        return any(pattern in text for pattern in patterns)

    def _execute_skill(
        self,
        arguments: dict[str, object],
    ) -> ToolExecutionResult:
        """Execute a skill through the Skill tool.

        Checks bundled skills first, then falls back to slash commands.
        """
        from src.agent.agent_slash_commands import find_slash_command, get_slash_command_specs
        from src.agent.bundled_skills import find_bundled_skill, get_bundled_skills

        skill_name = arguments.get('skill')
        if not isinstance(skill_name, str) or not skill_name.strip():
            return ToolExecutionResult(
                name='Skill',
                ok=False,
                content='skill must be a non-empty string',
            )

        # Normalize: strip leading '/' if present
        skill_name = skill_name.strip().lstrip('/')
        args = arguments.get('args', '')
        if not isinstance(args, str):
            args = str(args) if args is not None else ''

        # 1. Check bundled skills first
        bundled = find_bundled_skill(skill_name)
        if bundled is not None:
            prompt = bundled.get_prompt(self, args.strip())
            return ToolExecutionResult(
                name='Skill',
                ok=True,
                content=prompt,
                metadata={
                    'action': 'skill',
                    'skill_name': skill_name,
                    'source': 'bundled',
                    'should_query': True,
                },
            )

        # 2. Fall back to slash commands
        spec = find_slash_command(skill_name)
        if spec is None:
            available_cmds = sorted(
                name
                for s in get_slash_command_specs()
                for name in s.names
            )
            available_skills = [sk.name for sk in get_bundled_skills()]
            all_available = sorted(set(available_cmds + available_skills))
            return ToolExecutionResult(
                name='Skill',
                ok=False,
                content=(
                    f'Unknown skill: {skill_name}. '
                    f'Available skills: {", ".join(all_available[:30])}'
                ),
                metadata={'action': 'skill_not_found', 'skill_name': skill_name},
            )

        # Invoke the slash command handler
        input_text = f'/{skill_name} {args}'.strip()
        result = spec.handler(self, args.strip(), input_text)

        if result.output:
            content = result.output
        elif result.prompt:
            content = result.prompt
        else:
            content = f'Skill /{skill_name} completed.'

        return ToolExecutionResult(
            name='Skill',
            ok=True,
            content=content,
            metadata={
                'action': 'skill',
                'skill_name': skill_name,
                'command_name': spec.names[0],
                'handled': result.handled,
                'should_query': result.should_query,
            },
        )

    def _execute_delegate_agent(
        self,
        arguments: dict[str, object],
        *,
        tool_name: str = 'Agent',
    ) -> ToolExecutionResult:
        return execute_delegate_agent(self, arguments, tool_name=tool_name)

    def _append_runtime_tool_followup_events(
        self,
        stream_events: list[dict[str, object]],
        *,
        tool_call: ToolCall,
        tool_result: ToolExecutionResult,
    ) -> None:
        metadata = tool_result.metadata
        if metadata.get('action') == 'plugin_virtual_tool':
            stream_events.append(
                {
                    'type': 'plugin_virtual_tool_result',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_call.name,
                    'plugin_name': metadata.get('plugin_name'),
                    'virtual_tool': metadata.get('virtual_tool'),
                }
            )
        plugin_delegate_preflight = metadata.get('plugin_delegate_preflight_messages')
        if isinstance(plugin_delegate_preflight, list) and plugin_delegate_preflight:
            stream_events.append(
                {
                    'type': 'plugin_delegate_preflight',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_call.name,
                    'message_count': len(plugin_delegate_preflight),
                }
            )
        plugin_delegate_after = metadata.get('plugin_delegate_after_messages')
        if isinstance(plugin_delegate_after, list) and plugin_delegate_after:
            stream_events.append(
                {
                    'type': 'plugin_delegate_after',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_call.name,
                    'message_count': len(plugin_delegate_after),
                }
            )
        if tool_call.name not in ('Agent', 'delegate_agent'):
            return
        delegate_batches = metadata.get('delegate_batches')
        if isinstance(delegate_batches, list):
            for batch in delegate_batches:
                if not isinstance(batch, dict):
                    continue
                stream_events.append(
                    {
                        'type': 'delegate_batch_result',
                        'tool_call_id': tool_call.id,
                        'group_id': metadata.get('group_id'),
                        'batch_index': batch.get('batch_index'),
                        'status': batch.get('status'),
                        'labels': batch.get('labels'),
                        'completed_children': batch.get('completed_children'),
                        'failed_children': batch.get('failed_children'),
                        'skipped_children': batch.get('skipped_children'),
                    }
                )
        child_results = metadata.get('child_results')
        if isinstance(child_results, list):
            for child in child_results:
                if not isinstance(child, dict):
                    continue
                stream_events.append(
                    {
                        'type': 'delegate_subtask_result',
                        'tool_call_id': tool_call.id,
                        'group_id': metadata.get('group_id'),
                        'label': child.get('label'),
                        'index': child.get('index'),
                        'session_id': child.get('session_id'),
                        'stop_reason': child.get('stop_reason'),
                        'tool_calls': child.get('tool_calls'),
                        'turns': child.get('turns'),
                        'resume_used': child.get('resume_used'),
                        'resumed_from_session_id': child.get('resumed_from_session_id'),
                        'depends_on': child.get('depends_on'),
                        'batch_index': child.get('batch_index'),
                    }
                )
        if metadata.get('group_id') is not None:
            stream_events.append(
                {
                    'type': 'delegate_group_result',
                    'tool_call_id': tool_call.id,
                    'group_id': metadata.get('group_id'),
                    'group_status': metadata.get('group_status'),
                    'subtask_count': metadata.get('subtask_count'),
                    'completed_children': metadata.get('completed_children'),
                    'failed_children': metadata.get('failed_children'),
                    'resumed_children': metadata.get('resumed_children'),
                    'strategy': metadata.get('strategy'),
                    'max_failures': metadata.get('max_failures'),
                    'batch_count': len(delegate_batches) if isinstance(delegate_batches, list) else 0,
                    'dependency_skips': metadata.get('dependency_skips'),
                }
            )

    def _preview_text(self, text: str, limit: int) -> str:
        normalized = ' '.join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + '...'

    def _ensure_scratchpad_directory(self, session_id: str) -> Path:
        scratchpad_directory = (self.runtime_config.scratchpad_root / session_id).resolve()
        scratchpad_directory.mkdir(parents=True, exist_ok=True)
        return scratchpad_directory

    def _append_file_history_replay_if_needed(
        self,
        session: AgentSessionState,
        file_history: tuple[dict[str, object], ...],
    ) -> None:
        if not file_history:
            return
        replay_count = len(file_history)
        unique_paths = sorted(
            {
                path
                for entry in file_history
                for path in (
                    entry.get('changed_paths')
                    if isinstance(entry.get('changed_paths'), list)
                    else ([entry.get('path')] if isinstance(entry.get('path'), str) else [])
                )
                if isinstance(path, str) and path
            }
        )
        snapshot_count = sum(
            1
            for entry in file_history
            for key in ('before_snapshot_id', 'after_snapshot_id')
            if isinstance(entry.get(key), str) and entry.get(key)
        )
        for message in reversed(session.messages):
            if message.metadata.get('kind') != 'file_history_replay':
                continue
            if message.metadata.get('file_history_count') == replay_count:
                return
            break
        session.append_user(
            self._render_file_history_replay(file_history),
            metadata={
                'kind': 'file_history_replay',
                'file_history_count': replay_count,
                'file_history_unique_paths': len(unique_paths),
                'file_history_snapshot_count': snapshot_count,
            },
            message_id=f'file_history_replay_{replay_count}',
        )

    def _render_file_history_replay(
        self,
        file_history: tuple[dict[str, object], ...],
    ) -> str:
        unique_paths = sorted(
            {
                path
                for entry in file_history
                for path in (
                    entry.get('changed_paths')
                    if isinstance(entry.get('changed_paths'), list)
                    else ([entry.get('path')] if isinstance(entry.get('path'), str) else [])
                )
                if isinstance(path, str) and path
            }
        )
        snapshot_count = sum(
            1
            for entry in file_history
            for key in ('before_snapshot_id', 'after_snapshot_id')
            if isinstance(entry.get(key), str) and entry.get(key)
        )
        lines = [
            '<system-reminder>',
            'Recent file history from this saved session:',
            f'- History entries: {len(file_history)}',
            f'- Unique changed paths: {len(unique_paths)}',
            f'- Snapshot ids: {snapshot_count}',
        ]
        if unique_paths:
            preview_paths = ', '.join(unique_paths[:4])
            if len(unique_paths) > 4:
                preview_paths += f', ... (+{len(unique_paths) - 4} more)'
            lines.append(f'- Changed path preview: {preview_paths}')
        for entry in file_history[-10:]:
            action = str(entry.get('action', entry.get('tool_name', 'tool')))
            turn = entry.get('turn_index')
            path = entry.get('path')
            command = entry.get('command')
            details = [f'action={action}']
            history_entry_id = entry.get('history_entry_id')
            if isinstance(history_entry_id, str) and history_entry_id:
                details.append(f'entry_id={history_entry_id}')
            if turn is not None:
                details.append(f'turn={turn}')
            if path:
                details.append(f'path={path}')
            if command:
                details.append(f'command={command}')
            child_session_ids = entry.get('child_session_ids')
            if isinstance(child_session_ids, list) and child_session_ids:
                details.append(f'child_sessions={len(child_session_ids)}')
            delegate_batch_count = entry.get('delegate_batch_count')
            if isinstance(delegate_batch_count, int) and not isinstance(delegate_batch_count, bool):
                details.append(f'batches={delegate_batch_count}')
            dependency_skips = entry.get('dependency_skips')
            if isinstance(dependency_skips, int) and not isinstance(dependency_skips, bool):
                details.append(f'dependency_skips={dependency_skips}')
            lines.append(f"- {'; '.join(details)}")
            before_snapshot_id = entry.get('before_snapshot_id')
            if isinstance(before_snapshot_id, str) and before_snapshot_id:
                lines.append(f'  before_snapshot: {before_snapshot_id}')
            after_snapshot_id = entry.get('after_snapshot_id')
            if isinstance(after_snapshot_id, str) and after_snapshot_id:
                lines.append(f'  after_snapshot: {after_snapshot_id}')
            before_preview = entry.get('before_preview')
            if isinstance(before_preview, str) and before_preview:
                lines.append(f'  before: {before_preview}')
            after_preview = entry.get('after_preview')
            if isinstance(after_preview, str) and after_preview:
                lines.append(f'  after: {after_preview}')
            result_preview = entry.get('result_preview')
            if isinstance(result_preview, str) and result_preview:
                lines.append(f'  result: {result_preview}')
        if len(file_history) > 10:
            lines.append(f'- ... plus {len(file_history) - 10} older file-history entries')
        lines.extend(
            [
                '',
                'Use this replayed history when continuing the task so you avoid repeating prior edits or commands.',
                '</system-reminder>',
            ]
        )
        return '\n'.join(lines)

    def _append_compaction_replay_if_needed(
        self,
        session: AgentSessionState,
    ) -> None:
        compact_messages = [
            message for message in session.messages
            if message.metadata.get('kind') == 'compact_boundary'
        ]
        snipped_messages = [
            message for message in session.messages
            if message.metadata.get('kind') == 'snipped_message'
        ]
        if not compact_messages and not snipped_messages:
            return
        for message in reversed(session.messages):
            if message.metadata.get('kind') != 'compaction_replay':
                continue
            return
        session.append_user(
            self._render_compaction_replay(compact_messages, snipped_messages),
            metadata={
                'kind': 'compaction_replay',
                'compact_boundary_count': len(compact_messages),
                'snipped_message_count': len(snipped_messages),
            },
            message_id=(
                f'compaction_replay_{len(compact_messages)}_{len(snipped_messages)}'
            ),
        )

    def _render_compaction_replay(
        self,
        compact_messages,
        snipped_messages,
    ) -> str:
        lines = [
            '<system-reminder>',
            'This resumed session already contains compacted or snipped history.',
            f'- Compact boundaries: {len(compact_messages)}',
            f'- Snipped/tombstoned messages: {len(snipped_messages)}',
        ]
        latest_boundary = compact_messages[-1] if compact_messages else None
        if latest_boundary is not None:
            lines.append(
                f"- Latest compact boundary id: {latest_boundary.message_id or '(none)'}"
            )
            depth = latest_boundary.metadata.get('compaction_depth')
            if isinstance(depth, int) and not isinstance(depth, bool):
                lines.append(f'- Latest compaction depth: {depth}')
            compacted_lineages = latest_boundary.metadata.get('compacted_lineage_ids')
            if isinstance(compacted_lineages, list) and compacted_lineages:
                lines.append(f'- Latest compacted lineages: {len(compacted_lineages)}')
            max_source_mutation_serial = latest_boundary.metadata.get('max_source_mutation_serial')
            if (
                isinstance(max_source_mutation_serial, int)
                and not isinstance(max_source_mutation_serial, bool)
                and max_source_mutation_serial > 0
            ):
                lines.append(
                    f'- Latest source mutation serial: {max_source_mutation_serial}'
                )
            source_mutation_totals = latest_boundary.metadata.get('source_mutation_totals')
            if isinstance(source_mutation_totals, dict) and source_mutation_totals:
                rendered = ', '.join(
                    f'{name}:{count}'
                    for name, count in sorted(source_mutation_totals.items())
                    if isinstance(name, str)
                    and name
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                )
                if rendered:
                    lines.append(f'- Latest compacted mutations: {rendered}')
            preserved_tail = latest_boundary.metadata.get('preserved_tail_ids')
            if isinstance(preserved_tail, list) and preserved_tail:
                lines.append(
                    '- Latest preserved tail ids: '
                    + ', '.join(str(item) for item in preserved_tail[:4])
                )
        if snipped_messages:
            last_ids = [
                message.message_id or '(none)'
                for message in snipped_messages[-3:]
            ]
            lines.append(f"- Recent snipped ids: {', '.join(last_ids)}")
            snipped_lineages = [
                str(message.metadata.get('snipped_from_lineage_id'))
                for message in snipped_messages[-3:]
                if isinstance(message.metadata.get('snipped_from_lineage_id'), str)
            ]
            if snipped_lineages:
                lines.append(f"- Recent snipped lineages: {', '.join(snipped_lineages)}")
        lines.extend(
            [
                '',
                'Use the surviving transcript plus the compacted summaries as the authoritative context when continuing.',
                '</system-reminder>',
            ]
        )
        return '\n'.join(lines)

    def _apply_hook_policy_before_prompt_hooks(self, prompt: str) -> str:
        if self.hook_policy_runtime is None or not self.hook_policy_runtime.manifests:
            return prompt
        injections = self.hook_policy_runtime.before_prompt_messages()
        managed_settings = self.hook_policy_runtime.managed_settings()
        safe_env = self.hook_policy_runtime.safe_env()
        trusted = self.hook_policy_runtime.is_trusted()
        if not injections and not managed_settings and not safe_env and trusted:
            return prompt
        lines = ['<system-reminder>', 'Workspace hook/policy guidance:']
        lines.append(
            f'- Trust mode: {"trusted" if trusted else "untrusted"}'
        )
        if not trusted:
            lines.append(
                '- Untrusted workspaces should favor inspection-first behavior. '
                'Avoid unnecessary writes or shell actions unless the task clearly requires them.'
            )
        for entry in injections:
            lines.append(f'- Before prompt: {entry}')
        if managed_settings:
            lines.append(
                '- Managed settings: '
                + ', '.join(f'{key}={value}' for key, value in sorted(managed_settings.items()))
            )
        if safe_env:
            lines.append(
                '- Safe environment values loaded for tools: '
                + ', '.join(sorted(safe_env))
            )
        lines.extend(['</system-reminder>', '', prompt])
        return '\n'.join(lines)

    def _build_plugin_tool_runtime_message(
        self,
        *,
        tool_name: str,
        preflight_messages: tuple[str, ...],
        block_message: str | None,
        plugin_messages: tuple[str, ...],
        hook_policy_preflight_messages: tuple[str, ...] = (),
        hook_policy_block_message: str | None = None,
        hook_policy_messages: tuple[str, ...] = (),
        delegate_preflight_messages: tuple[str, ...] = (),
        delegate_after_messages: tuple[str, ...] = (),
    ) -> str | None:
        if (
            block_message is None
            and not plugin_messages
            and not preflight_messages
            and hook_policy_block_message is None
            and not hook_policy_preflight_messages
            and not hook_policy_messages
            and not delegate_preflight_messages
            and not delegate_after_messages
        ):
            return None
        plugin_only = (
            hook_policy_block_message is None
            and not hook_policy_preflight_messages
            and not hook_policy_messages
        )
        lines = [
            '<system-reminder>',
            (
                f'Plugin tool runtime guidance for `{tool_name}`:'
                if plugin_only
                else f'Runtime tool guidance for `{tool_name}`:'
            ),
        ]
        for message in preflight_messages:
            lines.append(f'- Before tool: {message}')
        for message in hook_policy_preflight_messages:
            lines.append(f'- Hook/policy before tool: {message}')
        for message in delegate_preflight_messages:
            lines.append(f'- Before delegate: {message}')
        if block_message is not None:
            lines.append(f'- Blocked: {block_message}')
        if hook_policy_block_message is not None:
            lines.append(f'- Hook/policy blocked: {hook_policy_block_message}')
        for message in plugin_messages:
            lines.append(f'- After result: {message}')
        for message in hook_policy_messages:
            lines.append(f'- Hook/policy after result: {message}')
        for message in delegate_after_messages:
            lines.append(f'- After delegate: {message}')
        lines.extend(
            [
                '',
                'Use this runtime guidance when deciding the next tool call or assistant response.',
                '</system-reminder>',
            ]
        )
        return '\n'.join(lines)

    def _plugin_tool_preflight_messages(self, tool_name: str) -> tuple[str, ...]:
        if self.plugin_runtime is None:
            return ()
        return self.plugin_runtime.tool_preflight_injections(tool_name)

    def _plugin_block_message(self, tool_name: str) -> str | None:
        if self.plugin_runtime is None:
            return None
        return self.plugin_runtime.blocked_tool_message(tool_name)

    def _plugin_tool_result_messages(self, tool_name: str) -> tuple[str, ...]:
        if self.plugin_runtime is None:
            return ()
        return self.plugin_runtime.tool_result_injections(tool_name)

    def _hook_policy_tool_preflight_messages(self, tool_name: str) -> tuple[str, ...]:
        if self.hook_policy_runtime is None:
            return ()
        return self.hook_policy_runtime.before_tool_messages(tool_name)

    def _hook_policy_block_message(self, tool_name: str) -> str | None:
        if self.hook_policy_runtime is None:
            return None
        return self.hook_policy_runtime.denied_tool_message(tool_name)

    def _hook_policy_tool_result_messages(self, tool_name: str) -> tuple[str, ...]:
        if self.hook_policy_runtime is None:
            return ()
        return self.hook_policy_runtime.after_tool_messages(tool_name)

    def _persist_session(
        self,
        session: AgentSessionState,
        result: AgentRunResult,
    ) -> AgentRunResult:
        if result.session_id is None:
            return result
        persist_events = list(result.events)
        if self.plugin_runtime is not None:
            persist_messages = self.plugin_runtime.before_persist_injections()
            if persist_messages:
                session.append_user(
                    self._render_plugin_persist_message(persist_messages),
                    metadata={
                        'kind': 'plugin_persist',
                        'message_count': len(persist_messages),
                    },
                    message_id=f'plugin_persist_{result.session_id}',
                )
                persist_events.append(
                    {
                        'type': 'plugin_before_persist',
                        'session_id': result.session_id,
                        'message_count': len(persist_messages),
                    }
                )
        persisted_result, session_path = persist_agent_run(
            session=session,
            result=result,
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            plugin_state=(
                self.plugin_runtime.export_session_state()
                if self.plugin_runtime is not None
                else {}
            ),
        )
        self.last_session_path = session_path
        return replace(
            persisted_result,
            session_path=self.last_session_path,
            events=tuple(persist_events),
            transcript=session.transcript(),
        )

    def render_system_prompt(self) -> str:
        prompt_context = self.build_prompt_context()
        parts = self.build_system_prompt_parts(prompt_context)
        return render_system_prompt(parts)

    def render_context_report(self, prompt: str | None = None) -> str:
        session = self.last_session if prompt is None else None
        strategy = 'current Python session'
        if session is None:
            session = self.build_session(prompt)
            strategy = 'one-shot Python session preview'
        report = collect_context_usage(
            session=session,
            model=self.model_config.model,
            strategy=strategy,
        )
        return format_context_usage(report)

    def render_context_snapshot_report(self) -> str:
        prompt_context = self.build_prompt_context()
        return render_agent_context_report(prompt_context, self.model_config.model)

    def render_permissions_report(self) -> str:
        permissions = self.runtime_config.permissions
        lines = [
            '# Permissions',
            '',
            f'- File write tools: {"enabled" if permissions.allow_file_write else "disabled"}',
            f'- Shell commands: {"enabled" if permissions.allow_shell_commands else "disabled"}',
            f'- Destructive shell commands: {"enabled" if permissions.allow_destructive_shell_commands else "disabled"}',
        ]
        if self.hook_policy_runtime is not None and self.hook_policy_runtime.manifests:
            lines.append(
                f'- Workspace trust mode: {"trusted" if self.hook_policy_runtime.is_trusted() else "untrusted"}'
            )
            denied_tools = sorted(
                {
                    name
                    for manifest in self.hook_policy_runtime.manifests
                    for name in manifest.deny_tools
                }
            )
            if denied_tools:
                lines.append('- Policy-denied tools: ' + ', '.join(denied_tools))
        return '\n'.join(lines)

    def render_tools_report(self) -> str:
        permissions = self.runtime_config.permissions
        lines = ['# Tools', '']
        for tool in self.tool_registry.values():
            state = 'enabled'
            if tool.name == 'bash' and not permissions.allow_shell_commands:
                state = 'blocked by permissions'
            if tool.name in {'write_file', 'edit_file'} and not permissions.allow_file_write:
                state = 'blocked by permissions'
            if (
                self.hook_policy_runtime is not None
                and self.hook_policy_runtime.denied_tool_message(tool.name) is not None
            ):
                state = 'blocked by hook policy'
            lines.append(f'- `{tool.name}`: {tool.description} [{state}]')
        return '\n'.join(lines)

    def render_agents_report(self, *, show_all: bool = False) -> str:
        snapshot = self.load_agent_registry()
        return render_agents_report(
            snapshot,
            cwd=self.runtime_config.cwd,
            show_all=show_all,
        )

    def render_agent_detail_report(self, agent_type: str) -> str:
        snapshot = self.load_agent_registry()
        return render_agent_detail(snapshot, agent_type)

    def render_memory_report(self) -> str:
        prompt_context = self.build_prompt_context()
        claude_md = prompt_context.user_context.get('claudeMd')
        if not claude_md:
            return '# Memory\n\nNo CLAUDE.md memory files are currently loaded.'
        return '\n'.join(['# Memory', '', claude_md])

    def render_account_report(self, profile: str | None = None) -> str:
        if self.account_runtime is None:
            return '# Account\n\nNo local account runtime is available.'
        if profile:
            return self.account_runtime.render_profile(profile)
        return '\n'.join(['# Account', '', self.account_runtime.render_summary()])

    def render_search_report(
        self,
        query: str | None = None,
        *,
        provider: str | None = None,
        max_results: int | None = None,
        domains: tuple[str, ...] = (),
    ) -> str:
        if self.search_runtime is None:
            return (
                '# Search\n\nNo local search provider is available. '
                'Add a .claw-search.json or .claude/search.json manifest, '
                'or set SEARXNG_BASE_URL (no API key required), BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY.'
            )
        if not self.search_runtime.providers:
            return (
                '# Search\n\nNo local search provider is available. '
                'Add a .claw-search.json or .claude/search.json manifest, '
                'or set SEARXNG_BASE_URL (no API key required), BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY.'
            )
        if query:
            try:
                return self.search_runtime.render_search_results(
                    query,
                    provider_name=provider,
                    max_results=max_results,
                    domains=domains,
                    timeout_seconds=self.runtime_config.command_timeout_seconds,
                )
            except (KeyError, LookupError, OSError, ValueError) as exc:
                return f'# Search\n\nSearch failed: {exc}'
        if provider:
            return self.search_runtime.render_provider(provider)
        return '\n'.join(['# Search', '', self.search_runtime.render_summary()])

    def render_search_providers_report(self, query: str | None = None) -> str:
        if self.search_runtime is None or not self.search_runtime.has_search_runtime():
            return '# Search Providers\n\nNo local search providers discovered.'
        return self.search_runtime.render_providers_index(query=query)

    def render_search_activate_report(self, provider: str) -> str:
        if self.search_runtime is None or not self.search_runtime.providers:
            return '# Search\n\nNo local search provider is available.'
        try:
            report = self.search_runtime.activate_provider(provider)
        except KeyError:
            return f'# Search\n\nUnknown search provider: {provider}'
        clear_context_caches()
        self.tool_context = replace(
            self.tool_context,
            search_runtime=self.search_runtime,
        )
        return '\n'.join(['# Search', '', report.as_text()])

    def render_search_toggle_report(self, enabled: bool | None = None) -> str:
        if self.search_runtime is None:
            return '# Search\n\nNo local search runtime is available.'
        report = (
            self.search_runtime.toggle_search_enabled()
            if enabled is None
            else self.search_runtime.set_search_enabled(enabled)
        )
        clear_context_caches()
        self.tool_context = replace(
            self.tool_context,
            search_runtime=self.search_runtime,
        )
        return '\n'.join(['# Search', '', report.as_text()])

    def render_search_context_report(self, context_size: str | None = None) -> str:
        if self.search_runtime is None:
            return '# Search\n\nNo local search runtime is available.'
        if context_size is None:
            return '\n'.join(['# Search', '', self.search_runtime.render_summary()])
        try:
            report = self.search_runtime.set_context_size(context_size)
        except ValueError as exc:
            return f'# Search\n\n{exc}'
        clear_context_caches()
        self.tool_context = replace(
            self.tool_context,
            search_runtime=self.search_runtime,
        )
        return '\n'.join(['# Search', '', report.as_text()])

    def render_account_profiles_report(self, query: str | None = None) -> str:
        if self.account_runtime is None:
            return '# Account Profiles\n\nNo local account runtime is available.'
        return self.account_runtime.render_profiles_index(query=query)

    def render_account_login_report(
        self,
        target: str,
        *,
        provider: str | None = None,
        auth_mode: str | None = None,
    ) -> str:
        if self.account_runtime is None:
            return '# Account\n\nNo local account runtime is available.'
        report = self.account_runtime.login(target, provider=provider, auth_mode=auth_mode)
        clear_context_caches()
        return '\n'.join(['# Account', '', report.as_text()])

    def render_account_logout_report(self) -> str:
        if self.account_runtime is None:
            return '# Account\n\nNo local account runtime is available.'
        report = self.account_runtime.logout(reason='slash_or_cli_logout')
        clear_context_caches()
        return '\n'.join(['# Account', '', report.as_text()])

    def render_config_report(self) -> str:
        if self.config_runtime is None:
            return '# Config\n\nNo local config runtime is available.'
        return '\n'.join(['# Config', '', self.config_runtime.render_summary()])

    def render_lsp_report(self) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP\n\nNo local LSP runtime is available.'
        return '\n'.join(['# LSP', '', self.lsp_runtime.render_summary()])

    def render_lsp_document_symbols_report(self, file_path: str) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Document Symbols\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_document_symbols(file_path)
        except KeyError as exc:
            return f'# LSP Document Symbols\n\n{exc}'

    def render_lsp_workspace_symbols_report(
        self,
        query: str,
        *,
        max_results: int = 50,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Workspace Symbols\n\nNo local LSP runtime is available.'
        return self.lsp_runtime.render_workspace_symbols(query, max_results=max_results)

    def render_lsp_definition_report(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        max_results: int = 20,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Definition\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_definition(
                file_path,
                line,
                character,
                max_results=max_results,
            )
        except KeyError as exc:
            return f'# LSP Definition\n\n{exc}'

    def render_lsp_references_report(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        max_results: int = 50,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP References\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_references(
                file_path,
                line,
                character,
                max_results=max_results,
            )
        except KeyError as exc:
            return f'# LSP References\n\n{exc}'

    def render_lsp_hover_report(self, file_path: str, line: int, character: int) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Hover\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_hover(file_path, line, character)
        except KeyError as exc:
            return f'# LSP Hover\n\n{exc}'

    def render_lsp_diagnostics_report(self, file_path: str | None = None) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Diagnostics\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_diagnostics(file_path)
        except KeyError as exc:
            return f'# LSP Diagnostics\n\n{exc}'

    def render_lsp_prepare_call_hierarchy_report(
        self,
        file_path: str,
        line: int,
        character: int,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Call Hierarchy\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_prepare_call_hierarchy(file_path, line, character)
        except KeyError as exc:
            return f'# LSP Call Hierarchy\n\n{exc}'

    def render_lsp_incoming_calls_report(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        max_results: int = 50,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Incoming Calls\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_incoming_calls(
                file_path,
                line,
                character,
                max_results=max_results,
            )
        except KeyError as exc:
            return f'# LSP Incoming Calls\n\n{exc}'

    def render_lsp_outgoing_calls_report(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        max_results: int = 50,
    ) -> str:
        if self.lsp_runtime is None or not self.lsp_runtime.has_lsp_support():
            return '# LSP Outgoing Calls\n\nNo local LSP runtime is available.'
        try:
            return self.lsp_runtime.render_outgoing_calls(
                file_path,
                line,
                character,
                max_results=max_results,
            )
        except KeyError as exc:
            return f'# LSP Outgoing Calls\n\n{exc}'

    def render_config_effective_report(self) -> str:
        if self.config_runtime is None:
            return '# Config Effective\n\nNo local config runtime is available.'
        return '\n'.join(['# Config Effective', '', self.config_runtime.render_effective_config()])

    def render_config_source_report(self, source: str) -> str:
        if self.config_runtime is None:
            return '# Config Source\n\nNo local config runtime is available.'
        return '\n'.join(['# Config Source', '', self.config_runtime.render_source(source)])

    def render_config_value_report(self, key_path: str, source: str | None = None) -> str:
        if self.config_runtime is None:
            return '# Config Value\n\nNo local config runtime is available.'
        try:
            rendered = self.config_runtime.render_value(key_path, source=source)
        except KeyError as exc:
            label = source if source is not None else key_path
            return f'# Config Value\n\nUnknown config key or source: {label or exc.args[0]}'
        return '\n'.join(['# Config Value', '', rendered])

    def render_mcp_report(self, query: str | None = None) -> str:
        if self.mcp_runtime is None:
            return '# MCP\n\nNo local MCP manifests, servers, or resources discovered.'
        if query:
            return self.mcp_runtime.render_resource_index(query=query)
        return '\n'.join(['# MCP', '', self.mcp_runtime.render_summary()])

    def render_remote_report(self, target: str | None = None) -> str:
        if self.remote_runtime is None:
            return '# Remote\n\nNo local remote runtime is available.'
        if target:
            report = self.remote_runtime.connect(target)
            clear_context_caches()
            return '\n'.join(['# Remote', '', report.as_text()])
        return '\n'.join(['# Remote', '', self.remote_runtime.render_summary()])

    def render_remote_mode_report(self, target: str, *, mode: str) -> str:
        if self.remote_runtime is None:
            return '# Remote\n\nNo local remote runtime is available.'
        report = self.remote_runtime.connect(target, mode=mode)
        clear_context_caches()
        return '\n'.join(['# Remote', '', report.as_text()])

    def render_remote_profiles_report(self, query: str | None = None) -> str:
        if self.remote_runtime is None:
            return '# Remote Profiles\n\nNo local remote runtime is available.'
        return self.remote_runtime.render_profiles_index(query=query)

    def render_remote_disconnect_report(self) -> str:
        if self.remote_runtime is None:
            return '# Remote\n\nNo local remote runtime is available.'
        report = self.remote_runtime.disconnect()
        clear_context_caches()
        return '\n'.join(['# Remote', '', report.as_text()])

    def render_worktree_report(self) -> str:
        if self.worktree_runtime is None:
            return '# Worktree\n\nNo local worktree runtime is available.'
        return '\n'.join(['# Worktree', '', self.worktree_runtime.render_summary()])

    def render_worktree_enter_report(self, name: str | None = None) -> str:
        if self.worktree_runtime is None:
            return '# Worktree\n\nNo local worktree runtime is available.'
        try:
            report = self.worktree_runtime.enter(name=name)
        except (RuntimeError, ValueError) as exc:
            return f'# Worktree\n\n{exc}'
        self._apply_runtime_cwd_update(Path(report.worktree_path or self.runtime_config.cwd))
        return '\n'.join(['# Worktree', '', report.as_text()])

    def render_worktree_exit_report(
        self,
        *,
        action: str = 'keep',
        discard_changes: bool = False,
    ) -> str:
        if self.worktree_runtime is None:
            return '# Worktree\n\nNo local worktree runtime is available.'
        try:
            report = self.worktree_runtime.exit(
                action=action,
                discard_changes=discard_changes,
            )
        except (RuntimeError, ValueError) as exc:
            return f'# Worktree\n\n{exc}'
        target_cwd = report.original_cwd or report.current_cwd or str(self.runtime_config.cwd)
        self._apply_runtime_cwd_update(Path(target_cwd))
        return '\n'.join(['# Worktree', '', report.as_text()])

    def render_worktree_history_report(self) -> str:
        if self.worktree_runtime is None:
            return '# Worktree History\n\nNo local worktree runtime is available.'
        return self.worktree_runtime.render_history()

    def render_mcp_resources_report(self, query: str | None = None) -> str:
        if self.mcp_runtime is None:
            return '# MCP Resources\n\nNo local MCP manifests, servers, or resources discovered.'
        return self.mcp_runtime.render_resource_index(query=query)

    def render_mcp_resource_report(self, uri: str) -> str:
        if self.mcp_runtime is None:
            return '# MCP Resource\n\nNo local MCP manifests, servers, or resources discovered.'
        return self.mcp_runtime.render_resource(uri)

    def render_mcp_tools_report(
        self,
        query: str | None = None,
        *,
        server: str | None = None,
    ) -> str:
        if self.mcp_runtime is None:
            return '# MCP Tools\n\nNo local MCP manifests, servers, or resources discovered.'
        return self.mcp_runtime.render_tool_index(query=query, server_name=server)

    def render_mcp_call_tool_report(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        server: str | None = None,
    ) -> str:
        if self.mcp_runtime is None:
            return '# MCP Tool Result\n\nNo local MCP manifests, servers, or resources discovered.'
        try:
            return self.mcp_runtime.render_tool_call(
                tool_name,
                arguments=arguments,
                server_name=server,
            )
        except FileNotFoundError as exc:
            return f'# MCP Tool Result\n\n{exc}'

    def render_tasks_report(self, status: str | None = None) -> str:
        if self.task_runtime is None:
            return '# Tasks\n\nNo local task runtime is available.'
        return self.task_runtime.render_tasks(status=status)

    def render_next_tasks_report(self) -> str:
        if self.task_runtime is None:
            return '# Next Tasks\n\nNo local task runtime is available.'
        return self.task_runtime.render_next_tasks()

    def render_plan_report(self) -> str:
        if self.plan_runtime is None:
            return '# Plan\n\nNo local plan runtime is available.'
        return self.plan_runtime.render_plan()

    def render_task_report(self, task_id: str) -> str:
        if self.task_runtime is None:
            return '# Task\n\nNo local task runtime is available.'
        return self.task_runtime.render_task(task_id)

    def render_ask_user_report(self) -> str:
        if self.ask_user_runtime is None:
            return '# Ask User\n\nNo local ask-user runtime is available.'
        return '\n'.join(['# Ask User', '', self.ask_user_runtime.render_summary()])

    def render_ask_user_history_report(self) -> str:
        if self.ask_user_runtime is None:
            return '# Ask User History\n\nNo local ask-user runtime is available.'
        return self.ask_user_runtime.render_history()

    def render_teams_report(self, query: str | None = None) -> str:
        if self.team_runtime is None:
            return '# Teams\n\nNo local team runtime is available.'
        return self.team_runtime.render_teams_index(query=query)

    def render_team_report(self, team_name: str) -> str:
        if self.team_runtime is None:
            return '# Team\n\nNo local team runtime is available.'
        try:
            return self.team_runtime.render_team(team_name)
        except KeyError:
            return f'# Team\n\nUnknown team: {team_name}'

    def render_team_messages_report(self, team_name: str | None = None) -> str:
        if self.team_runtime is None:
            return '# Team Messages\n\nNo local team runtime is available.'
        try:
            return self.team_runtime.render_messages(team_name=team_name)
        except KeyError:
            return f'# Team Messages\n\nUnknown team: {team_name}'

    def render_workflows_report(self, query: str | None = None) -> str:
        if self.workflow_runtime is None or not self.workflow_runtime.has_workflows():
            return '# Workflows\n\nNo local workflow runtime is available.'
        return self.workflow_runtime.render_workflows_index(query=query)

    def render_workflow_report(self, workflow_name: str) -> str:
        if self.workflow_runtime is None or not self.workflow_runtime.has_workflows():
            return '# Workflow\n\nNo local workflow runtime is available.'
        try:
            return self.workflow_runtime.render_workflow(workflow_name)
        except KeyError:
            return f'# Workflow\n\nUnknown workflow: {workflow_name}'

    def render_workflow_run_report(
        self,
        workflow_name: str,
        *,
        arguments: dict[str, Any] | None = None,
    ) -> str:
        if self.workflow_runtime is None or not self.workflow_runtime.has_workflows():
            return '# Workflow Run\n\nNo local workflow runtime is available.'
        try:
            return self.workflow_runtime.render_run_report(
                workflow_name,
                arguments=arguments,
            )
        except KeyError:
            return f'# Workflow Run\n\nUnknown workflow: {workflow_name}'

    def render_remote_triggers_report(self, query: str | None = None) -> str:
        if self.remote_trigger_runtime is None or not self.remote_trigger_runtime.has_state():
            return '# Remote Triggers\n\nNo local remote trigger runtime is available.'
        return self.remote_trigger_runtime.render_trigger_index(query=query)

    def render_remote_trigger_report(self, trigger_id: str) -> str:
        if self.remote_trigger_runtime is None or not self.remote_trigger_runtime.has_state():
            return '# Remote Trigger\n\nNo local remote trigger runtime is available.'
        try:
            return self.remote_trigger_runtime.render_trigger(trigger_id)
        except KeyError:
            return f'# Remote Trigger\n\nUnknown remote trigger: {trigger_id}'

    def render_remote_trigger_action_report(
        self,
        action: str,
        *,
        trigger_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> str:
        if self.remote_trigger_runtime is None:
            return '# Remote Trigger\n\nNo local remote trigger runtime is available.'
        normalized = action.strip().lower()
        try:
            if normalized == 'list':
                return self.remote_trigger_runtime.render_trigger_index()
            if normalized == 'get':
                if not trigger_id:
                    return '# Remote Trigger\n\ntrigger_id is required for get'
                return self.remote_trigger_runtime.render_trigger(trigger_id)
            if normalized == 'create':
                created = self.remote_trigger_runtime.create_trigger(body or {})
                return self.remote_trigger_runtime.render_trigger(created.trigger_id)
            if normalized == 'update':
                if not trigger_id:
                    return '# Remote Trigger\n\ntrigger_id is required for update'
                updated = self.remote_trigger_runtime.update_trigger(trigger_id, body or {})
                return self.remote_trigger_runtime.render_trigger(updated.trigger_id)
            if normalized == 'run':
                if not trigger_id:
                    return '# Remote Trigger Run\n\ntrigger_id is required for run'
                return self.remote_trigger_runtime.render_run_report(trigger_id, body=body)
        except (KeyError, TypeError, ValueError) as exc:
            return f'# Remote Trigger\n\n{exc}'
        return '# Remote Trigger\n\naction must be one of list, get, create, update, or run'

    def render_hook_policy_report(self) -> str:
        if self.hook_policy_runtime is None:
            return '# Hook Policy\n\nNo local hook or policy manifests discovered.'
        return '\n'.join(['# Hook Policy', '', self.hook_policy_runtime.render_summary()])

    def render_trust_report(self) -> str:
        trusted = True
        settings: dict[str, Any] = {}
        env_values: dict[str, str] = {}
        if self.hook_policy_runtime is not None:
            trusted = self.hook_policy_runtime.is_trusted()
            settings = self.hook_policy_runtime.managed_settings()
            env_values = self.hook_policy_runtime.safe_env()
        lines = [
            '# Trust',
            '',
            f'- Workspace trust mode: {"trusted" if trusted else "untrusted"}',
        ]
        if settings:
            lines.append('- Managed settings:')
            lines.extend(f'  - {key}={value}' for key, value in sorted(settings.items()))
        if env_values:
            lines.append('- Safe environment values:')
            lines.extend(f'  - {key}={value}' for key, value in sorted(env_values.items()))
        return '\n'.join(lines)

    def render_status_report(self) -> str:
        token_counter = describe_token_counter(self.model_config.model)
        lines = [
            '# Status',
            '',
            f'- Model: {self.model_config.model}',
            f'- LLM backend: {self.llm_backend}',
            f'- Token counter: {token_counter.backend} ({token_counter.source})',
            f'- Registered tools: {len(self.tool_registry)}',
            f'- Streaming model responses: {self.runtime_config.stream_model_responses}',
            f'- Session ID: {self.active_session_id or "none"}',
            f'- Last session loaded: {"yes" if self.last_session is not None else "no"}',
        ]
        if self.last_session is not None:
            token_budget = calculate_token_budget(
                session=self.last_session,
                model=self.model_config.model,
                budget_config=self.runtime_config.budget_config,
                output_schema=self.runtime_config.output_schema,
            )
            lines.append(
                f'- Prompt budget: {token_budget.projected_input_tokens:,} / {token_budget.soft_input_limit_tokens:,} soft'
            )
            lines.append(
                f'- Prompt hard limit: {token_budget.hard_input_limit_tokens:,}'
            )
        if self.hook_policy_runtime is not None and self.hook_policy_runtime.manifests:
            lines.append(
                f'- Workspace trust mode: {"trusted" if self.hook_policy_runtime.is_trusted() else "untrusted"}'
            )
        if self.mcp_runtime is not None:
            if self.mcp_runtime.resources:
                lines.append(f'- MCP local resources: {len(self.mcp_runtime.resources)}')
            if self.mcp_runtime.servers:
                lines.append(f'- MCP servers: {len(self.mcp_runtime.servers)}')
        if self.remote_runtime is not None and self.remote_runtime.has_remote_config():
            lines.append(f'- Remote profiles: {len(self.remote_runtime.profiles)}')
            if self.remote_runtime.active_connection is not None:
                connection = self.remote_runtime.active_connection
                lines.append(
                    f'- Active remote: {connection.mode} -> {connection.target}'
                )
        if self.search_runtime is not None:
            lines.append(f'- Web search enabled: {self.search_runtime.web_search_enabled}')
            lines.append(
                f'- Web search context size: {self.search_runtime.web_search_context_size}'
            )
            lines.append(
                f'- Web search default max results: {self.search_runtime.default_max_results}'
            )
            lines.append(f'- Search providers: {len(self.search_runtime.providers)}')
            active_provider = self.search_runtime.current_provider()
            if active_provider is not None:
                lines.append(
                    f'- Active search provider: {active_provider.name} ({active_provider.provider})'
                )
        if self.account_runtime is not None and self.account_runtime.has_account_state():
            lines.append(f'- Account profiles: {len(self.account_runtime.profiles)}')
            if self.account_runtime.active_session is not None:
                session = self.account_runtime.active_session
                lines.append(
                    f'- Active account: {session.provider} -> {session.identity}'
                )
        if self.ask_user_runtime is not None and self.ask_user_runtime.has_state():
            lines.append(f'- Ask-user queued answers: {len(self.ask_user_runtime.queued_answers)}')
            lines.append(f'- Ask-user history: {len(self.ask_user_runtime.history)}')
        if self.config_runtime is not None and self.config_runtime.has_config():
            lines.append(f'- Config sources: {len(self.config_runtime.sources)}')
            lines.append(
                f'- Effective config keys: {len(self.config_runtime.list_keys())}'
            )
        if self.lsp_runtime is not None and self.lsp_runtime.has_lsp_support():
            lines.append(
                f'- LSP indexed files: {len(self.lsp_runtime._workspace_files())}'
            )
        if self.plan_runtime is not None and self.plan_runtime.steps:
            lines.append(f'- Local plan steps: {len(self.plan_runtime.steps)}')
        if self.task_runtime is not None and self.task_runtime.tasks:
            lines.append(f'- Local tasks: {len(self.task_runtime.tasks)}')
        if self.team_runtime is not None and self.team_runtime.has_team_state():
            lines.append(f'- Local teams: {len(self.team_runtime.teams)}')
            lines.append(f'- Team messages: {len(self.team_runtime.messages)}')
        if self.last_session_path is not None:
            lines.append(f'- Session path: {self.last_session_path}')
        if self.last_run_result is not None:
            lines.extend(
                [
                    f'- Last run turns: {self.last_run_result.turns}',
                    f'- Last run tool calls: {self.last_run_result.tool_calls}',
                    f'- Last run total tokens: {self.last_run_result.usage.total_tokens}',
                    f'- Last run total cost: ${self.last_run_result.total_cost_usd:.6f}',
                ]
            )
            if self.last_run_result.scratchpad_directory is not None:
                lines.append(
                    f'- Scratchpad directory: {self.last_run_result.scratchpad_directory}'
                )
        else:
            lines.append('- Last run: none')
        if self.agent_manager is not None:
            lines.extend(self.agent_manager.summary_lines())
        return '\n'.join(lines)

    def render_token_budget_report(self) -> str:
        session = self.last_session or self.build_session()
        snapshot = calculate_token_budget(
            session=session,
            model=self.model_config.model,
            budget_config=self.runtime_config.budget_config,
            output_schema=self.runtime_config.output_schema,
        )
        return format_token_budget(snapshot)

    def _finalize_managed_agent(self, result: AgentRunResult) -> None:
        if self.managed_agent_id is None or self.agent_manager is None:
            self.resume_source_session_id = None
            return
        self.agent_manager.finish_agent(
            self.managed_agent_id,
            session_id=result.session_id,
            session_path=result.session_path,
            turns=result.turns,
            tool_calls=result.tool_calls,
            stop_reason=result.stop_reason,
        )
        self.resume_source_session_id = None

    def _accumulate_usage(self, result: AgentRunResult) -> None:
        """Add a run's usage to the cumulative session totals."""
        self.cumulative_usage = self.cumulative_usage + result.usage
        self.cumulative_cost_usd += result.total_cost_usd

    def _refresh_runtime_views_for_tool_result(
        self,
        tool_name: str,
        tool_result: ToolExecutionResult,
    ) -> None:
        if not tool_result.ok:
            return
        cwd_update = tool_result.metadata.get('cwd_update')
        if isinstance(cwd_update, str) and cwd_update:
            self._apply_runtime_cwd_update(Path(cwd_update))
        refresh_tool_names = {
            'update_plan',
            'plan_clear',
            'task_create',
            'task_update',
            'task_start',
            'task_complete',
            'task_block',
            'task_cancel',
            'todo_write',
            'search_activate_provider',
            'remote_connect',
            'remote_disconnect',
            'account_login',
            'account_logout',
            'config_set',
            'ask_user_question',
            'team_create',
            'team_delete',
            'send_message',
            'workflow_run',
            'remote_trigger',
            'worktree_enter',
            'worktree_exit',
        }
        if tool_name not in refresh_tool_names:
            return
        clear_context_caches()
        additional_dirs = tuple(
            str(path) for path in self.runtime_config.additional_working_directories
        )
        if tool_name.startswith('remote_'):
            self.remote_runtime = RemoteRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name == 'remote_trigger':
            self.remote_trigger_runtime = RemoteTriggerRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name.startswith('search_'):
            self.search_runtime = SearchRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name.startswith('account_'):
            self.account_runtime = AccountRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name == 'ask_user_question':
            self.ask_user_runtime = AskUserRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name == 'config_set':
            self.config_runtime = ConfigRuntime.from_workspace(self.runtime_config.cwd)
        if tool_name.startswith('task_') or tool_name == 'todo_write':
            self.task_runtime = TaskRuntime.from_workspace(self.runtime_config.cwd)
        if tool_name.startswith('plan_') or tool_name == 'update_plan':
            self.plan_runtime = PlanRuntime.from_workspace(self.runtime_config.cwd)
        if tool_name.startswith('team_') or tool_name == 'send_message':
            self.team_runtime = TeamRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name.startswith('workflow_'):
            self.workflow_runtime = WorkflowRuntime.from_workspace(
                self.runtime_config.cwd,
                additional_working_directories=additional_dirs,
            )
        if tool_name.startswith('worktree_'):
            self.worktree_runtime = WorktreeRuntime.from_workspace(self.runtime_config.cwd)
        self.tool_context = replace(
            self.tool_context,
            tool_registry=self.tool_registry,
            search_runtime=self.search_runtime,
            account_runtime=self.account_runtime,
            ask_user_runtime=self.ask_user_runtime,
            config_runtime=self.config_runtime,
            lsp_runtime=self.lsp_runtime,
            remote_runtime=self.remote_runtime,
            remote_trigger_runtime=self.remote_trigger_runtime,
            plan_runtime=self.plan_runtime,
            task_runtime=self.task_runtime,
            team_runtime=self.team_runtime,
            workflow_runtime=self.workflow_runtime,
            worktree_runtime=self.worktree_runtime,
        )

    def _apply_runtime_cwd_update(self, new_cwd: Path) -> None:
        resolved_cwd = new_cwd.resolve()
        if resolved_cwd == self.runtime_config.cwd.resolve():
            return
        self.runtime_config = replace(self.runtime_config, cwd=resolved_cwd)
        clear_context_caches()
        additional_dirs = tuple(
            str(path) for path in self.runtime_config.additional_working_directories
        )
        self.plugin_runtime = PluginRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.hook_policy_runtime = HookPolicyRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.mcp_runtime = MCPRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.remote_runtime = RemoteRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.remote_trigger_runtime = RemoteTriggerRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.search_runtime = SearchRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.account_runtime = AccountRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.ask_user_runtime = AskUserRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.config_runtime = ConfigRuntime.from_workspace(self.runtime_config.cwd)
        self.lsp_runtime = LSPRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.task_runtime = TaskRuntime.from_workspace(self.runtime_config.cwd)
        self.plan_runtime = PlanRuntime.from_workspace(self.runtime_config.cwd)
        self.team_runtime = TeamRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.workflow_runtime = WorkflowRuntime.from_workspace(
            self.runtime_config.cwd,
            additional_dirs,
        )
        self.worktree_runtime = WorktreeRuntime.from_workspace(self.runtime_config.cwd)
        self.runtime_config = self._apply_hook_policy_budget_overrides(self.runtime_config)
        registry = dict(default_tool_registry())
        if self.plugin_runtime is not None:
            alias_tools = self.plugin_runtime.register_tool_aliases(registry)
            if alias_tools:
                registry = {**registry, **alias_tools}
            virtual_tools = self.plugin_runtime.register_virtual_tools(registry)
            if virtual_tools:
                registry = {**registry, **virtual_tools}
        self.tool_registry = registry
        self.tool_context = build_tool_context(
            self.runtime_config,
            tool_registry=self.tool_registry,
            extra_env=(
                self.hook_policy_runtime.safe_env()
                if self.hook_policy_runtime is not None
                else None
            ),
            search_runtime=self.search_runtime,
            account_runtime=self.account_runtime,
            ask_user_runtime=self.ask_user_runtime,
            config_runtime=self.config_runtime,
            lsp_runtime=self.lsp_runtime,
            mcp_runtime=self.mcp_runtime,
            remote_runtime=self.remote_runtime,
            remote_trigger_runtime=self.remote_trigger_runtime,
            plan_runtime=self.plan_runtime,
            task_runtime=self.task_runtime,
            team_runtime=self.team_runtime,
            workflow_runtime=self.workflow_runtime,
            worktree_runtime=self.worktree_runtime,
        )

    def _apply_plugin_before_prompt_hooks(self, prompt: str) -> str:
        if self.plugin_runtime is None:
            return prompt
        injections = self.plugin_runtime.before_prompt_injections()
        state_reminder = self.plugin_runtime.runtime_state_reminder()
        if not injections and not state_reminder:
            return prompt
        lines = ['<system-reminder>', 'Plugin before-prompt hooks:']
        lines.extend(f'- {entry}' for entry in injections)
        if state_reminder:
            lines.extend(['', state_reminder])
        lines.extend(['</system-reminder>', '', prompt])
        return '\n'.join(lines)

    def _apply_plugin_resume_hooks(
        self,
        prompt: str,
        *,
        resumed: bool,
    ) -> str:
        if not resumed or self.plugin_runtime is None:
            return prompt
        injections = self.plugin_runtime.on_resume_injections()
        if not injections:
            return prompt
        lines = ['<system-reminder>', 'Plugin resume hooks:']
        lines.extend(f'- {entry}' for entry in injections)
        lines.extend(['</system-reminder>', '', prompt])
        return '\n'.join(lines)

    def _render_plugin_persist_message(
        self,
        messages: tuple[str, ...],
    ) -> str:
        lines = ['<system-reminder>', 'Plugin persist hooks:']
        lines.extend(f'- {entry}' for entry in messages)
        lines.extend(
            [
                '',
                'This session state was persisted with plugin lifecycle guidance.',
                '</system-reminder>',
            ]
        )
        return '\n'.join(lines)

    def _append_plugin_after_turn_events(
        self,
        result: AgentRunResult,
        *,
        prompt: str,
        turn_index: int,
    ) -> AgentRunResult:
        if self.plugin_runtime is None:
            return result
        injections = self.plugin_runtime.after_turn_injections()
        if not injections:
            return result
        appended = list(result.events)
        for entry in injections:
            appended.append(
                {
                    'type': 'plugin_after_turn',
                    'turn_index': turn_index,
                    'message': entry,
                    'prompt_preview': self._preview_text(prompt, 120),
                    'stop_reason': result.stop_reason,
                }
            )
        return replace(result, events=tuple(appended))

    def _append_runtime_after_turn_events(
        self,
        result: AgentRunResult,
        *,
        prompt: str,
        turn_index: int,
    ) -> AgentRunResult:
        updated = self._append_plugin_after_turn_events(
            result,
            prompt=prompt,
            turn_index=turn_index,
        )
        if self.hook_policy_runtime is None:
            return updated
        injections = self.hook_policy_runtime.after_turn_messages()
        if not injections:
            return updated
        appended = list(updated.events)
        for entry in injections:
            appended.append(
                {
                    'type': 'hook_policy_after_turn',
                    'turn_index': turn_index,
                    'message': entry,
                    'prompt_preview': self._preview_text(prompt, 120),
                    'stop_reason': updated.stop_reason,
                    'trusted': self.hook_policy_runtime.is_trusted(),
                }
            )
        return replace(updated, events=tuple(appended))


def _optional_policy_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_policy_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
