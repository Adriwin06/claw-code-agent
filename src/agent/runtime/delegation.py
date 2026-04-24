from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.agent.tools.execution import AgentTool
from src.agent.models.types import (
    AgentPermissions,
    AgentRunResult,
    ModelConfig,
    ToolExecutionResult,
)
from src.agent.profiles.builtin import (
    ALL_AGENT_DISALLOWED_TOOLS,
    GENERAL_PURPOSE_AGENT,
    AgentDefinition,
)
from src.agent.context.constants import CLAUDE_MODEL_IDS
from src.agent.profiles.registry import find_agent_definition
from src.session.session_store import load_agent_session


def resolve_agent_definition(agent: Any, arguments: dict[str, object]) -> AgentDefinition:
    subagent_type = arguments.get('subagent_type')
    if isinstance(subagent_type, str) and subagent_type:
        agent_def = find_agent_definition(agent.runtime_config.cwd, subagent_type)
        if agent_def is not None:
            return agent_def
    return GENERAL_PURPOSE_AGENT


def resolve_child_model_config(
    agent: Any,
    arguments: dict[str, object],
    agent_def: AgentDefinition,
) -> ModelConfig:
    model_override = arguments.get('model')
    agent_model = agent_def.model
    if isinstance(model_override, str) and model_override.strip():
        return replace(agent.model_config, model=model_override.strip())
    if agent_model and agent_model != 'inherit':
        return replace(
            agent.model_config,
            model=resolve_agent_model_override(agent, agent_model),
        )
    return agent.model_config


def resolve_agent_model_override(agent: Any, agent_model: str) -> str:
    normalized = agent_model.strip()
    if not normalized:
        return agent.model_config.model
    canonical_claude_model = CLAUDE_MODEL_IDS.get(normalized.lower())
    if canonical_claude_model is None:
        return normalized
    parent_model = agent.model_config.model.strip()
    if not parent_model:
        return canonical_claude_model
    if '/' not in parent_model:
        return canonical_claude_model if 'claude' in parent_model.lower() else parent_model
    provider, _ = parent_model.split('/', 1)
    if provider.lower() == 'anthropic':
        return f'{provider}/{canonical_claude_model}'
    return parent_model


def filter_tools_for_agent(
    agent: Any,
    agent_def: AgentDefinition,
) -> dict[str, AgentTool]:
    base_tools = {
        name: tool
        for name, tool in agent.tool_registry.items()
        if name not in ('delegate_agent', 'Agent')
    }
    if agent_def.tools is not None:
        allowed = set(agent_def.tools)
        base_tools = {
            name: tool
            for name, tool in base_tools.items()
            if name in allowed
        }
    if agent_def.disallowed_tools:
        denied = set(agent_def.disallowed_tools)
        base_tools = {
            name: tool
            for name, tool in base_tools.items()
            if name not in denied
        }
    return {
        name: tool
        for name, tool in base_tools.items()
        if name not in ALL_AGENT_DISALLOWED_TOOLS
    }


def execute_delegate_agent(
    agent: Any,
    arguments: dict[str, object],
    *,
    tool_name: str = 'Agent',
) -> ToolExecutionResult:
    agent_def = resolve_agent_definition(agent, arguments)
    max_turns = arguments.get('max_turns')
    if max_turns is not None and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1
    ):
        return ToolExecutionResult(
            name=tool_name,
            ok=False,
            content='max_turns must be an integer >= 1',
        )
    subtasks = normalize_delegate_subtasks(arguments)
    if not subtasks:
        return ToolExecutionResult(
            name=tool_name,
            ok=False,
            content='prompt must be a non-empty string or subtasks must contain at least one prompt',
        )

    if agent_def.disallowed_tools and (
        'edit_file' in agent_def.disallowed_tools
        or 'write_file' in agent_def.disallowed_tools
    ):
        child_permissions = AgentPermissions(
            allow_file_write=False,
            allow_shell_commands=agent.runtime_config.permissions.allow_shell_commands,
            allow_destructive_shell_commands=False,
        )
    else:
        child_permissions = AgentPermissions(
            allow_file_write=(
                agent.runtime_config.permissions.allow_file_write
                and bool(arguments.get('allow_write', False))
            ),
            allow_shell_commands=(
                agent.runtime_config.permissions.allow_shell_commands
                and bool(arguments.get('allow_shell', False))
            ),
            allow_destructive_shell_commands=False,
        )

    parent_max_turns = agent.runtime_config.max_turns
    fallback_max_turns = None if parent_max_turns is None else min(parent_max_turns, 6)
    effective_max_turns = max_turns or agent_def.max_turns or fallback_max_turns

    child_runtime_config = replace(
        agent.runtime_config,
        max_turns=effective_max_turns,
        permissions=child_permissions,
        auto_compact_threshold_tokens=agent.runtime_config.auto_compact_threshold_tokens,
    )

    child_model_config = resolve_child_model_config(agent, arguments, agent_def)
    child_tools = filter_tools_for_agent(agent, agent_def)
    include_parent_context = bool(arguments.get('include_parent_context', True))
    continue_on_error = bool(arguments.get('continue_on_error', True))
    max_failures = arguments.get('max_failures')
    if isinstance(max_failures, bool) or (
        max_failures is not None and not isinstance(max_failures, int)
    ):
        max_failures = None
    if isinstance(max_failures, int) and max_failures < 0:
        max_failures = None
    strategy = normalize_delegate_strategy(arguments.get('strategy'))
    child_summaries: list[dict[str, object]] = []
    child_session_ids: list[str] = []
    prior_results: list[dict[str, str]] = []
    completed_labels: set[str] = set()
    failed_labels: set[str] = set()
    delegate_preflight_messages = (
        agent.plugin_runtime.before_delegate_injections()
        if agent.plugin_runtime is not None
        else ()
    )
    delegate_after_messages: tuple[str, ...] = ()
    group_id: str | None = None
    if agent.agent_manager is not None and len(subtasks) > 1:
        group_id = agent.agent_manager.start_group(
            label=str(arguments.get('label') or 'delegated_group'),
            parent_agent_id=agent.managed_agent_id,
            strategy=strategy,
        )
    planned_batches = plan_delegate_batches(subtasks, strategy)
    batch_summaries: list[dict[str, object]] = []
    failed_children = 0
    dependency_skips = 0
    child_result: AgentRunResult | None = None
    stop_processing = False
    for batch_index, batch in enumerate(planned_batches, start=1):
        if stop_processing:
            break
        batch_completed = 0
        batch_failed = 0
        batch_skipped = 0
        batch_labels: list[str] = []
        for subtask in batch:
            index = int(subtask.get('_delegate_index', len(child_summaries) + 1))
            subtask_label = str(subtask.get('label') or f'subtask_{index}')
            batch_labels.append(subtask_label)
            dependencies = tuple(
                item for item in subtask.get('depends_on', ()) if isinstance(item, str) and item
            )
            unmet_dependencies = [
                dependency for dependency in dependencies if dependency not in completed_labels
            ]
            blocked_dependencies = [
                dependency for dependency in dependencies if dependency in failed_labels
            ]
            if unmet_dependencies:
                skip_reason = (
                    'skipped_dependency'
                    if blocked_dependencies
                    else 'pending_dependency'
                )
                child_result = AgentRunResult(
                    final_output=(
                        'Skipped delegated subtask because dependencies were not satisfied: '
                        + ', '.join(unmet_dependencies)
                    ),
                    turns=0,
                    tool_calls=0,
                    transcript=(),
                    stop_reason=skip_reason,
                )
                summary = {
                    'index': index,
                    'label': subtask_label,
                    'session_id': '',
                    'turns': child_result.turns,
                    'tool_calls': child_result.tool_calls,
                    'stop_reason': skip_reason,
                    'output_preview': agent._preview_text(child_result.final_output, 220),
                    'resume_used': False,
                    'resumed_from_session_id': '',
                    'depends_on': list(dependencies),
                    'batch_index': batch_index,
                }
                child_summaries.append(summary)
                failed_children += 1
                batch_failed += 1
                batch_skipped += 1
                dependency_skips += 1
                failed_labels.add(subtask_label)
                if isinstance(max_failures, int) and failed_children > max_failures:
                    stop_processing = True
                    break
                if not continue_on_error:
                    stop_processing = True
                    break
                continue

            child_system_prompt = agent_def.system_prompt or agent.custom_system_prompt
            child_override_prompt = None
            if agent_def.system_prompt:
                child_override_prompt = agent_def.system_prompt
            else:
                child_override_prompt = agent.override_system_prompt

            child_append_prompt = agent.append_system_prompt
            if agent_def.critical_system_reminder:
                reminder = (
                    f'\n\n<system-reminder>\n{agent_def.critical_system_reminder}\n</system-reminder>'
                )
                child_append_prompt = (child_append_prompt or '') + reminder

            child_agent = agent.__class__(
                model_config=child_model_config,
                runtime_config=replace(
                    child_runtime_config,
                    max_turns=subtask.get('max_turns', child_runtime_config.max_turns),
                    disable_claude_md_discovery=agent_def.omit_claude_md,
                ),
                custom_system_prompt=child_system_prompt if not child_override_prompt else None,
                append_system_prompt=child_append_prompt,
                override_system_prompt=child_override_prompt,
                tool_registry=child_tools,
                agent_manager=agent.agent_manager,
                parent_agent_id=agent.managed_agent_id,
                managed_group_id=group_id,
                managed_child_index=index,
                managed_label=subtask_label,
            )
            if group_id is not None and child_agent.managed_agent_id is not None:
                agent.agent_manager.register_group_child(
                    group_id,
                    child_agent.managed_agent_id,
                    child_index=index,
                )
            resume_session_id = subtask.get('resume_session_id')
            child_prompt = str(subtask['prompt'])
            if agent_def.initial_prompt and not (
                isinstance(resume_session_id, str) and resume_session_id
            ):
                child_prompt = f'{agent_def.initial_prompt.strip()}\n\n{child_prompt}'.strip()
            if delegate_preflight_messages:
                child_prompt = prepend_plugin_delegate_context(
                    child_prompt,
                    delegate_preflight_messages,
                )
            if include_parent_context and prior_results:
                child_prompt = prepend_delegate_context(child_prompt, prior_results)
            resume_used = False
            if isinstance(resume_session_id, str) and resume_session_id:
                try:
                    stored_child_session = load_agent_session(
                        resume_session_id,
                        directory=child_runtime_config.session_directory,
                    )
                except OSError:
                    child_result = AgentRunResult(
                        final_output=f'Unable to load delegated session {resume_session_id}.',
                        turns=0,
                        tool_calls=0,
                        transcript=(),
                        stop_reason='resume_load_error',
                        session_id=resume_session_id,
                    )
                    failed_children += 1
                    batch_failed += 1
                    summary = {
                        'index': index,
                        'label': subtask_label,
                        'session_id': resume_session_id,
                        'turns': child_result.turns,
                        'tool_calls': child_result.tool_calls,
                        'stop_reason': child_result.stop_reason or 'resume_load_error',
                        'output_preview': agent._preview_text(child_result.final_output, 220),
                        'resume_used': True,
                        'resumed_from_session_id': resume_session_id,
                        'depends_on': list(dependencies),
                        'batch_index': batch_index,
                    }
                    child_summaries.append(summary)
                    prior_results.append(
                        {
                            'label': summary['label'],
                            'output_preview': str(summary['output_preview']),
                        }
                    )
                    failed_labels.add(subtask_label)
                    if isinstance(max_failures, int) and failed_children > max_failures:
                        stop_processing = True
                        break
                    if not continue_on_error:
                        stop_processing = True
                        break
                    continue
                child_result = child_agent.resume(child_prompt, stored_child_session)
                resume_used = True
            else:
                child_result = child_agent.run(child_prompt)
            if group_id is not None and child_agent.managed_agent_id is not None:
                agent.agent_manager.register_group_child(
                    group_id,
                    child_agent.managed_agent_id,
                    child_index=index,
                )
            summary = {
                'index': index,
                'label': subtask_label,
                'session_id': child_result.session_id or '',
                'turns': child_result.turns,
                'tool_calls': child_result.tool_calls,
                'stop_reason': child_result.stop_reason or 'stop',
                'output_preview': agent._preview_text(child_result.final_output, 220),
                'resume_used': resume_used,
                'resumed_from_session_id': (
                    str(resume_session_id)
                    if isinstance(resume_session_id, str) and resume_session_id
                    else ''
                ),
                'depends_on': list(dependencies),
                'batch_index': batch_index,
            }
            child_summaries.append(summary)
            if child_result.session_id:
                child_session_ids.append(child_result.session_id)
            prior_results.append(
                {
                    'label': summary['label'],
                    'output_preview': str(summary['output_preview']),
                }
            )
            if child_result.stop_reason in {'backend_error', 'budget_exceeded'}:
                failed_children += 1
                batch_failed += 1
                failed_labels.add(subtask_label)
                if isinstance(max_failures, int) and failed_children > max_failures:
                    stop_processing = True
                    break
                if not continue_on_error:
                    stop_processing = True
                    break
            else:
                batch_completed += 1
                completed_labels.add(subtask_label)
        batch_status = 'completed'
        if batch_failed and batch_completed:
            batch_status = 'partial'
        elif batch_failed:
            batch_status = 'failed'
        batch_summaries.append(
            {
                'batch_index': batch_index,
                'labels': batch_labels,
                'completed_children': batch_completed,
                'failed_children': batch_failed,
                'skipped_children': batch_skipped,
                'status': batch_status,
            }
        )
    assert child_result is not None
    completed_children = len(child_summaries) - failed_children
    resumed_children = sum(
        1 for summary in child_summaries if summary.get('resume_used')
    )
    group_status = 'completed'
    if failed_children and completed_children:
        group_status = 'partial'
    elif failed_children:
        group_status = 'failed'
    delegate_after_messages = (
        agent.plugin_runtime.after_delegate_injections()
        if agent.plugin_runtime is not None
        else ()
    )
    if group_id is not None and agent.agent_manager is not None:
        agent.agent_manager.finish_group(
            group_id,
            status=group_status,
            completed_children=completed_children,
            failed_children=failed_children,
            batch_count=len(batch_summaries),
            max_batch_size=max((len(batch['labels']) for batch in batch_summaries), default=0),
            dependency_skips=dependency_skips,
        )
    summary_lines = [
        (
            'Delegated agent completed the subtask.'
            if len(child_summaries) == 1
            else f'Delegated agent completed {len(child_summaries)} sequential subtasks.'
        ),
    ]
    if group_id is not None:
        summary_lines.append(f'group_id={group_id}')
        summary_lines.append(f'group_status={group_status}')
        summary_lines.append(f'resumed_children={resumed_children}')
        summary_lines.append(f'strategy={strategy}')
        summary_lines.append(f'batch_count={len(batch_summaries)}')
        summary_lines.append(f'dependency_skips={dependency_skips}')
        summary_lines.append('')
    if delegate_preflight_messages:
        summary_lines.append('Plugin delegate preflight:')
        summary_lines.extend(f'- {message}' for message in delegate_preflight_messages)
        summary_lines.append('')
    for batch in batch_summaries:
        summary_lines.append(
            f"[batch {batch['batch_index']}] status={batch['status']} "
            f"labels={','.join(batch['labels']) or '(none)'} "
            f"completed={batch['completed_children']} failed={batch['failed_children']} "
            f"skipped={batch['skipped_children']}"
        )
    if batch_summaries:
        summary_lines.append('')
    for summary in child_summaries:
        summary_lines.extend(
            [
                f"[{summary['label']}]",
                f"batch_index={summary['batch_index']}",
                f"session_id={summary['session_id']}",
                f"turns={summary['turns']}",
                f"tool_calls={summary['tool_calls']}",
                f"stop_reason={summary['stop_reason']}",
                f"resume_used={summary['resume_used']}",
                f"resumed_from_session_id={summary['resumed_from_session_id']}",
                f"depends_on={','.join(summary.get('depends_on', [])) or '(none)'}",
                f"output_preview={summary['output_preview']}",
                '',
            ]
        )
    if delegate_after_messages:
        summary_lines.append('Plugin delegate completion:')
        summary_lines.extend(f'- {message}' for message in delegate_after_messages)
        summary_lines.append('')
    summary_lines.append('Final delegated output:')
    summary_lines.append(child_result.final_output)
    return ToolExecutionResult(
        name=tool_name,
        ok=True,
        content='\n'.join(summary_lines).strip(),
        metadata={
            'action': tool_name,
            'subagent_type': agent_def.agent_type,
            'child_session_id': child_result.session_id,
            'child_session_ids': child_session_ids,
            'child_turns': child_result.turns,
            'child_tool_calls': child_result.tool_calls,
            'child_stop_reason': child_result.stop_reason,
            'child_results': child_summaries,
            'subtask_count': len(child_summaries),
            'group_id': group_id,
            'group_status': group_status,
            'failed_children': failed_children,
            'completed_children': completed_children,
            'resumed_children': resumed_children,
            'strategy': strategy,
            'max_failures': max_failures,
            'delegate_batches': batch_summaries,
            'dependency_skips': dependency_skips,
            'plugin_delegate_preflight_messages': list(delegate_preflight_messages),
            'plugin_delegate_after_messages': list(delegate_after_messages),
        },
    )


def normalize_delegate_subtasks(
    arguments: dict[str, object],
) -> list[dict[str, object]]:
    subtasks: list[dict[str, object]] = []
    raw_subtasks = arguments.get('subtasks')
    if isinstance(raw_subtasks, list):
        for index, item in enumerate(raw_subtasks, start=1):
            if isinstance(item, str) and item.strip():
                subtasks.append(
                    {
                        'prompt': item.strip(),
                        'label': f'subtask_{index}',
                        '_delegate_index': index,
                    }
                )
                continue
            if isinstance(item, dict):
                prompt = item.get('prompt')
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                label = item.get('label')
                max_turns = item.get('max_turns')
                task: dict[str, object] = {
                    'prompt': prompt.strip(),
                    'label': (
                        label if isinstance(label, str) and label.strip() else f'subtask_{index}'
                    ),
                }
                resume_session_id = item.get('resume_session_id')
                if resume_session_id is None:
                    resume_session_id = item.get('session_id')
                if isinstance(resume_session_id, str) and resume_session_id.strip():
                    task['resume_session_id'] = resume_session_id.strip()
                depends_on = item.get('depends_on')
                if isinstance(depends_on, list):
                    task['depends_on'] = tuple(
                        dependency.strip()
                        for dependency in depends_on
                        if isinstance(dependency, str) and dependency.strip()
                    )
                if (
                    isinstance(max_turns, int)
                    and not isinstance(max_turns, bool)
                    and max_turns > 0
                ):
                    task['max_turns'] = max_turns
                task['_delegate_index'] = index
                subtasks.append(task)
    prompt = arguments.get('prompt')
    if isinstance(prompt, str) and prompt.strip():
        if not subtasks:
            task: dict[str, object] = {'prompt': prompt.strip(), 'label': 'subtask_1'}
            resume_session_id = arguments.get('resume_session_id')
            if resume_session_id is None:
                resume_session_id = arguments.get('session_id')
            if isinstance(resume_session_id, str) and resume_session_id.strip():
                task['resume_session_id'] = resume_session_id.strip()
            task['_delegate_index'] = 1
            subtasks.append(task)
    return [
        {
            **task,
            '_delegate_index': int(task.get('_delegate_index', index)),
        }
        for index, task in enumerate(subtasks[:8], start=1)
    ]


def normalize_delegate_strategy(strategy: object) -> str:
    if not isinstance(strategy, str) or not strategy.strip():
        return 'serial'
    normalized = strategy.strip().lower().replace('-', '_')
    if normalized in {'graph', 'topological', 'dependency_graph', 'parallel', 'parallel_batches'}:
        return 'topological'
    return 'serial'


def plan_delegate_batches(
    subtasks: list[dict[str, object]],
    strategy: str,
) -> list[list[dict[str, object]]]:
    if strategy != 'topological':
        return [subtasks]
    remaining = list(subtasks)
    scheduled_labels: set[str] = set()
    known_labels = {
        str(task.get('label'))
        for task in subtasks
        if isinstance(task.get('label'), str) and str(task.get('label')).strip()
    }
    batches: list[list[dict[str, object]]] = []
    while remaining:
        ready: list[dict[str, object]] = []
        blocked: list[dict[str, object]] = []
        for task in remaining:
            dependencies = tuple(
                item for item in task.get('depends_on', ()) if isinstance(item, str) and item
            )
            if any(dependency not in known_labels for dependency in dependencies):
                blocked.append(task)
                continue
            if all(dependency in scheduled_labels for dependency in dependencies):
                ready.append(task)
            else:
                blocked.append(task)
        if not ready:
            batches.append(blocked)
            break
        batches.append(
            sorted(
                ready,
                key=lambda task: int(task.get('_delegate_index', 0)),
            )
        )
        scheduled_labels.update(
            str(task.get('label'))
            for task in ready
            if isinstance(task.get('label'), str) and str(task.get('label')).strip()
        )
        remaining = blocked
    return batches


def delegated_task_units(arguments: dict[str, object]) -> int:
    subtasks = arguments.get('subtasks')
    if isinstance(subtasks, list):
        count = sum(
            1
            for item in subtasks
            if (isinstance(item, str) and item.strip()) or (
                isinstance(item, dict)
                and isinstance(item.get('prompt'), str)
                and item.get('prompt', '').strip()
            )
        )
        if count:
            return count
    return 1


def prepend_delegate_context(
    prompt: str,
    prior_results: list[dict[str, str]],
) -> str:
    lines = [
        '<system-reminder>',
        'Prior delegated subtask summaries:',
    ]
    for result in prior_results[-4:]:
        lines.append(f"- {result['label']}: {result['output_preview']}")
    lines.extend(['</system-reminder>', '', prompt])
    return '\n'.join(lines)


def prepend_plugin_delegate_context(
    prompt: str,
    messages: tuple[str, ...],
) -> str:
    if not messages:
        return prompt
    lines = [
        '<system-reminder>',
        'Plugin delegate guidance:',
    ]
    lines.extend(f'- {message}' for message in messages)
    lines.extend(['</system-reminder>', '', prompt])
    return '\n'.join(lines)
