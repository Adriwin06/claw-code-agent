from __future__ import annotations

from typing import Any

from src.agent.agent_session import AgentSessionState
from src.agent.agent_types import BudgetConfig, UsageStats
from src.agent.run_state import PromptPreflightResult
from src.core.governance.token_budget import calculate_token_budget
from src.agent.agent_context_usage import collect_context_usage, estimate_tokens
from src.session.compact import MAX_COMPACT_FAILURES, compact_conversation


def preflight_prompt_length(
    agent: Any,
    session: AgentSessionState,
    stream_events: list[dict[str, object]],
    *,
    turn_index: int,
    calculate_token_budget_fn,
    compact_conversation_fn,
    reduce_context_pressure_fn,
) -> PromptPreflightResult:
    snapshot = calculate_token_budget_fn(
        session=session,
        model=agent.model_config.model,
        budget_config=agent.runtime_config.budget_config,
        output_schema=agent.runtime_config.output_schema,
    )
    if not snapshot.exceeds_soft_limit and not snapshot.exceeds_hard_limit:
        return PromptPreflightResult()

    stream_events.append(
        {
            'type': 'prompt_length_check',
            'turn_index': turn_index,
            'projected_input_tokens': snapshot.projected_input_tokens,
            'soft_input_limit_tokens': snapshot.soft_input_limit_tokens,
            'hard_input_limit_tokens': snapshot.hard_input_limit_tokens,
            'soft_overflow_tokens': snapshot.soft_overflow_tokens,
            'overflow_tokens': snapshot.overflow_tokens,
            'exceeds_hard_limit': snapshot.exceeds_hard_limit,
        }
    )

    target_tokens = snapshot.soft_input_limit_tokens
    if snapshot.exceeds_hard_limit:
        target_tokens = snapshot.hard_input_limit_tokens
    if target_tokens < 0:
        target_tokens = 0

    if reduce_context_pressure_fn(
        session,
        stream_events,
        turn_index=turn_index,
        target_tokens=target_tokens,
        allow_compaction=True,
    ):
        recovered = calculate_token_budget_fn(
            session=session,
            model=agent.model_config.model,
            budget_config=agent.runtime_config.budget_config,
            output_schema=agent.runtime_config.output_schema,
        )
        stream_events.append(
            {
                'type': 'prompt_length_recovery',
                'turn_index': turn_index,
                'strategy': 'heuristic',
                'projected_input_tokens': recovered.projected_input_tokens,
                'soft_input_limit_tokens': recovered.soft_input_limit_tokens,
                'hard_input_limit_tokens': recovered.hard_input_limit_tokens,
                'exceeds_hard_limit': recovered.exceeds_hard_limit,
                'exceeds_soft_limit': recovered.exceeds_soft_limit,
            }
        )
        if not recovered.exceeds_soft_limit and not recovered.exceeds_hard_limit:
            return PromptPreflightResult()
        snapshot = recovered

    if agent._compact_consecutive_failures >= MAX_COMPACT_FAILURES:
        stream_events.append(
            {
                'type': 'auto_compact_circuit_breaker',
                'turn_index': turn_index,
                'consecutive_failures': agent._compact_consecutive_failures,
            }
        )
    elif can_auto_compact_with_summary(agent, session):
        compact_result = compact_conversation_fn(
            agent,
            custom_instructions=(
                'Automatically collapse earlier conversation context to fit the next model '
                'turn. Preserve the active task, recent file changes, failures, pending work, '
                'and exact next step.'
            ),
        )
        if compact_result.error is None:
            agent._compact_consecutive_failures = 0
            recovered = calculate_token_budget_fn(
                session=session,
                model=agent.model_config.model,
                budget_config=agent.runtime_config.budget_config,
                output_schema=agent.runtime_config.output_schema,
            )
            stream_events.append(
                {
                    'type': 'auto_compact_summary',
                    'turn_index': turn_index,
                    'pre_compact_token_count': compact_result.pre_compact_token_count,
                    'post_compact_token_count': compact_result.post_compact_token_count,
                    'true_post_compact_token_count': compact_result.true_post_compact_token_count,
                    'summary_usage_tokens': compact_result.usage.total_tokens,
                    'ptl_retries': compact_result.ptl_retries,
                    'projected_input_tokens': recovered.projected_input_tokens,
                    'soft_input_limit_tokens': recovered.soft_input_limit_tokens,
                    'hard_input_limit_tokens': recovered.hard_input_limit_tokens,
                    'exceeds_hard_limit': recovered.exceeds_hard_limit,
                    'exceeds_soft_limit': recovered.exceeds_soft_limit,
                }
            )
            if not recovered.exceeds_soft_limit and not recovered.exceeds_hard_limit:
                return PromptPreflightResult(
                    usage_increment=compact_result.usage,
                    model_calls_increment=1,
                )
            snapshot = recovered
            if compact_result.usage.total_tokens:
                return PromptPreflightResult(
                    usage_increment=compact_result.usage,
                    model_calls_increment=1,
                    stop_reason=(
                        'prompt_too_long'
                        if recovered.exceeds_hard_limit
                        else None
                    ),
                    reason=(
                        build_prompt_length_error(recovered)
                        if recovered.exceeds_hard_limit
                        else None
                    ),
                )
        else:
            agent._compact_consecutive_failures += 1
            stream_events.append(
                {
                    'type': 'auto_compact_failed',
                    'turn_index': turn_index,
                    'reason': compact_result.error,
                    'consecutive_failures': agent._compact_consecutive_failures,
                }
            )

    if snapshot.exceeds_hard_limit:
        return PromptPreflightResult(
            stop_reason='prompt_too_long',
            reason=build_prompt_length_error(snapshot),
        )

    stream_events.append(
        {
            'type': 'prompt_length_warning',
            'turn_index': turn_index,
            'projected_input_tokens': snapshot.projected_input_tokens,
            'soft_input_limit_tokens': snapshot.soft_input_limit_tokens,
            'hard_input_limit_tokens': snapshot.hard_input_limit_tokens,
            'soft_overflow_tokens': snapshot.soft_overflow_tokens,
        }
    )
    return PromptPreflightResult()


def can_auto_compact_with_summary(agent: Any, session: AgentSessionState) -> bool:
    prefix_count = agent._compact_prefix_count(session)
    preserve_count = max(agent.runtime_config.compact_preserve_messages, 1)
    return len(session.messages) - prefix_count > preserve_count


def build_prompt_length_error(snapshot: Any) -> str:
    return (
        'Stopped before the next model call because the prompt would exceed the '
        'effective input budget. '
        f'Projected prompt tokens: {snapshot.projected_input_tokens:,}; '
        f'hard input limit: {snapshot.hard_input_limit_tokens:,}; '
        f'soft input limit: {snapshot.soft_input_limit_tokens:,}.'
    )


def reactive_compact_session(
    agent: Any,
    session: AgentSessionState,
    stream_events: list[dict[str, object]],
    *,
    turn_index: int,
) -> bool:
    return reduce_context_pressure(
        agent,
        session,
        stream_events,
        turn_index=turn_index,
        target_tokens=0,
        allow_compaction=True,
        reactive=True,
    )


def reduce_context_pressure(
    agent: Any,
    session: AgentSessionState,
    stream_events: list[dict[str, object]],
    *,
    turn_index: int,
    target_tokens: int,
    allow_compaction: bool,
    reactive: bool = False,
) -> bool:
    changed = False
    for _ in range(6):
        usage_report = collect_context_usage(
            session=session,
            model=agent.model_config.model,
            strategy='reactive_compact' if reactive else 'context_pressure',
        )
        if usage_report.total_tokens <= target_tokens:
            break
        if snip_session_pass(
            agent,
            session,
            stream_events,
            turn_index=turn_index,
            target_tokens=target_tokens,
            current_total=usage_report.total_tokens,
            reactive=reactive,
        ):
            changed = True
            continue
        if allow_compaction and compact_session_pass(
            agent,
            session,
            stream_events,
            turn_index=turn_index,
            usage_total=usage_report.total_tokens,
            reactive=reactive,
        ):
            changed = True
            if reactive:
                continue
            break
        break
    return changed


def snip_session_pass(
    agent: Any,
    session: AgentSessionState,
    stream_events: list[dict[str, object]],
    *,
    turn_index: int,
    target_tokens: int,
    current_total: int,
    reactive: bool,
) -> bool:
    prefix_count = agent._compact_prefix_count(session)
    tail_count = min(
        max(agent.runtime_config.compact_preserve_messages, 0),
        max(len(session.messages) - prefix_count, 0),
    )
    candidate_indexes = [
        index
        for index in range(prefix_count, max(len(session.messages) - tail_count, prefix_count))
        if agent._message_can_be_snipped(session.messages[index])
    ]
    if not candidate_indexes:
        return False
    snipped_count = 0
    tokens_removed = 0
    snipped_message_ids: list[str] = []
    for index in candidate_indexes:
        if current_total <= target_tokens and not reactive:
            break
        message = session.messages[index]
        original_tokens = estimate_tokens(message.content, agent.model_config.model)
        replacement = agent._build_snipped_message_content(message)
        replacement_tokens = estimate_tokens(replacement, agent.model_config.model)
        if replacement_tokens >= original_tokens:
            continue
        session.tombstone_message(
            index,
            summary=replacement,
            stop_reason='snipped_for_context',
            mutation_kind='snip_tombstone',
            metadata={
                'kind': 'snipped_message',
                'original_token_estimate': original_tokens,
                'replacement_token_estimate': replacement_tokens,
                'snipped_turn_index': turn_index,
                'snipped_from_role': message.role,
                'snipped_from_message_id': message.message_id,
                'snipped_from_kind': message.metadata.get('kind'),
                'snipped_from_lineage_id': message.metadata.get('lineage_id'),
                'snipped_from_revision': message.metadata.get('revision'),
            },
        )
        delta = original_tokens - replacement_tokens
        current_total -= delta
        tokens_removed += delta
        snipped_count += 1
        if session.messages[index].message_id:
            snipped_message_ids.append(session.messages[index].message_id)
        if reactive and snipped_count >= 3:
            break
    if not snipped_count:
        return False
    stream_events.append(
        {
            'type': 'reactive_snip_boundary' if reactive else 'snip_boundary',
            'turn_index': turn_index,
            'snipped_message_count': snipped_count,
            'estimated_tokens_removed': tokens_removed,
            'snipped_message_ids': snipped_message_ids,
        }
    )
    return True


def compact_session_pass(
    agent: Any,
    session: AgentSessionState,
    stream_events: list[dict[str, object]],
    *,
    turn_index: int,
    usage_total: int,
    reactive: bool,
) -> bool:
    prefix_count = agent._compact_prefix_count(session)
    preserve_messages = max(agent.runtime_config.compact_preserve_messages, 0)
    if reactive:
        preserve_messages = max(preserve_messages // 2, 1)
    tail_count = min(
        preserve_messages,
        max(len(session.messages) - prefix_count, 0),
    )
    compact_end = len(session.messages) - tail_count
    if compact_end <= prefix_count:
        return False
    candidates = session.messages[prefix_count:compact_end]
    preserved_tail = list(session.messages[compact_end:])
    if not candidates:
        return False
    compacted_tokens = sum(
        usage.tokens
        for usage in (
            collect_context_usage(
                session=AgentSessionState(
                    system_prompt_parts=session.system_prompt_parts,
                    user_context=session.user_context,
                    system_context=session.system_context,
                    messages=list(candidates),
                ),
                model=agent.model_config.model,
                strategy='compacted_segment',
            ).categories
        )
        if usage.name != 'Free space'
    )
    compact_message = agent._build_compact_boundary_message(
        candidates,
        turn_index=turn_index,
        estimated_tokens_before=usage_total,
        estimated_tokens_removed=compacted_tokens,
        preserved_tail_count=tail_count,
        preserved_tail=preserved_tail,
    )
    session.messages = (
        session.messages[:prefix_count]
        + [compact_message]
        + session.messages[compact_end:]
    )
    stream_events.append(
        {
            'type': 'reactive_compact_boundary' if reactive else 'compact_boundary',
            'turn_index': turn_index,
            'compacted_message_count': len(candidates),
            'estimated_tokens_before': usage_total,
            'estimated_tokens_removed': compacted_tokens,
            'preserved_tail_count': tail_count,
            'preserved_tail_ids': [
                message.message_id for message in preserved_tail if message.message_id
            ],
            'compaction_depth': compact_message.metadata.get('compaction_depth'),
            'nested_compaction_count': compact_message.metadata.get('nested_compaction_count'),
            'compacted_message_ids': [
                message.message_id for message in candidates if message.message_id
            ],
        }
    )
    return True


def check_token_budget(
    usage: UsageStats,
    budget: BudgetConfig,
) -> str | None:
    if budget.max_total_tokens is not None and usage.total_tokens > budget.max_total_tokens:
        return (
            'Stopped because the total token budget was exceeded '
            f'({usage.total_tokens} > {budget.max_total_tokens}).'
        )
    if budget.max_input_tokens is not None and usage.input_tokens > budget.max_input_tokens:
        return (
            'Stopped because the input token budget was exceeded '
            f'({usage.input_tokens} > {budget.max_input_tokens}).'
        )
    if budget.max_output_tokens is not None and usage.output_tokens > budget.max_output_tokens:
        return (
            'Stopped because the output token budget was exceeded '
            f'({usage.output_tokens} > {budget.max_output_tokens}).'
        )
    if (
        budget.max_reasoning_tokens is not None
        and usage.reasoning_tokens > budget.max_reasoning_tokens
    ):
        return (
            'Stopped because the reasoning token budget was exceeded '
            f'({usage.reasoning_tokens} > {budget.max_reasoning_tokens}).'
        )
    return None
