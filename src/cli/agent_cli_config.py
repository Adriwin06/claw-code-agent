from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..agent_runtime import LocalCodingAgent
from ..agent_types import (
    AgentPermissions,
    AgentRuntimeConfig,
    BudgetConfig,
    ModelConfig,
    ModelPricing,
    OutputSchemaConfig,
)


def _add_agent_common_args(parser: argparse.ArgumentParser, *, include_backend: bool) -> None:
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'))
    if include_backend:
        parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
        parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY', 'local-token'))
        parser.add_argument(
            '--llm-backend',
            default=os.environ.get('CLAW_LLM_BACKEND', 'openai_compat'),
            help='LLM transport backend (openai_compat or litellm)',
        )
        parser.add_argument('--temperature', type=float, default=0.0)
        parser.add_argument('--timeout-seconds', type=float, default=120.0)
        parser.add_argument('--input-cost-per-million', type=float, default=0.0)
        parser.add_argument('--output-cost-per-million', type=float, default=0.0)
    parser.add_argument('--cwd', default='.')
    parser.add_argument('--add-dir', action='append', default=[])
    parser.add_argument('--disable-claude-md', action='store_true')
    parser.add_argument('--allow-write', action='store_true')
    parser.add_argument('--allow-shell', action='store_true')
    parser.add_argument('--unsafe', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--auto-snip-threshold', type=int)
    parser.add_argument('--auto-compact-threshold', type=int)
    parser.add_argument('--compact-preserve-messages', type=int, default=4)
    parser.add_argument('--max-total-tokens', type=int)
    parser.add_argument('--max-input-tokens', type=int)
    parser.add_argument('--max-output-tokens', type=int)
    parser.add_argument('--max-reasoning-tokens', type=int)
    parser.add_argument('--max-budget-usd', type=float)
    parser.add_argument('--max-tool-calls', type=int)
    parser.add_argument('--max-delegated-tasks', type=int)
    parser.add_argument('--max-model-calls', type=int)
    parser.add_argument('--max-session-turns', type=int)
    parser.add_argument('--response-schema-file')
    parser.add_argument('--response-schema-name')
    parser.add_argument('--response-schema-strict', action='store_true')
    parser.add_argument('--scratchpad-root')
    parser.add_argument('--system-prompt')
    parser.add_argument('--append-system-prompt')
    parser.add_argument('--override-system-prompt')


def _load_output_schema_config(args: argparse.Namespace) -> OutputSchemaConfig | None:
    schema_file = getattr(args, 'response_schema_file', None)
    if not schema_file:
        return None
    payload = json.loads(Path(schema_file).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('response schema file must contain a top-level JSON object')
    name = getattr(args, 'response_schema_name', None) or Path(schema_file).stem
    return OutputSchemaConfig(
        name=name,
        schema=payload,
        strict=bool(getattr(args, 'response_schema_strict', False)),
    )


def _build_runtime_config(args: argparse.Namespace) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        cwd=Path(args.cwd).resolve(),
        max_turns=getattr(args, 'max_turns', 12),
        permissions=AgentPermissions(
            allow_file_write=args.allow_write,
            allow_shell_commands=args.allow_shell,
            allow_destructive_shell_commands=args.unsafe,
        ),
        stream_model_responses=bool(getattr(args, 'stream', False)),
        auto_snip_threshold_tokens=getattr(args, 'auto_snip_threshold', None),
        auto_compact_threshold_tokens=getattr(args, 'auto_compact_threshold', None),
        compact_preserve_messages=max(0, int(getattr(args, 'compact_preserve_messages', 4))),
        additional_working_directories=tuple(Path(path).resolve() for path in args.add_dir),
        disable_claude_md_discovery=args.disable_claude_md,
        budget_config=BudgetConfig(
            max_total_tokens=getattr(args, 'max_total_tokens', None),
            max_input_tokens=getattr(args, 'max_input_tokens', None),
            max_output_tokens=getattr(args, 'max_output_tokens', None),
            max_reasoning_tokens=getattr(args, 'max_reasoning_tokens', None),
            max_total_cost_usd=getattr(args, 'max_budget_usd', None),
            max_tool_calls=getattr(args, 'max_tool_calls', None),
            max_delegated_tasks=getattr(args, 'max_delegated_tasks', None),
            max_model_calls=getattr(args, 'max_model_calls', None),
            max_session_turns=getattr(args, 'max_session_turns', None),
        ),
        output_schema=_load_output_schema_config(args),
        session_directory=(Path('.port_sessions') / 'agent').resolve(),
        scratchpad_root=(
            Path(getattr(args, 'scratchpad_root')).resolve()
            if getattr(args, 'scratchpad_root', None)
            else (Path('.port_sessions') / 'scratchpad').resolve()
        ),
    )


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        model=args.model,
        base_url=getattr(args, 'base_url', os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')),
        api_key=getattr(args, 'api_key', os.environ.get('OPENAI_API_KEY', 'local-token')),
        llm_backend=getattr(
            args,
            'llm_backend',
            os.environ.get('CLAW_LLM_BACKEND', 'openai_compat'),
        ),
        temperature=getattr(args, 'temperature', 0.0),
        timeout_seconds=getattr(args, 'timeout_seconds', 120.0),
        pricing=ModelPricing(
            input_cost_per_million_tokens_usd=float(
                getattr(args, 'input_cost_per_million', 0.0) or 0.0
            ),
            output_cost_per_million_tokens_usd=float(
                getattr(args, 'output_cost_per_million', 0.0) or 0.0
            ),
        ),
    )


def _build_agent(args: argparse.Namespace) -> LocalCodingAgent:
    model_config = _build_model_config(args)
    return LocalCodingAgent(
        model_config=model_config,
        runtime_config=_build_runtime_config(args),
        llm_backend=model_config.llm_backend,
        custom_system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        override_system_prompt=args.override_system_prompt,
    )


def _append_agent_forwarded_args(
    command: list[str],
    args: argparse.Namespace,
    *,
    include_backend: bool,
) -> None:
    command.extend(['--cwd', str(args.cwd)])
    command.extend(['--max-turns', str(getattr(args, 'max_turns', 12))])
    if include_backend:
        command.extend(['--model', str(args.model)])
        command.extend(['--base-url', str(args.base_url)])
        command.extend(['--api-key', str(args.api_key)])
        command.extend(['--llm-backend', str(args.llm_backend)])
        command.extend(['--temperature', str(args.temperature)])
        command.extend(['--timeout-seconds', str(args.timeout_seconds)])
        command.extend(['--input-cost-per-million', str(args.input_cost_per_million)])
        command.extend(['--output-cost-per-million', str(args.output_cost_per_million)])
    else:
        command.extend(['--model', str(args.model)])
    for path in getattr(args, 'add_dir', []):
        command.extend(['--add-dir', str(path)])
    for flag in (
        ('--disable-claude-md', getattr(args, 'disable_claude_md', False)),
        ('--allow-write', getattr(args, 'allow_write', False)),
        ('--allow-shell', getattr(args, 'allow_shell', False)),
        ('--unsafe', getattr(args, 'unsafe', False)),
        ('--stream', getattr(args, 'stream', False)),
        ('--show-transcript', getattr(args, 'show_transcript', False)),
        (
            '--response-schema-strict',
            getattr(args, 'response_schema_strict', False),
        ),
    ):
        if flag[1]:
            command.append(flag[0])
    for name, value in (
        ('--auto-snip-threshold', getattr(args, 'auto_snip_threshold', None)),
        ('--auto-compact-threshold', getattr(args, 'auto_compact_threshold', None)),
        ('--compact-preserve-messages', getattr(args, 'compact_preserve_messages', None)),
        ('--max-total-tokens', getattr(args, 'max_total_tokens', None)),
        ('--max-input-tokens', getattr(args, 'max_input_tokens', None)),
        ('--max-output-tokens', getattr(args, 'max_output_tokens', None)),
        ('--max-reasoning-tokens', getattr(args, 'max_reasoning_tokens', None)),
        ('--max-budget-usd', getattr(args, 'max_budget_usd', None)),
        ('--max-tool-calls', getattr(args, 'max_tool_calls', None)),
        ('--max-delegated-tasks', getattr(args, 'max_delegated_tasks', None)),
        ('--max-model-calls', getattr(args, 'max_model_calls', None)),
        ('--max-session-turns', getattr(args, 'max_session_turns', None)),
        ('--response-schema-file', getattr(args, 'response_schema_file', None)),
        ('--response-schema-name', getattr(args, 'response_schema_name', None)),
        ('--scratchpad-root', getattr(args, 'scratchpad_root', None)),
        ('--system-prompt', getattr(args, 'system_prompt', None)),
        ('--append-system-prompt', getattr(args, 'append_system_prompt', None)),
        ('--override-system-prompt', getattr(args, 'override_system_prompt', None)),
    ):
        if value is not None:
            command.extend([name, str(value)])


def _add_agent_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('session_id')
    parser.add_argument('prompt')
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--show-transcript', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--api-key')
    parser.add_argument('--llm-backend')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--timeout-seconds', type=float)
    parser.add_argument('--input-cost-per-million', type=float)
    parser.add_argument('--output-cost-per-million', type=float)
    parser.add_argument('--allow-write', action='store_true')
    parser.add_argument('--allow-shell', action='store_true')
    parser.add_argument('--unsafe', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--auto-snip-threshold', type=int)
    parser.add_argument('--auto-compact-threshold', type=int)
    parser.add_argument('--compact-preserve-messages', type=int)
    parser.add_argument('--max-total-tokens', type=int)
    parser.add_argument('--max-input-tokens', type=int)
    parser.add_argument('--max-output-tokens', type=int)
    parser.add_argument('--max-reasoning-tokens', type=int)
    parser.add_argument('--max-budget-usd', type=float)
    parser.add_argument('--max-tool-calls', type=int)
    parser.add_argument('--max-delegated-tasks', type=int)
    parser.add_argument('--max-model-calls', type=int)
    parser.add_argument('--max-session-turns', type=int)
    parser.add_argument('--response-schema-file')
    parser.add_argument('--response-schema-name')
    parser.add_argument('--response-schema-strict', action='store_true')
    parser.add_argument('--scratchpad-root')
