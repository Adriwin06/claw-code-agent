from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from src.agent.runtime.agent import LocalCodingAgent
from src.agent.models.types import (
    AgentPermissions,
    AgentRuntimeConfig,
    BudgetConfig,
    ModelConfig,
    ModelPricing,
    OutputSchemaConfig,
)
from src.features.system.env_runtime import clean_env_value


_DEFAULT_MODEL = 'Qwen/Qwen3-Coder-30B-A3B-Instruct'
_LOCAL_BASE_URL_DEFAULT = 'http://127.0.0.1:8000/v1'
_LOCAL_API_KEY_DEFAULT = 'local-token'
_PROVIDER_ENV_ALIASES = {
    'azure': 'azure_openai',
    'azure_openai': 'azure_openai',
    'anthropic': 'anthropic',
    'deepseek': 'deepseek',
    'gemini': 'gemini',
    'google': 'gemini',
    'groq': 'groq',
    'mistral': 'mistral',
    'loic': 'loic',
    'loic_api': 'loic',
    'ollama': 'ollama',
    'ollama_chat': 'ollama',
    'openai': 'openai',
    'openrouter': 'openrouter',
    'together': 'together',
    'together_ai': 'together',
    'xai': 'xai',
}
_PROVIDER_API_KEY_ENV_VARS = {
    'anthropic': ('ANTHROPIC_API_KEY',),
    'azure_openai': ('AZURE_OPENAI_API_KEY',),
    'deepseek': ('DEEPSEEK_API_KEY',),
    'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
    'groq': ('GROQ_API_KEY',),
    'loic': ('LOIC_API_KEY',),
    'mistral': ('MISTRAL_API_KEY',),
    'openai': ('OPENAI_API_KEY',),
    'openrouter': ('OPENROUTER_API_KEY',),
    'together': ('TOGETHER_API_KEY',),
    'xai': ('XAI_API_KEY',),
}
_BASE_URL_PROVIDER_HINTS = (
    ('openrouter.ai', 'openrouter'),
    ('api.openai.com', 'openai'),
    ('api.anthropic.com', 'anthropic'),
    ('generativelanguage.googleapis.com', 'gemini'),
    ('api.mistral.ai', 'mistral'),
    ('api.groq.com', 'groq'),
    ('api.x.ai', 'xai'),
    ('api.deepseek.com', 'deepseek'),
    ('api.together.xyz', 'together'),
    ('loic.exaload.fr', 'loic'),
    ('loic.exaload.app', 'loic'),
)


def _looks_like_ollama_base_url(base_url: str | None) -> bool:
    if not isinstance(base_url, str):
        return False
    normalized = clean_env_value(base_url).strip().lower()
    if not normalized:
        return False
    return ':11434' in normalized or 'ollama' in normalized


def _looks_like_local_base_url(base_url: str | None) -> bool:
    if not isinstance(base_url, str):
        return False
    normalized = clean_env_value(base_url).strip().lower()
    if not normalized:
        return False
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or '').strip().lower()
    return hostname in {'127.0.0.1', 'localhost', 'host.docker.internal'}


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        if name in os.environ:
            return clean_env_value(os.environ.get(name, default))
    return default


def _env_nonempty(*names: str, default: str | None = None) -> str | None:
    for name in names:
        if name not in os.environ:
            continue
        raw = os.environ.get(name)
        if not isinstance(raw, str):
            continue
        stripped = clean_env_value(raw).strip()
        if stripped:
            return stripped
    return default


def _env_float(*names: str, default: float) -> float:
    raw = _env_first(*names)
    if not isinstance(raw, str):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag_disabled(name: str) -> bool:
    """Return True if the named env var is set to a truthy disable value (1/true/yes)."""
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes')


def _env_optional_int(*names: str) -> int | None:
    raw = _env_first(*names)
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _normalize_provider_name(provider: str | None) -> str | None:
    if not isinstance(provider, str):
        return None
    normalized = clean_env_value(provider).strip().lower()
    if not normalized:
        return None
    return _PROVIDER_ENV_ALIASES.get(normalized, normalized)


def _provider_from_model_name(model: str | None) -> str | None:
    if not isinstance(model, str):
        return None
    normalized_model = clean_env_value(model).strip()
    if not normalized_model or '/' not in normalized_model:
        return None
    return _normalize_provider_name(normalized_model.split('/', 1)[0])


def _provider_from_base_url(base_url: str | None) -> str | None:
    if _looks_like_ollama_base_url(base_url):
        return 'ollama'
    if not isinstance(base_url, str):
        return None
    normalized = clean_env_value(base_url).strip().lower()
    if not normalized:
        return None
    for marker, provider in _BASE_URL_PROVIDER_HINTS:
        if marker in normalized:
            return provider
    return None


def _resolve_provider_name(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> str | None:
    return (
        _normalize_provider_name(provider)
        or _provider_from_base_url(base_url)
        or _provider_from_model_name(model)
    )


def _default_api_key_for_provider(
    provider: str | None,
    *,
    base_url: str | None = None,
) -> str:
    if provider == 'ollama':
        return 'ollama'
    if _looks_like_local_base_url(base_url):
        return _LOCAL_API_KEY_DEFAULT
    return _LOCAL_API_KEY_DEFAULT


def _resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    explicit = clean_env_value(explicit_api_key).strip() if isinstance(explicit_api_key, str) else ''
    if explicit:
        return explicit

    override = _env_nonempty('LLM_API_KEY')
    if override is not None:
        return override

    # Loïc endpoint always wins — overrides LLM_PROVIDER for key lookup
    if _provider_from_base_url(base_url) == 'loic':
        loic_key = _env_nonempty(*_PROVIDER_API_KEY_ENV_VARS.get('loic', ()))
        if loic_key is not None:
            return loic_key

    resolved_provider = _resolve_provider_name(
        provider,
        model=model,
        base_url=base_url,
    )
    if resolved_provider is not None:
        env_names = _PROVIDER_API_KEY_ENV_VARS.get(resolved_provider, ())
        provider_key = _env_nonempty(*env_names)
        if provider_key is not None:
            return provider_key

    return _default_api_key_for_provider(resolved_provider, base_url=base_url)


def _default_api_key_from_env() -> str:
    model = _env_nonempty('LLM_MODEL')
    provider = _env_nonempty('LLM_PROVIDER')
    base_url = _env_nonempty('LOIC_API_BASE', 'LLM_API_BASE', default=_LOCAL_BASE_URL_DEFAULT)
    return _resolve_api_key(
        provider=provider,
        model=model,
        base_url=base_url,
    )


def _normalize_base_url(
    base_url: str | None,
    *,
    model: str | None = None,
) -> str:
    if not isinstance(base_url, str):
        return _LOCAL_BASE_URL_DEFAULT
    normalized_base_url = clean_env_value(base_url).strip()
    if not normalized_base_url:
        return _LOCAL_BASE_URL_DEFAULT
    if not _looks_like_ollama_base_url(normalized_base_url):
        return normalized_base_url
    if _provider_from_model_name(model) == 'ollama':
        return normalized_base_url
    parsed = urlparse(normalized_base_url)
    if parsed.path.rstrip('/'):
        return normalized_base_url
    return f'{normalized_base_url.rstrip("/")}/v1'


def _normalize_model_name(
    model: str | None,
    *,
    provider: str | None = None,
    base_url: str | None = None,
) -> str:
    if not isinstance(model, str):
        return _DEFAULT_MODEL
    normalized_model = clean_env_value(model).strip()
    if not normalized_model:
        return _DEFAULT_MODEL
    if '/' in normalized_model:
        return normalized_model

    # Loïc endpoint always wins — overrides LLM_PROVIDER (e.g. ollama_chat) because
    # LOIC_API_BASE is an OpenAI-compat passthrough, not a native Ollama server.
    if _provider_from_base_url(base_url) == 'loic':
        return f'openai/{normalized_model}'

    normalized_provider = _normalize_provider_name(provider)
    if normalized_provider is None:
        normalized_provider = _provider_from_base_url(base_url)
    if normalized_provider == 'ollama':
        if _looks_like_ollama_base_url(base_url):
            return f'openai/{normalized_model}'
        return f'ollama/{normalized_model}'
    if normalized_provider:
        return f'{normalized_provider}/{normalized_model}'
    return normalized_model


def _default_model_from_env() -> str:
    provider = _env_nonempty('LLM_PROVIDER')
    base_url = _env_nonempty('LOIC_API_BASE', 'LLM_API_BASE')
    # When the resolved base URL is a Loïc endpoint, prefer LOIC_MODEL so users
    # can keep per-provider model selections side-by-side in the same .env.
    if _provider_from_base_url(base_url) == 'loic':
        model = _env_first('LOIC_MODEL', 'LLM_MODEL', default=_DEFAULT_MODEL)
    else:
        model = _env_first('LLM_MODEL', default=_DEFAULT_MODEL)
    return _normalize_model_name(model, provider=provider, base_url=base_url)


def _add_agent_common_args(parser: argparse.ArgumentParser, *, include_backend: bool) -> None:
    parser.add_argument('--model', default=_default_model_from_env())
    if include_backend:
        parser.add_argument(
            '--base-url',
            default=_env_first('LOIC_API_BASE', 'LLM_API_BASE', default=_LOCAL_BASE_URL_DEFAULT),
        )
        parser.add_argument(
            '--api-key',
            default=None,
        )
        parser.add_argument(
            '--llm-backend',
            default=os.environ.get('CLAW_LLM_BACKEND', 'litellm'),
            help='LLM transport backend (litellm only)',
        )
        parser.add_argument(
            '--temperature',
            type=float,
            default=_env_float('AGENT_TEMPERATURE', 'LLM_TEMPERATURE', default=0.0),
        )
        parser.add_argument('--timeout-seconds', type=float, default=120.0)
        parser.add_argument('--input-cost-per-million', type=float, default=0.0)
        parser.add_argument('--output-cost-per-million', type=float, default=0.0)
    parser.add_argument('--cwd', default='.')
    parser.add_argument('--add-dir', action='append', default=[])
    parser.add_argument('--disable-claude-md', action='store_true')
    # File writes and shell commands are allowed by default. Use --no-write / --no-shell to disable.
    _write_default = not _env_flag_disabled('CLAW_NO_WRITE')
    _shell_default = not _env_flag_disabled('CLAW_NO_SHELL')
    parser.add_argument(
        '--allow-write',
        action='store_true',
        default=_write_default,
        help='Allow the agent to write files (default: on; disable with CLAW_NO_WRITE=1)',
    )
    parser.add_argument(
        '--no-write',
        action='store_false',
        dest='allow_write',
        help='Disallow file writes',
    )
    parser.add_argument(
        '--allow-shell',
        action='store_true',
        default=_shell_default,
        help='Allow the agent to run shell commands (default: on; disable with CLAW_NO_SHELL=1)',
    )
    parser.add_argument(
        '--no-shell',
        action='store_false',
        dest='allow_shell',
        help='Disallow shell commands',
    )
    parser.add_argument('--unsafe', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--auto-snip-threshold', type=int)
    parser.add_argument('--auto-compact-threshold', type=int)
    parser.add_argument('--compact-preserve-messages', type=int, default=4)
    parser.add_argument('--max-total-tokens', type=int)
    parser.add_argument('--max-input-tokens', type=int)
    parser.add_argument('--max-output-tokens', type=int, default=_env_optional_int('LLM_MAX_TOKENS'))
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
    parser.add_argument('--system-prompt', default=_env_first('LLM_SYSTEM_PROMPT'))
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
    cwd = Path(args.cwd).resolve()
    return AgentRuntimeConfig(
        cwd=cwd,
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
        session_directory=(cwd / '.port_sessions' / 'agent').resolve(),
        scratchpad_root=(
            Path(getattr(args, 'scratchpad_root')).resolve()
            if getattr(args, 'scratchpad_root', None)
            else (cwd / '.port_sessions' / 'scratchpad').resolve()
        ),
    )


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    base_url = getattr(args, 'base_url', None)
    if base_url is None:
        base_url = _env_first('LOIC_API_BASE', 'LLM_API_BASE', default=_LOCAL_BASE_URL_DEFAULT)
    base_url = clean_env_value(str(base_url))
    provider = _env_nonempty('LLM_PROVIDER')
    raw_model = getattr(args, 'model', None) or _default_model_from_env()
    model = clean_env_value(str(raw_model))
    normalized_model = _normalize_model_name(
        model,
        provider=provider,
        base_url=base_url,
    )
    normalized_base_url = _normalize_base_url(base_url, model=normalized_model)
    api_key = _resolve_api_key(
        explicit_api_key=getattr(args, 'api_key', None),
        provider=provider,
        model=normalized_model,
        base_url=normalized_base_url,
    )
    loic_headers: tuple[tuple[str, str], ...] = ()
    if _provider_from_base_url(normalized_base_url) == 'loic':
        loic_headers = (('X-API-Key', str(api_key)),)
    return ModelConfig(
        model=normalized_model,
        base_url=normalized_base_url,
        api_key=str(api_key),
        extra_headers=loic_headers,
        llm_backend=getattr(
            args,
            'llm_backend',
            os.environ.get('CLAW_LLM_BACKEND', 'litellm'),
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
    max_turns = getattr(args, 'max_turns', 12)
    if max_turns is not None:
        command.extend(['--max-turns', str(max_turns)])
    if include_backend:
        resolved_api_key = _resolve_api_key(
            explicit_api_key=getattr(args, 'api_key', None),
            provider=_env_nonempty('LLM_PROVIDER'),
            model=str(getattr(args, 'model', '')),
            base_url=str(getattr(args, 'base_url', _LOCAL_BASE_URL_DEFAULT)),
        )
        command.extend(['--model', str(args.model)])
        command.extend(['--base-url', str(args.base_url)])
        command.extend(['--api-key', resolved_api_key])
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
    parser.add_argument('--show-usage', action='store_true', help='print token/cost usage after the turn')
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
