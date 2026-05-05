from __future__ import annotations

import difflib
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Union

from src.agent.models.types import AgentPermissions, AgentRuntimeConfig, ToolExecutionResult

if TYPE_CHECKING:
    from src.agent.runtime.dependencies import AgentRuntimeDependencies
    from src.features.system.account_runtime import AccountRuntime
    from src.features.collaboration.ask_user_runtime import AskUserRuntime
    from src.features.system.config_runtime import ConfigRuntime
    from src.features.integration.lsp_runtime import LSPRuntime
    from src.features.integration.mcp_runtime import MCPRuntime
    from src.features.orchestration.plan_runtime import PlanRuntime
    from src.features.integration.remote_runtime import RemoteRuntime
    from src.features.integration.remote_trigger_runtime import RemoteTriggerRuntime
    from src.features.integration.search_runtime import SearchRuntime
    from src.features.orchestration.task_runtime import TaskRuntime
    from src.features.collaboration.team_runtime import TeamRuntime
    from src.features.orchestration.workflow_runtime import WorkflowRuntime
    from src.features.orchestration.worktree_runtime import WorktreeRuntime


class ToolPermissionError(RuntimeError):
    """Raised when the runtime configuration does not allow a tool action."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete because of invalid input or state."""


@dataclass(frozen=True)
class ToolExecutionContext:
    root: Path
    command_timeout_seconds: float
    max_output_chars: int
    permissions: AgentPermissions
    extra_env: dict[str, str] = field(default_factory=dict)
    tool_registry: dict[str, 'AgentTool'] | None = None
    search_runtime: 'SearchRuntime | None' = None
    account_runtime: 'AccountRuntime | None' = None
    ask_user_runtime: 'AskUserRuntime | None' = None
    config_runtime: 'ConfigRuntime | None' = None
    lsp_runtime: 'LSPRuntime | None' = None
    mcp_runtime: 'MCPRuntime | None' = None
    remote_runtime: 'RemoteRuntime | None' = None
    remote_trigger_runtime: 'RemoteTriggerRuntime | None' = None
    plan_runtime: 'PlanRuntime | None' = None
    task_runtime: 'TaskRuntime | None' = None
    team_runtime: 'TeamRuntime | None' = None
    workflow_runtime: 'WorkflowRuntime | None' = None
    worktree_runtime: 'WorktreeRuntime | None' = None


ToolHandler = Callable[
    [dict[str, Any], ToolExecutionContext],
    Union[str, tuple[str, dict[str, Any]]],
]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai_tool(self) -> dict[str, object]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        try:
            result = self.handler(arguments, context)
            if isinstance(result, tuple):
                content, metadata = result
            else:
                content, metadata = result, {}
            return ToolExecutionResult(name=self.name, ok=True, content=content, metadata=metadata)
        except ToolPermissionError as exc:
            return ToolExecutionResult(
                name=self.name,
                ok=False,
                content=str(exc),
                metadata={'error_kind': 'permission_denied'},
            )
        except (ToolExecutionError, OSError, subprocess.SubprocessError) as exc:
            return ToolExecutionResult(
                name=self.name,
                ok=False,
                content=str(exc),
                metadata={'error_kind': 'tool_execution_error'},
            )


@dataclass(frozen=True)
class ToolStreamUpdate:
    kind: str
    content: str = ''
    stream: str | None = None
    result: ToolExecutionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_tool_context(
    config: AgentRuntimeConfig,
    *,
    dependencies: 'AgentRuntimeDependencies | None' = None,
    extra_env: dict[str, str] | None = None,
    tool_registry: dict[str, AgentTool] | None = None,
    search_runtime: 'SearchRuntime | None' = None,
    account_runtime: 'AccountRuntime | None' = None,
    ask_user_runtime: 'AskUserRuntime | None' = None,
    config_runtime: 'ConfigRuntime | None' = None,
    lsp_runtime: 'LSPRuntime | None' = None,
    mcp_runtime: 'MCPRuntime | None' = None,
    remote_runtime: 'RemoteRuntime | None' = None,
    remote_trigger_runtime: 'RemoteTriggerRuntime | None' = None,
    plan_runtime: 'PlanRuntime | None' = None,
    task_runtime: 'TaskRuntime | None' = None,
    team_runtime: 'TeamRuntime | None' = None,
    workflow_runtime: 'WorkflowRuntime | None' = None,
    worktree_runtime: 'WorktreeRuntime | None' = None,
) -> ToolExecutionContext:
    if dependencies is not None:
        search_runtime = search_runtime or dependencies.search_runtime
        account_runtime = account_runtime or dependencies.account_runtime
        ask_user_runtime = ask_user_runtime or dependencies.ask_user_runtime
        config_runtime = config_runtime or dependencies.config_runtime
        lsp_runtime = lsp_runtime or dependencies.lsp_runtime
        mcp_runtime = mcp_runtime or dependencies.mcp_runtime
        remote_runtime = remote_runtime or dependencies.remote_runtime
        remote_trigger_runtime = (
            remote_trigger_runtime or dependencies.remote_trigger_runtime
        )
        plan_runtime = plan_runtime or dependencies.plan_runtime
        task_runtime = task_runtime or dependencies.task_runtime
        team_runtime = team_runtime or dependencies.team_runtime
        workflow_runtime = workflow_runtime or dependencies.workflow_runtime
        worktree_runtime = worktree_runtime or dependencies.worktree_runtime
    return ToolExecutionContext(
        root=config.cwd.resolve(),
        command_timeout_seconds=config.command_timeout_seconds,
        max_output_chars=config.max_output_chars,
        permissions=config.permissions,
        extra_env=dict(extra_env or {}),
        tool_registry=tool_registry,
        search_runtime=search_runtime,
        account_runtime=account_runtime,
        ask_user_runtime=ask_user_runtime,
        config_runtime=config_runtime,
        lsp_runtime=lsp_runtime,
        mcp_runtime=mcp_runtime,
        remote_runtime=remote_runtime,
        remote_trigger_runtime=remote_trigger_runtime,
        plan_runtime=plan_runtime,
        task_runtime=task_runtime,
        team_runtime=team_runtime,
        workflow_runtime=workflow_runtime,
        worktree_runtime=worktree_runtime,
    )


def execute_tool(
    tool_registry: dict[str, AgentTool],
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    tool = tool_registry.get(name)
    if tool is None:
        return _handle_unknown_tool(name, arguments, context)
    return tool.execute(arguments, context)


def execute_tool_streaming(
    tool_registry: dict[str, AgentTool],
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> Iterator[ToolStreamUpdate]:
    tool = tool_registry.get(name)
    if tool is None:
        result = _handle_unknown_tool(name, arguments, context)
        yield ToolStreamUpdate(
            kind='result',
            result=result,
        )
        return

    if name == 'bash':
        yield from _stream_bash(arguments, context)
        return

    result = tool.execute(arguments, context)
    if result.ok and result.content and name != 'delegate_agent':
        yield from _stream_static_text_result(result)
        return
    yield ToolStreamUpdate(kind='result', result=result)


def _handle_unknown_tool(
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    mcp_runtime = context.mcp_runtime
    if mcp_runtime is not None:
        matched_server = mcp_runtime.get_server(name)
        if matched_server is None:
            try:
                content, metadata = mcp_runtime.call_tool(
                    name,
                    arguments=arguments,
                    max_chars=context.max_output_chars,
                )
            except FileNotFoundError:
                pass
            except OSError as exc:
                return ToolExecutionResult(
                    name=name,
                    ok=False,
                    content=f'Failed to call MCP tool alias {name}: {exc}',
                    metadata={
                        'action': 'mcp_tool_alias',
                        'error_kind': 'tool_execution_error',
                    },
                )
            else:
                return ToolExecutionResult(
                    name=name,
                    ok=not bool(metadata.get('is_error')),
                    content=content,
                    metadata={
                        'action': 'mcp_tool_alias',
                        'server_name': metadata.get('server_name'),
                        'tool_name': metadata.get('tool_name'),
                        'mcp_is_error': metadata.get('is_error'),
                    },
                )

    return ToolExecutionResult(
        name=name,
        ok=False,
        content=_render_unknown_tool_message(name, context),
        metadata={'error_kind': 'unknown_tool'},
    )


def _render_unknown_tool_message(name: str, context: ToolExecutionContext) -> str:
    lines = [f'Unknown tool: {name}']
    suggestions = _suggest_tool_names(name, context)
    if suggestions:
        lines.append('Closest available tools: ' + ', '.join(suggestions))

    mcp_runtime = context.mcp_runtime
    if mcp_runtime is not None:
        matched_server = mcp_runtime.get_server(name)
        if matched_server is not None:
            lines.append(
                f'`{matched_server.name}` is a configured MCP server, not a direct tool.'
            )
            lines.append(
                'Use `mcp_list_tools` with `{"server": "'
                + matched_server.name
                + '"}` to discover its tools, then call `mcp_call_tool` with '
                + '`{"server": "'
                + matched_server.name
                + '", "tool_name": "...", "arguments": {...}}`.'
            )
        elif mcp_runtime.servers:
            lines.append(
                'If you intended to use an MCP server tool, discover it with `mcp_list_tools` '
                'and invoke it with `mcp_call_tool`.'
            )

    if not suggestions:
        lines.append('Use `list_available_tools` if you are unsure which tools exist in this session.')
    return '\n'.join(lines)


def _suggest_tool_names(name: str, context: ToolExecutionContext) -> list[str]:
    candidates = sorted((context.tool_registry or {}).keys())
    suggestions = difflib.get_close_matches(name, candidates, n=5, cutoff=0.45)
    if context.mcp_runtime is not None and context.mcp_runtime.get_server(name) is not None:
        for candidate in ('mcp_list_tools', 'mcp_call_tool', 'list_available_tools'):
            if candidate in candidates and candidate not in suggestions:
                suggestions.append(candidate)
    return suggestions[:5]


def default_tool_registry() -> dict[str, AgentTool]:
    from src.agent.tools.registry import default_tool_registry as _default_tool_registry

    return _default_tool_registry()


def serialize_tool_result(result: ToolExecutionResult) -> str:
    payload = {
        'tool': result.name,
        'ok': result.ok,
        'content': result.content,
    }
    if result.metadata:
        payload['metadata'] = result.metadata
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f'{head}\n...[truncated]...\n{tail}'


def _snapshot_text(text: str, limit: int = 240) -> str:
    normalized = ' '.join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + '...'


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f'{key} must be a non-empty string')
    return value


def _coerce_int(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolExecutionError(f'{key} must be an integer')
    return value


def _coerce_float(arguments: dict[str, Any], key: str, default: float) -> float:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolExecutionError(f'{key} must be a number')
    return float(value)


def _resolve_path(raw_path: str, context: ToolExecutionContext, *, allow_missing: bool = True) -> Path:
    expanded = Path(raw_path).expanduser()
    candidate = expanded if expanded.is_absolute() else context.root / expanded
    resolved = candidate.resolve(strict=not allow_missing)
    try:
        resolved.relative_to(context.root)
    except ValueError as exc:
        raise ToolExecutionError(
            f'Path {raw_path!r} escapes the workspace root {context.root}'
        ) from exc
    return resolved


def _ensure_write_allowed(context: ToolExecutionContext) -> None:
    if not context.permissions.allow_file_write:
        raise ToolPermissionError(
            'File write tools are disabled. Re-run with --allow-write to enable edits.'
        )


def _ensure_shell_allowed(command: str, context: ToolExecutionContext) -> None:
    if not context.permissions.allow_shell_commands:
        raise ToolPermissionError(
            'Shell commands are disabled. Re-run with --allow-shell to enable bash.'
        )
    if context.permissions.allow_destructive_shell_commands:
        return
    destructive_patterns = [
        r'(^|[;&|])\s*rm\s',
        r'(^|[;&|])\s*mv\s',
        r'(^|[;&|])\s*dd\s',
        r'(^|[;&|])\s*shutdown\s',
        r'(^|[;&|])\s*reboot\s',
        r'(^|[;&|])\s*mkfs',
        r'(^|[;&|])\s*chmod\s+-r\s+777',
        r'(^|[;&|])\s*chown\s+-r',
        r'(^|[;&|])\s*git\s+reset\s+--hard',
        r'(^|[;&|])\s*git\s+clean\s+-fd',
        r'(^|[;&|])\s*:\s*>\s*',
    ]
    lowered = command.lower()
    if any(re.search(pattern, lowered) for pattern in destructive_patterns):
        raise ToolPermissionError(
            'Potentially destructive shell command blocked. Re-run with --unsafe to allow it.'
        )


def _list_dir(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    raw_path = arguments.get('path', '.')
    if not isinstance(raw_path, str):
        raise ToolExecutionError('path must be a string')
    max_entries = _coerce_int(arguments, 'max_entries', 200)
    target = _resolve_path(raw_path, context)
    if not target.exists():
        raise ToolExecutionError(f'Path not found: {raw_path}')
    if not target.is_dir():
        raise ToolExecutionError(f'Path is not a directory: {raw_path}')
    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    lines: list[str] = []
    for entry in entries[:max_entries]:
        kind = 'dir' if entry.is_dir() else 'file'
        rel = entry.relative_to(context.root)
        lines.append(f'{kind}\t{rel}')
    if len(entries) > max_entries:
        lines.append(f'... truncated at {max_entries} entries ...')
    return '\n'.join(lines) if lines else '(empty directory)'


def _read_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False)
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    text = target.read_text(encoding='utf-8', errors='replace')
    start_line = arguments.get('start_line')
    end_line = arguments.get('end_line')
    if start_line is not None and (isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1):
        raise ToolExecutionError('start_line must be an integer >= 1')
    if end_line is not None and (isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1):
        raise ToolExecutionError('end_line must be an integer >= 1')
    lines = text.splitlines()
    total_lines = len(lines)
    start_idx = max((start_line or 1) - 1, 0)
    end_idx = end_line if end_line is not None else total_lines
    selected = lines[start_idx:end_idx]
    rel = target.relative_to(context.root)
    if start_line is not None or end_line is not None:
        actual_end = min(end_idx, total_lines)
        header = f'# File: {rel} (lines {start_idx + 1}-{actual_end} of {total_lines})'
    else:
        header = f'# File: {rel} ({total_lines} lines)'
    numbered = '\n'.join(f'{start_idx + idx + 1}: {line}' for idx, line in enumerate(selected))
    rendered = f'{header}\n{numbered}'
    return _truncate_output(rendered, context.max_output_chars)


def _write_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context)
    content = arguments.get('content')
    if not isinstance(content, str):
        raise ToolExecutionError('content must be a string')
    previous_text: str | None = None
    previous_sha256: str | None = None
    if target.exists() and target.is_file():
        previous_text = target.read_text(encoding='utf-8', errors='replace')
        previous_sha256 = hashlib.sha256(previous_text.encode('utf-8')).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    rel = target.relative_to(context.root)
    new_sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()
    return (
        f'wrote {rel} ({len(content)} chars)',
        {
            'action': 'write_file',
            'path': str(rel),
            'before_exists': previous_text is not None,
            'before_sha256': previous_sha256,
            'before_size': len(previous_text) if previous_text is not None else 0,
            'before_preview': (
                _snapshot_text(previous_text)
                if previous_text is not None
                else None
            ),
            'after_sha256': new_sha256,
            'after_size': len(content),
            'after_preview': _snapshot_text(content),
            'content_length': len(content),
        },
    )


def _edit_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False)
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    # Accept old_str (primary, Claude Code convention) or old_text (legacy alias)
    _raw_old = arguments.get('old_str')
    old_str = _raw_old if _raw_old is not None else arguments.get('old_text')
    _raw_new = arguments.get('new_str')
    new_str = _raw_new if _raw_new is not None else arguments.get('new_text')
    replace_all = arguments.get('replace_all', False)
    if not isinstance(old_str, str):
        raise ToolExecutionError('old_str must be a string')
    if not isinstance(new_str, str):
        raise ToolExecutionError('new_str must be a string')
    if not isinstance(replace_all, bool):
        raise ToolExecutionError('replace_all must be a boolean')
    current = target.read_text(encoding='utf-8', errors='replace')
    occurrences = current.count(old_str)
    if occurrences == 0:
        first_line = old_str.splitlines()[0].strip() if old_str.strip() else ''
        hint = ''
        if first_line:
            file_lines = current.splitlines()
            nearby = [
                f'  line {i + 1}: {line}'
                for i, line in enumerate(file_lines)
                if first_line[:60] in line
            ][:5]
            if nearby:
                hint = '\n\nLines containing the start of old_str:\n' + '\n'.join(nearby)
        raise ToolExecutionError(
            f'old_str not found in {target.name!r}. '
            f'It must match the file content exactly (including whitespace and indentation).'
            + hint
        )
    if occurrences > 1 and not replace_all:
        raise ToolExecutionError(
            f'old_str matched {occurrences} times; pass replace_all=true to replace every match, '
            f'or include more surrounding context to uniquely identify the location'
        )
    before_sha256 = hashlib.sha256(current.encode('utf-8')).hexdigest()
    updated = current.replace(old_str, new_str) if replace_all else current.replace(old_str, new_str, 1)
    target.write_text(updated, encoding='utf-8')
    rel = target.relative_to(context.root)
    replaced = occurrences if replace_all else 1
    after_sha256 = hashlib.sha256(updated.encode('utf-8')).hexdigest()
    return (
        f'edited {rel}; replaced {replaced} occurrence(s)',
        {
            'action': 'edit_file',
            'path': str(rel),
            'before_sha256': before_sha256,
            'after_sha256': after_sha256,
            'before_size': len(current),
            'after_size': len(updated),
            'before_preview': _snapshot_text(current),
            'after_preview': _snapshot_text(updated),
            'old_str_preview': _snapshot_text(old_str),
            'new_str_preview': _snapshot_text(new_str),
            'replaced_occurrences': replaced,
        },
    )


def _multi_edit(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    """Apply multiple sequential edits to a single file in one tool call."""
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False)
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    raw_edits = arguments.get('edits')
    if not isinstance(raw_edits, list) or not raw_edits:
        raise ToolExecutionError('edits must be a non-empty array of {old_str, new_str} objects')
    current = target.read_text(encoding='utf-8', errors='replace')
    before_sha256 = hashlib.sha256(current.encode('utf-8')).hexdigest()
    updated = current
    for i, edit in enumerate(raw_edits):
        if not isinstance(edit, dict):
            raise ToolExecutionError(f'edits[{i}] must be an object with old_str and new_str')
        _raw_old = edit.get('old_str')
        old_str = _raw_old if _raw_old is not None else edit.get('old_text')
        _raw_new = edit.get('new_str')
        new_str = _raw_new if _raw_new is not None else edit.get('new_text')
        if not isinstance(old_str, str):
            raise ToolExecutionError(f'edits[{i}].old_str must be a string')
        if not isinstance(new_str, str):
            raise ToolExecutionError(f'edits[{i}].new_str must be a string')
        occurrences = updated.count(old_str)
        if occurrences == 0:
            raise ToolExecutionError(
                f'edits[{i}].old_str was not found in the file. '
                f'Note: edits are applied sequentially, so earlier edits may affect later matches.'
            )
        if occurrences > 1:
            raise ToolExecutionError(
                f'edits[{i}].old_str matched {occurrences} times; '
                f'include more surrounding context to uniquely identify the location'
            )
        updated = updated.replace(old_str, new_str, 1)
    target.write_text(updated, encoding='utf-8')
    rel = target.relative_to(context.root)
    after_sha256 = hashlib.sha256(updated.encode('utf-8')).hexdigest()
    return (
        f'edited {rel}; applied {len(raw_edits)} edit(s)',
        {
            'action': 'multi_edit',
            'path': str(rel),
            'before_sha256': before_sha256,
            'after_sha256': after_sha256,
            'before_size': len(current),
            'after_size': len(updated),
            'edits_applied': len(raw_edits),
        },
    )


def _notebook_edit(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False)
    if target.suffix != '.ipynb':
        raise ToolExecutionError('notebook_edit requires a .ipynb target')
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    cell_index = arguments.get('cell_index')
    if isinstance(cell_index, bool) or not isinstance(cell_index, int) or cell_index < 0:
        raise ToolExecutionError('cell_index must be an integer >= 0')
    source = arguments.get('source')
    if not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    cell_type = arguments.get('cell_type', 'code')
    if cell_type is not None and not isinstance(cell_type, str):
        raise ToolExecutionError('cell_type must be a string')
    create_cell = arguments.get('create_cell', False)
    if not isinstance(create_cell, bool):
        raise ToolExecutionError('create_cell must be a boolean')

    raw = target.read_text(encoding='utf-8', errors='replace')
    before_sha256 = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f'Notebook is not valid JSON: {target}') from exc
    if not isinstance(notebook, dict):
        raise ToolExecutionError('Notebook payload must be a JSON object')
    cells = notebook.get('cells')
    if not isinstance(cells, list):
        raise ToolExecutionError('Notebook does not contain a cells array')

    while len(cells) <= cell_index:
        if not create_cell:
            raise ToolExecutionError(
                f'Notebook cell {cell_index} does not exist; pass create_cell=true to append missing cells'
            )
        cells.append(
            {
                'cell_type': cell_type or 'code',
                'metadata': {},
                'source': [],
                'outputs': [],
                'execution_count': None,
            }
        )
    cell = cells[cell_index]
    if not isinstance(cell, dict):
        raise ToolExecutionError(f'Notebook cell {cell_index} is not a JSON object')
    existing_type = cell.get('cell_type')
    if not isinstance(existing_type, str):
        existing_type = 'code'
    source_lines = source.splitlines(keepends=True)
    if source and not source.endswith('\n'):
        source_lines = [*source_lines[:-1], source_lines[-1]]
    cell['cell_type'] = cell_type or existing_type
    cell['source'] = source_lines
    if cell['cell_type'] == 'code':
        cell.setdefault('outputs', [])
        cell.setdefault('execution_count', None)
    updated = json.dumps(notebook, ensure_ascii=True, indent=1) + '\n'
    target.write_text(updated, encoding='utf-8')
    after_sha256 = hashlib.sha256(updated.encode('utf-8')).hexdigest()
    rel = target.relative_to(context.root)
    return (
        f'updated notebook cell {cell_index} in {rel}',
        {
            'action': 'notebook_edit',
            'path': str(rel),
            'cell_index': cell_index,
            'cell_type': cell['cell_type'],
            'before_sha256': before_sha256,
            'after_sha256': after_sha256,
            'before_preview': _snapshot_text(raw),
            'after_preview': _snapshot_text(updated),
        },
    )


def _glob_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    pattern = _require_string(arguments, 'pattern')
    matches = sorted(context.root.glob(pattern))
    if not matches:
        return '(no matches)'
    root_resolved = context.root.resolve()
    validated: list[str] = []
    for path in matches:
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError:
            continue
        validated.append(str(path.relative_to(context.root)))
    if not validated:
        return '(no matches)'
    return _truncate_output('\n'.join(validated), context.max_output_chars)


def _grep_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    pattern = _require_string(arguments, 'pattern')
    raw_path = arguments.get('path', '.')
    if not isinstance(raw_path, str):
        raise ToolExecutionError('path must be a string')
    literal = arguments.get('literal', False)
    if not isinstance(literal, bool):
        raise ToolExecutionError('literal must be a boolean')
    max_matches = _coerce_int(arguments, 'max_matches', 100)
    root = _resolve_path(raw_path, context)
    if not root.exists():
        raise ToolExecutionError(f'Path not found: {raw_path}')
    try:
        regex = re.compile(re.escape(pattern) if literal else pattern)
    except re.error as exc:
        raise ToolExecutionError(f'Invalid regex pattern: {exc}') from exc
    hits: list[str] = []
    file_iter = root.rglob('*') if root.is_dir() else [root]
    for file_path in file_iter:
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = file_path.relative_to(context.root)
                hits.append(f'{rel}:{line_no}: {line}')
                if len(hits) >= max_matches:
                    return '\n'.join(hits + [f'... truncated at {max_matches} matches ...'])
    return '\n'.join(hits) if hits else '(no matches)'


def _format_bash_output(stdout: str, stderr: str, exit_code: int) -> str:
    """Format bash output cleanly: show exit code only on failure, label stderr."""
    parts: list[str] = []
    if exit_code != 0:
        parts.append(f'[exit {exit_code}]')
    if stdout.strip():
        parts.append(stdout.rstrip())
    if stderr.strip():
        parts.append(f'[stderr]\n{stderr.rstrip()}')
    if not parts:
        parts.append('[no output]')
    return '\n'.join(parts)


def _run_bash(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    command = _require_string(arguments, 'command')
    _ensure_shell_allowed(command, context)
    raw_timeout = arguments.get('timeout')
    if raw_timeout is not None:
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            raise ToolExecutionError('timeout must be a positive number (seconds)')
        timeout_seconds = max(1.0, min(float(raw_timeout), 600.0))
    else:
        timeout_seconds = context.command_timeout_seconds
    bash_command, working_directory = _build_bash_invocation(
        command,
        context.root,
        extra_env=context.extra_env,
    )
    completed = subprocess.run(
        bash_command,
        shell=False,
        cwd=working_directory,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=_build_subprocess_env(context),
    )
    stdout = completed.stdout or ''
    stderr = completed.stderr or ''
    output = _format_bash_output(stdout, stderr, completed.returncode)
    return (
        _truncate_output(output, context.max_output_chars),
        {
            'action': 'bash',
            'command': command,
            'exit_code': completed.returncode,
            'stdout_preview': _snapshot_text(stdout),
            'stderr_preview': _snapshot_text(stderr),
            'output_preview': _snapshot_text(output),
        },
    )


def _web_fetch(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    raw_url = _require_string(arguments, 'url')
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in {'http', 'https'}:
        raise ToolExecutionError('url must use http or https scheme')
    request = urllib.request.Request(
        raw_url,
        headers={'User-Agent': 'claw-code-agent/1.0'},
    )
    try:
        with urllib.request.urlopen(request, timeout=context.command_timeout_seconds) as response:
            raw_bytes = response.read(max_chars + 1)
            content_type = response.headers.get_content_type() if hasattr(response, 'headers') else None
            text = raw_bytes.decode('utf-8', errors='replace')
    except (urllib.error.URLError, OSError) as exc:
        raise ToolExecutionError(f'Failed to fetch {raw_url}: {exc}') from exc
    truncated = len(text) > max_chars
    rendered = text[:max_chars]
    if truncated:
        rendered += '\n...[truncated]...'
    return (
        _truncate_output(rendered, context.max_output_chars),
        {
            'action': 'web_fetch',
            'url': raw_url,
            'content_type': content_type,
            'truncated': truncated,
            'fetched_chars': len(text),
            'preview': _snapshot_text(rendered),
        },
    )


def _search_status(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    provider = arguments.get('provider')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    if provider:
        return runtime.render_provider(provider)
    return '\n'.join(['# Search', '', runtime.render_summary()])


def _search_list_providers(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_providers = _coerce_int(arguments, 'max_providers', 20)
    providers = runtime.list_providers(query=query, limit=max_providers)
    lines = ['# Search Providers', '']
    if not providers:
        lines.append('No local search providers discovered.')
        return '\n'.join(lines)
    current = runtime.current_provider()
    current_name = current.name if current is not None else None
    for provider in providers:
        details = [provider.name, provider.provider, provider.base_url]
        if provider.api_key_env:
            details.append(f'api_key_env={provider.api_key_env}')
        if current_name == provider.name:
            details.append('active=true')
        lines.append('- ' + ' ; '.join(details))
    return '\n'.join(lines)


def _search_activate_provider(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_search_runtime(context)
    provider = _require_string(arguments, 'provider')
    try:
        report = runtime.activate_provider(provider)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown search provider: {provider}') from exc
    return (
        '\n'.join(['# Search', '', report.as_text()]),
        {
            'action': 'search_activate_provider',
            'provider': report.provider_name,
            'provider_kind': report.provider_kind,
            'base_url': report.base_url,
        },
    )


def _web_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    query = _require_string(arguments, 'query')
    provider = arguments.get('provider')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    raw_domains = arguments.get('domains', ())
    if raw_domains is None:
        raw_domains = ()
    if not isinstance(raw_domains, (list, tuple)):
        raise ToolExecutionError('domains must be an array of strings')
    domains: tuple[str, ...] = tuple(
        item.strip() for item in raw_domains if isinstance(item, str) and item.strip()
    )
    if len(domains) != len(raw_domains):
        raise ToolExecutionError('domains must contain only non-empty strings')
    max_results = arguments.get('max_results')
    if max_results is None:
        max_results = runtime.default_max_results
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ToolExecutionError('max_results must be an integer')
    if max_results < 1 or max_results > 20:
        raise ToolExecutionError('max_results must be between 1 and 20')
    try:
        selected_provider, results = runtime.search(
            query,
            provider_name=provider,
            max_results=max_results,
            domains=domains,
            timeout_seconds=context.command_timeout_seconds,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown search provider: {provider or exc.args[0]}') from exc
    except LookupError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except (ValueError, OSError, urllib.error.URLError) as exc:
        raise ToolExecutionError(f'Web search failed: {exc}') from exc
    lines = ['# Web Search', '']
    lines.append(f'- Provider: {selected_provider.name} ({selected_provider.provider})')
    lines.append(f'- Query: {query}')
    lines.append(f'- Results: {len(results)}')
    if domains:
        lines.append('- Domains: ' + ', '.join(domains))
    lines.append('')
    if not results:
        lines.append('No search results.')
    else:
        for result in results:
            lines.append(f'{result.rank}. {result.title}')
            lines.append(f'   {result.url}')
            if result.snippet:
                lines.append(f'   {result.snippet}')
    rendered = '\n'.join(lines)
    return (
        rendered,
        {
            'action': 'web_search',
            'provider': selected_provider.name,
            'provider_kind': selected_provider.provider,
            'query': query,
            'result_count': len(results),
            'domains': list(domains),
            'top_urls': [result.url for result in results[:5]],
        },
    )


def _tool_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    query = _require_string(arguments, 'query').lower()
    max_results = _coerce_int(arguments, 'max_results', 20)
    registry = context.tool_registry or default_tool_registry()
    matches: list[tuple[str, str]] = []
    for tool in registry.values():
        haystack = f'{tool.name} {tool.description}'.lower()
        if query in haystack:
            matches.append((tool.name, tool.description))
    if not matches:
        return '(no matching tools)'
    lines = ['# Tool Search', '']
    for name, description in matches[:max_results]:
        lines.append(f'- `{name}`: {description}')
    return '\n'.join(lines)


def _list_available_tools(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_results = _coerce_int(arguments, 'max_results', 50)
    registry = context.tool_registry or default_tool_registry()
    lowered_query = query.lower() if isinstance(query, str) and query.strip() else None
    matches: list[tuple[str, str]] = []
    for tool in registry.values():
        haystack = f'{tool.name} {tool.description}'.lower()
        if lowered_query is not None and lowered_query not in haystack:
            continue
        matches.append((tool.name, tool.description))
    if not matches:
        return '(no matching tools)'
    lines = ['# Available Tools', '']
    for name, description in matches[:max_results]:
        lines.append(f'- `{name}`: {description}')
    return '\n'.join(lines)


def _sleep(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    seconds = _coerce_float(arguments, 'seconds', 0.0)
    if seconds < 0.0 or seconds > 5.0:
        raise ToolExecutionError('seconds must be between 0.0 and 5.0')
    time.sleep(seconds)
    return (
        f'slept for {seconds:.3f} seconds',
        {
            'action': 'sleep',
            'seconds': seconds,
        },
    )


def _ask_user_question(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_ask_user_runtime(context)
    question = _require_string(arguments, 'question')
    header = arguments.get('header')
    question_id = arguments.get('question_id')
    if header is not None and not isinstance(header, str):
        raise ToolExecutionError('header must be a string')
    if question_id is not None and not isinstance(question_id, str):
        raise ToolExecutionError('question_id must be a string')
    raw_choices = arguments.get('choices', ())
    if raw_choices is None:
        raw_choices = ()
    if not isinstance(raw_choices, (list, tuple)):
        raise ToolExecutionError('choices must be an array of strings')
    choices = tuple(
        item.strip()
        for item in raw_choices
        if isinstance(item, str) and item.strip()
    )
    if len(choices) != len(raw_choices):
        raise ToolExecutionError('choices must contain only non-empty strings')
    allow_free_text = arguments.get('allow_free_text', True)
    if not isinstance(allow_free_text, bool):
        raise ToolExecutionError('allow_free_text must be a boolean')
    try:
        response = runtime.answer(
            question=question,
            choices=choices,
            question_id=question_id,
            header=header,
            allow_free_text=allow_free_text,
        )
    except LookupError as exc:
        raise ToolExecutionError(str(exc)) from exc
    lines = ['# Ask User', '']
    if header:
        lines.append(f'- Header: {header}')
    if question_id:
        lines.append(f'- Question ID: {question_id}')
    lines.append(f'- Question: {question}')
    if choices:
        lines.append('- Choices: ' + ', '.join(choices))
    lines.extend(
        [
            f'- Source: {response.source}',
            '',
            response.answer,
        ]
    )
    return (
        '\n'.join(lines),
        {
            'action': 'ask_user_question',
            'question': question,
            'question_id': question_id,
            'header': header,
            'source': response.source,
            'answer_preview': _snapshot_text(response.answer),
            'choices': list(choices),
        },
    )


def _account_status(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    profile = arguments.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise ToolExecutionError('profile must be a string')
    if profile:
        return runtime.render_profile(profile)
    return '\n'.join(['# Account', '', runtime.render_summary()])


def _account_list_profiles(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_profiles = _coerce_int(arguments, 'max_profiles', 20)
    profiles = runtime.list_profiles(query=query, limit=max_profiles)
    lines = ['# Account Profiles', '']
    if not profiles:
        lines.append('No local account profiles discovered.')
        return '\n'.join(lines)
    for profile in profiles:
        details = [profile.name, profile.provider, profile.identity]
        if profile.org:
            details.append(f'org={profile.org}')
        if profile.auth_mode:
            details.append(f'auth_mode={profile.auth_mode}')
        lines.append('- ' + ' ; '.join(details))
    return '\n'.join(lines)


def _account_login(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    target = _require_string(arguments, 'target')
    provider = arguments.get('provider')
    auth_mode = arguments.get('auth_mode')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    if auth_mode is not None and not isinstance(auth_mode, str):
        raise ToolExecutionError('auth_mode must be a string')
    report = runtime.login(target, provider=provider, auth_mode=auth_mode)
    return '\n'.join(['# Account', '', report.as_text()])


def _account_logout(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    reason = arguments.get('reason', 'tool_logout')
    if not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    report = runtime.logout(reason=reason)
    return '\n'.join(['# Account', '', report.as_text()])


def _config_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_config_runtime(context)
    source = arguments.get('source')
    prefix = arguments.get('prefix')
    if source is not None and not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    if prefix is not None and not isinstance(prefix, str):
        raise ToolExecutionError('prefix must be a string')
    limit = _coerce_int(arguments, 'limit', 100)
    try:
        rendered = runtime.render_keys(source=source, prefix=prefix, limit=limit)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config source: {exc.args[0]}') from exc
    return '\n'.join(['# Config Keys', '', rendered])


def _config_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_config_runtime(context)
    key_path = _require_string(arguments, 'key_path')
    source = arguments.get('source')
    if source is not None and not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    try:
        rendered = runtime.render_value(key_path, source=source)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config key or source: {exc.args[0]}') from exc
    return '\n'.join(['# Config Value', '', rendered])


def _config_set(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_config_runtime(context)
    key_path = _require_string(arguments, 'key_path')
    source = arguments.get('source', 'local')
    if not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    if 'value' not in arguments:
        raise ToolExecutionError('value is required')
    value = arguments.get('value')
    try:
        mutation = runtime.set_value(key_path, value, source=source)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config source: {exc.args[0]}') from exc
    return (
        f'set {key_path} in {mutation.source_name} config',
        _config_mutation_metadata(mutation, value),
    )


def _mcp_list_resources(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_resources = _coerce_int(arguments, 'max_resources', 50)
    resources = runtime.list_resources(query=query, limit=max_resources)
    if not resources:
        return '(no MCP resources)'
    lines: list[str] = []
    for resource in resources:
        details = [resource.uri, f'server={resource.server_name}']
        if resource.name:
            details.append(f'name={resource.name}')
        if resource.mime_type:
            details.append(f'mime={resource.mime_type}')
        if resource.resolved_path:
            details.append(f'path={resource.resolved_path}')
        lines.append(' ; '.join(details))
    return '\n'.join(lines)


def _mcp_read_resource(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    uri = _require_string(arguments, 'uri')
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    try:
        content = runtime.read_resource(uri, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return content


def _mcp_list_tools(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    query = arguments.get('query')
    server = arguments.get('server')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    if server is not None and not isinstance(server, str):
        raise ToolExecutionError('server must be a string')
    if server is not None and runtime.get_server(server) is None:
        raise ToolExecutionError(f'Unknown MCP server: {server}')
    max_tools = _coerce_int(arguments, 'max_tools', 50)
    tools = runtime.list_tools(query=query, server_name=server, limit=max_tools)
    if not tools:
        return '(no MCP tools)'
    lines: list[str] = []
    for tool in tools:
        details = [tool.name, f'server={tool.server_name}']
        if tool.description:
            details.append(tool.description)
        lines.append(' ; '.join(details))
    return '\n'.join(lines)


def _mcp_call_tool(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    tool_name = _require_string(arguments, 'tool_name')
    server = arguments.get('server')
    if server is not None and not isinstance(server, str):
        raise ToolExecutionError('server must be a string')
    raw_arguments = arguments.get('arguments', {})
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolExecutionError('arguments must be an object')
    arguments_preview = _snapshot_text(
        json.dumps(raw_arguments, ensure_ascii=True, sort_keys=True)
    )
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    try:
        content, metadata = runtime.call_tool(
            tool_name,
            arguments=raw_arguments,
            server_name=server,
            max_chars=max_chars,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        content,
        {
            'action': 'mcp_call_tool',
            'tool_name': tool_name,
            'server_name': metadata.get('server_name'),
            'requested_server': server,
            'arguments_preview': arguments_preview,
            'mcp_is_error': metadata.get('is_error'),
        },
    )


def _remote_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_remote_runtime(context)
    profile = arguments.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise ToolExecutionError('profile must be a string')
    if profile:
        return runtime.render_profile(profile)
    return '\n'.join(['# Remote', '', runtime.render_summary()])


def _remote_list_profiles(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_remote_runtime(context)
    query = arguments.get('query')
    mode = arguments.get('mode')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    if mode is not None and not isinstance(mode, str):
        raise ToolExecutionError('mode must be a string')
    max_profiles = _coerce_int(arguments, 'max_profiles', 20)
    return runtime.render_profiles_index(query=query, mode=mode, limit=max_profiles)


def _remote_connect(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_runtime(context)
    target = _require_string(arguments, 'target')
    mode = arguments.get('mode')
    if mode is not None and not isinstance(mode, str):
        raise ToolExecutionError('mode must be a string')
    report = runtime.connect(target, mode=mode)
    return (
        report.as_text(),
        {
            'action': 'remote_connect',
            'mode': report.mode,
            'target': report.target or target,
            'profile_name': report.profile_name,
            'session_url': report.session_url,
            'workspace_cwd': report.workspace_cwd,
        },
    )


def _remote_disconnect(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_runtime(context)
    reason = arguments.get('reason')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    report = runtime.disconnect(reason=reason or 'tool_request')
    return (
        report.as_text(),
        {
            'action': 'remote_disconnect',
            'mode': report.mode,
            'target': report.target,
        },
    )


def _worktree_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    _ = arguments
    runtime = _require_worktree_runtime(context)
    return '\n'.join(['# Worktree', '', runtime.render_summary()])


def _worktree_enter(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_worktree_runtime(context)
    name = arguments.get('name')
    if name is not None and not isinstance(name, str):
        raise ToolExecutionError('name must be a string')
    try:
        report = runtime.enter(name=name)
    except (RuntimeError, ValueError) as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        report.as_text(),
        {
            'action': 'worktree_enter',
            'cwd_update': report.worktree_path,
            'repo_root': report.repo_root,
            'worktree_path': report.worktree_path,
            'worktree_branch': report.worktree_branch,
            'session_name': report.session_name,
            'before_cwd': report.original_cwd,
            'after_cwd': report.worktree_path,
            'path': report.worktree_path,
        },
    )


def _worktree_exit(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_worktree_runtime(context)
    action = arguments.get('action', 'keep')
    discard_changes = arguments.get('discard_changes', False)
    if not isinstance(action, str):
        raise ToolExecutionError('action must be a string')
    if not isinstance(discard_changes, bool):
        raise ToolExecutionError('discard_changes must be a boolean')
    try:
        report = runtime.exit(
            action=action,
            discard_changes=discard_changes,
        )
    except (RuntimeError, ValueError) as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        report.as_text(),
        {
            'action': 'worktree_exit',
            'cwd_update': report.original_cwd or report.current_cwd,
            'repo_root': report.repo_root,
            'worktree_path': report.worktree_path,
            'worktree_branch': report.worktree_branch,
            'session_name': report.session_name,
            'before_cwd': report.worktree_path,
            'after_cwd': report.original_cwd or report.current_cwd,
            'exit_action': report.metadata.get('action'),
            'discard_changes': report.metadata.get('discard_changes'),
            'path': report.worktree_path,
        },
    )


def _workflow_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_workflow_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_workflows = _coerce_int(arguments, 'max_workflows', 50)
    workflows = runtime.list_workflows(query=query, limit=max_workflows)
    lines = ['# Workflows', '']
    if not workflows:
        lines.append('No local workflows discovered.')
        return '\n'.join(lines)
    for workflow in workflows:
        description = workflow.description or 'No description.'
        lines.append(f'- {workflow.name} ; steps={len(workflow.steps)} ; {description}')
    return '\n'.join(lines)


def _workflow_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_workflow_runtime(context)
    workflow_name = _require_string(arguments, 'workflow_name')
    try:
        return runtime.render_workflow(workflow_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown workflow: {workflow_name}') from exc


def _workflow_run(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_workflow_runtime(context)
    workflow_name = _require_string(arguments, 'workflow_name')
    raw_arguments = arguments.get('arguments', {})
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolExecutionError('arguments must be an object')
    try:
        rendered = runtime.render_run_report(workflow_name, arguments=raw_arguments)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown workflow: {workflow_name}') from exc
    return (
        rendered,
        {
            'action': 'workflow_run',
            'workflow_name': workflow_name,
            'argument_keys': sorted(str(key) for key in raw_arguments.keys()),
        },
    )


def _remote_trigger(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_trigger_runtime(context)
    action = _require_string(arguments, 'action').strip().lower()
    if action == 'list':
        query = arguments.get('query')
        if query is not None and not isinstance(query, str):
            raise ToolExecutionError('query must be a string')
        max_triggers = _coerce_int(arguments, 'max_triggers', 50)
        rendered = runtime.render_trigger_index(query=query)
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'listed_triggers': len(runtime.list_triggers(query=query, limit=max_triggers)),
            },
        )
    if action == 'get':
        trigger_id = _require_string(arguments, 'trigger_id')
        try:
            rendered = runtime.render_trigger(trigger_id)
        except KeyError as exc:
            raise ToolExecutionError(f'Unknown remote trigger: {trigger_id}') from exc
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger_id,
            },
        )
    body = arguments.get('body', {})
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ToolExecutionError('body must be an object')
    if action == 'create':
        try:
            trigger = runtime.create_trigger(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(str(exc)) from exc
        return (
            runtime.render_trigger(trigger.trigger_id),
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger.trigger_id,
            },
        )
    trigger_id = _require_string(arguments, 'trigger_id')
    if action == 'update':
        try:
            trigger = runtime.update_trigger(trigger_id, body)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(str(exc)) from exc
        return (
            runtime.render_trigger(trigger.trigger_id),
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger.trigger_id,
            },
        )
    if action == 'run':
        try:
            rendered = runtime.render_run_report(trigger_id, body=body)
        except KeyError as exc:
            raise ToolExecutionError(f'Unknown remote trigger: {trigger_id}') from exc
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger_id,
                'body_keys': sorted(str(key) for key in body.keys()),
            },
        )
    raise ToolExecutionError('action must be one of list, get, create, update, or run')


def _team_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_teams = _coerce_int(arguments, 'max_teams', 50)
    return runtime.render_teams_index(query=query, limit=max_teams)


def _team_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    try:
        return runtime.render_team(team_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc


def _team_create(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    description = arguments.get('description')
    members = arguments.get('members', ())
    metadata = arguments.get('metadata')
    if description is not None and not isinstance(description, str):
        raise ToolExecutionError('description must be a string')
    if not isinstance(members, (list, tuple)):
        raise ToolExecutionError('members must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        team = runtime.create_team(
            team_name,
            description=description,
            members=members,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Team already exists: {team_name}') from exc
    return (
        f'created team {team.name}',
        {
            'action': 'team_create',
            'team_name': team.name,
            'member_count': len(team.members),
            'path': str(runtime.state_path),
        },
    )


def _team_delete(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    try:
        team = runtime.delete_team(team_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc
    return (
        f'deleted team {team.name}',
        {
            'action': 'team_delete',
            'team_name': team.name,
            'path': str(runtime.state_path),
        },
    )


def _send_message(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    message = _require_string(arguments, 'message')
    sender = arguments.get('sender', 'agent')
    recipient = arguments.get('recipient')
    metadata = arguments.get('metadata')
    if sender is not None and not isinstance(sender, str):
        raise ToolExecutionError('sender must be a string')
    if recipient is not None and not isinstance(recipient, str):
        raise ToolExecutionError('recipient must be a string')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        stored = runtime.send_message(
            team_name=team_name,
            text=message,
            sender=sender or 'agent',
            recipient=recipient,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc
    return (
        f'sent message to team {stored.team_name}',
        {
            'action': 'send_message',
            'team_name': stored.team_name,
            'sender': stored.sender,
            'recipient': stored.recipient,
            'message_id': stored.message_id,
            'message_preview': _snapshot_text(stored.text),
            'path': str(runtime.state_path),
        },
    )


def _team_messages(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    team_name = arguments.get('team_name')
    if team_name is not None and not isinstance(team_name, str):
        raise ToolExecutionError('team_name must be a string')
    limit = _coerce_int(arguments, 'limit', 20)
    try:
        return runtime.render_messages(team_name=team_name, limit=limit)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc


def _task_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    status = arguments.get('status')
    owner = arguments.get('owner')
    actionable_only = arguments.get('actionable_only', False)
    if status is not None and not isinstance(status, str):
        raise ToolExecutionError('status must be a string')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if not isinstance(actionable_only, bool):
        raise ToolExecutionError('actionable_only must be a boolean')
    max_tasks = _coerce_int(arguments, 'max_tasks', 50)
    return runtime.render_tasks(
        status=status,
        owner=owner,
        actionable_only=actionable_only,
        limit=max_tasks,
    )


def _task_next(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    max_tasks = _coerce_int(arguments, 'max_tasks', 10)
    return runtime.render_next_tasks(limit=max_tasks)


def _task_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    return runtime.render_task(_require_string(arguments, 'task_id'))


def _plan_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    del arguments
    runtime = _require_plan_runtime(context)
    return runtime.render_plan()


def _update_plan(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_plan_runtime(context)
    items = arguments.get('items')
    if not isinstance(items, list):
        raise ToolExecutionError('items must be an array of plan step objects')
    explanation = arguments.get('explanation')
    if explanation is not None and not isinstance(explanation, str):
        raise ToolExecutionError('explanation must be a string')
    sync_tasks = arguments.get('sync_tasks', True)
    if not isinstance(sync_tasks, bool):
        raise ToolExecutionError('sync_tasks must be a boolean')
    mutation = runtime.update_plan(
        [item for item in items if isinstance(item, dict)],
        explanation=explanation,
        task_runtime=context.task_runtime,
        sync_tasks=sync_tasks,
    )
    return (
        f'updated plan with {mutation.after_count} step(s)',
        _plan_mutation_metadata(
            action='update_plan',
            mutation=mutation,
            total_steps=mutation.after_count,
            sync_tasks=sync_tasks,
        ),
    )


def _plan_clear(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_plan_runtime(context)
    sync_tasks = arguments.get('sync_tasks', True)
    if not isinstance(sync_tasks, bool):
        raise ToolExecutionError('sync_tasks must be a boolean')
    mutation = runtime.clear_plan(
        task_runtime=context.task_runtime if sync_tasks else None,
    )
    return (
        'cleared local plan',
        _plan_mutation_metadata(
            action='plan_clear',
            mutation=mutation,
            total_steps=0,
            sync_tasks=sync_tasks,
        ),
    )


def _task_create(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    title = _require_string(arguments, 'title')
    description = arguments.get('description')
    status = arguments.get('status', 'todo')
    priority = arguments.get('priority')
    task_id = arguments.get('task_id')
    active_form = arguments.get('active_form')
    owner = arguments.get('owner')
    blocks = arguments.get('blocks', [])
    blocked_by = arguments.get('blocked_by', [])
    metadata = arguments.get('metadata')
    if description is not None and not isinstance(description, str):
        raise ToolExecutionError('description must be a string')
    if status is not None and not isinstance(status, str):
        raise ToolExecutionError('status must be a string')
    if priority is not None and not isinstance(priority, str):
        raise ToolExecutionError('priority must be a string')
    if task_id is not None and not isinstance(task_id, str):
        raise ToolExecutionError('task_id must be a string')
    if active_form is not None and not isinstance(active_form, str):
        raise ToolExecutionError('active_form must be a string')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if not isinstance(blocks, list):
        raise ToolExecutionError('blocks must be an array of strings')
    if not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    mutation = runtime.create_task(
        title=title,
        description=description,
        status=status or 'pending',
        priority=priority,
        task_id=task_id,
        active_form=active_form,
        owner=owner,
        blocks=blocks,
        blocked_by=blocked_by,
        metadata=metadata,
    )
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task creation succeeded but returned no task object')
    return (
        f'created task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_create',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_update(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    title = arguments.get('title')
    description = arguments.get('description')
    status = arguments.get('status')
    priority = arguments.get('priority')
    active_form = arguments.get('active_form')
    owner = arguments.get('owner')
    blocks = arguments.get('blocks')
    blocked_by = arguments.get('blocked_by')
    metadata = arguments.get('metadata')
    for key, value in (
        ('title', title),
        ('description', description),
        ('status', status),
        ('priority', priority),
        ('active_form', active_form),
        ('owner', owner),
    ):
        if value is not None and not isinstance(value, str):
            raise ToolExecutionError(f'{key} must be a string')
    if blocks is not None and not isinstance(blocks, list):
        raise ToolExecutionError('blocks must be an array of strings')
    if blocked_by is not None and not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        mutation = runtime.update_task(
            task_id,
            title=title,
            description=description,
            status=status,
            priority=priority,
            active_form=active_form,
            owner=owner,
            blocks=blocks,
            blocked_by=blocked_by,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task update succeeded but returned no task object')
    return (
        f'updated task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_update',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_start(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    owner = arguments.get('owner')
    active_form = arguments.get('active_form')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if active_form is not None and not isinstance(active_form, str):
        raise ToolExecutionError('active_form must be a string')
    try:
        mutation = runtime.start_task(task_id, owner=owner, active_form=active_form)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task start succeeded but returned no task object')
    return (
        f'started task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_start',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_complete(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    try:
        mutation = runtime.complete_task(task_id)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = runtime.get_task(task_id)
    if task is None:
        raise ToolExecutionError(f'Task completion succeeded but task {task_id} not found')
    return (
        f'completed task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_complete',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_block(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    blocked_by = arguments.get('blocked_by')
    reason = arguments.get('reason')
    if blocked_by is not None and not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    try:
        mutation = runtime.block_task(task_id, blocked_by=blocked_by, reason=reason)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task block succeeded but returned no task object')
    return (
        f'blocked task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_block',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_cancel(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    reason = arguments.get('reason')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    try:
        mutation = runtime.cancel_task(task_id, reason=reason)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task cancellation succeeded but returned no task object')
    return (
        f'cancelled task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_cancel',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _todo_write(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    items = arguments.get('items')
    if not isinstance(items, list):
        raise ToolExecutionError('items must be an array of task objects')
    mutation = runtime.replace_tasks(
        [item for item in items if isinstance(item, dict)]
    )
    return (
        f'replaced todo list with {mutation.after_count} task(s)',
        _task_mutation_metadata(
            action='todo_write',
            mutation=mutation,
            total_tasks=mutation.after_count,
        ),
    )


def _enter_plan_mode(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    """Enter plan mode — focus on exploration and planning, not execution."""
    if getattr(context, '_plan_mode', False):
        return 'Already in plan mode.'
    context._plan_mode = True
    return (
        'Entered plan mode. Focus on exploring the codebase and creating a plan. '
        'Use read-only tools (Read, Grep, Glob) to investigate. '
        'When your plan is ready, call ExitPlanMode to return to normal mode.'
    )


def _exit_plan_mode(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    """Exit plan mode and return to normal execution."""
    if not getattr(context, '_plan_mode', False):
        return 'Not currently in plan mode.'
    context._plan_mode = False
    return 'Exited plan mode. You can now execute changes based on your plan.'


def _task_output(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    """Get output from a background task."""
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    block = arguments.get('block', True)
    timeout_ms = arguments.get('timeout', 30000)
    if not isinstance(timeout_ms, (int, float)):
        timeout_ms = 30000
    timeout_ms = max(0, min(timeout_ms, 600000))

    task = runtime.get_task(task_id)
    if task is None:
        raise ToolExecutionError(f'Task not found: {task_id}')

    if task.status in ('completed', 'cancelled', 'failed'):
        output = getattr(task, 'output', '') or getattr(task, 'result', '') or ''
        return (
            f'<retrieval_status>success</retrieval_status>\n'
            f'<task_id>{task_id}</task_id>\n'
            f'<status>{task.status}</status>\n'
            f'<output>{output}</output>'
        )

    if not block:
        return (
            f'<retrieval_status>not_ready</retrieval_status>\n'
            f'<task_id>{task_id}</task_id>\n'
            f'<status>{task.status}</status>'
        )

    # Blocking wait — poll until completion or timeout
    import time as _time
    deadline = _time.monotonic() + (timeout_ms / 1000.0)
    while _time.monotonic() < deadline:
        task = runtime.get_task(task_id)
        if task is None or task.status in ('completed', 'cancelled', 'failed'):
            break
        _time.sleep(0.1)

    if task is None:
        raise ToolExecutionError(f'Task disappeared: {task_id}')

    if task.status in ('completed', 'cancelled', 'failed'):
        output = getattr(task, 'output', '') or getattr(task, 'result', '') or ''
        return (
            f'<retrieval_status>success</retrieval_status>\n'
            f'<task_id>{task_id}</task_id>\n'
            f'<status>{task.status}</status>\n'
            f'<output>{output}</output>'
        )

    return (
        f'<retrieval_status>timeout</retrieval_status>\n'
        f'<task_id>{task_id}</task_id>\n'
        f'<status>{task.status}</status>'
    )


def _task_stop(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    """Stop a running background task."""
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')

    task = runtime.get_task(task_id)
    if task is None:
        raise ToolExecutionError(f'Task not found: {task_id}')

    if task.status not in ('pending', 'in_progress', 'running'):
        raise ToolExecutionError(
            f'Task {task_id} is not running (status: {task.status})'
        )

    # Cancel the task via the runtime
    mutation = runtime.cancel_task(task_id, reason='Stopped by TaskStop tool')
    description = getattr(task, 'title', '') or getattr(task, 'description', '') or task_id
    return (
        f'Successfully stopped task: {task_id} ({description})',
        {'action': 'task_stop', 'task_id': task_id},
    )


def _stream_bash(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> Iterator[ToolStreamUpdate]:
    try:
        command = _require_string(arguments, 'command')
        _ensure_shell_allowed(command, context)
        raw_timeout = arguments.get('timeout')
        if raw_timeout is not None:
            if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
                raise ToolExecutionError('timeout must be a positive number (seconds)')
            timeout_seconds = max(1.0, min(float(raw_timeout), 600.0))
        else:
            timeout_seconds = context.command_timeout_seconds
        bash_command, working_directory = _build_bash_invocation(
            command,
            context.root,
            extra_env=context.extra_env,
        )
        process = subprocess.Popen(
            bash_command,
            shell=False,
            cwd=working_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_build_subprocess_env(context),
        )
    except (ToolPermissionError, ToolExecutionError, OSError, subprocess.SubprocessError) as exc:
        yield ToolStreamUpdate(
            kind='result',
            result=ToolExecutionResult(name='bash', ok=False, content=str(exc)),
        )
        return

    stream_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    reader_threads: list[threading.Thread] = []
    stream_count = 0
    if process.stdout is not None:
        stream_count += 1
        reader_threads.append(
            _start_stream_reader(process.stdout, 'stdout', stream_queue)
        )
    if process.stderr is not None:
        stream_count += 1
        reader_threads.append(
            _start_stream_reader(process.stderr, 'stderr', stream_queue)
        )

    deadline = time.monotonic() + timeout_seconds
    timeout_error: str | None = None
    closed_streams = 0

    try:
        while closed_streams < stream_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_error = (
                    f'Command timed out after {timeout_seconds:.1f}s: {command}'
                )
                process.kill()
                break
            try:
                stream_name, line = stream_queue.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                closed_streams += 1
                continue
            if stream_name == 'stdout':
                stdout_chunks.append(line)
            else:
                stderr_chunks.append(line)
            yield ToolStreamUpdate(
                kind='delta',
                content=line,
                stream=stream_name,
            )
    finally:
        for thread in reader_threads:
            thread.join(timeout=0.2)
        for stream in (process.stdout, process.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    exit_code = process.wait()
    if timeout_error is not None:
        yield ToolStreamUpdate(
            kind='result',
            result=ToolExecutionResult(
                name='bash',
                ok=False,
                content=timeout_error,
                metadata={
                    'action': 'bash',
                    'command': command,
                    'exit_code': exit_code,
                    'timed_out': True,
                    'stdout_preview': _snapshot_text(''.join(stdout_chunks)),
                    'stderr_preview': _snapshot_text(''.join(stderr_chunks)),
                },
            ),
        )
        return

    stdout = ''.join(stdout_chunks)
    stderr = ''.join(stderr_chunks)
    output = _format_bash_output(stdout, stderr, exit_code)
    yield ToolStreamUpdate(
        kind='result',
        result=ToolExecutionResult(
            name='bash',
            ok=True,
            content=_truncate_output(output, context.max_output_chars),
            metadata={
                'action': 'bash',
                'command': command,
                'exit_code': exit_code,
                'streamed': True,
                'stdout_preview': _snapshot_text(stdout),
                'stderr_preview': _snapshot_text(stderr),
                'output_preview': _snapshot_text(output),
            },
        ),
    )


def _agent_tool_placeholder(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    raise ToolExecutionError(
        'Agent/delegate_agent must be handled by the runtime and is not available as a standalone tool handler'
    )


def _execute_skill(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str | tuple[str, dict[str, Any]]:
    """Execute a skill (slash command) from the Skill tool.

    The skill tool must be handled specially by the runtime because it
    needs access to the agent to invoke slash commands.  This handler
    is a placeholder that raises — the runtime intercepts `Skill` tool
    calls before they reach this handler.
    """
    raise ToolExecutionError(
        'Skill must be handled by the runtime and is not available as a standalone tool handler'
    )


def _lsp_query(arguments: dict[str, Any], context: ToolExecutionContext):
    runtime = _require_lsp_runtime(context)
    operation = _require_string(arguments, 'operation')
    file_path = _require_string(arguments, 'file_path')
    line = _coerce_int(arguments, 'line', 1)
    character = _coerce_int(arguments, 'character', 1)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_results = _coerce_int(arguments, 'max_results', 50)
    try:
        result = runtime.query(
            operation,
            file_path=file_path,
            line=line,
            character=character,
            query=query,
            max_results=max_results,
        )
    except KeyError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        result.content,
        {
            'action': 'lsp_query',
            'operation': result.operation,
            'file_path': file_path,
            'line': line,
            'character': character,
            'result_count': result.result_count,
            'file_count': result.file_count,
            'symbol_name': result.symbol_name,
        },
    )


def _require_account_runtime(context: ToolExecutionContext):
    if context.account_runtime is None:
        raise ToolExecutionError('No local account runtime is available.')
    return context.account_runtime


def _require_ask_user_runtime(context: ToolExecutionContext):
    if context.ask_user_runtime is None:
        raise ToolExecutionError('No local ask-user runtime is available.')
    return context.ask_user_runtime


def _require_search_runtime(context: ToolExecutionContext):
    if context.search_runtime is None:
        raise ToolExecutionError(
            'No local search provider is available. Add a .claw-search.json or .claude/search.json manifest, '
            'or set SEARXNG_BASE_URL (no API key required), BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY.'
        )
    if not context.search_runtime.web_search_enabled:
        raise ToolExecutionError('Web search is disabled. Use `/search on` to enable it.')
    if not context.search_runtime.providers:
        raise ToolExecutionError(
            'No local search provider is available. Add a .claw-search.json or .claude/search.json manifest, '
            'or set SEARXNG_BASE_URL (no API key required), BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY.'
        )
    return context.search_runtime


def _require_config_runtime(context: ToolExecutionContext):
    if context.config_runtime is None:
        raise ToolExecutionError('No local config runtime is available.')
    return context.config_runtime


def _require_lsp_runtime(context: ToolExecutionContext):
    if context.lsp_runtime is None or not context.lsp_runtime.has_lsp_support():
        raise ToolExecutionError(
            'No local LSP runtime is available. Add supported source files to the workspace or a .claw-lsp.json manifest.'
        )
    return context.lsp_runtime


def _require_mcp_runtime(context: ToolExecutionContext):
    if (
        context.mcp_runtime is None
        or (
            not context.mcp_runtime.resources
            and not context.mcp_runtime.servers
        )
    ):
        raise ToolExecutionError(
            'No MCP runtime is available. Add a .claw-mcp.json, .mcp.json, or mcpServers manifest first.'
        )
    return context.mcp_runtime


def _require_remote_runtime(context: ToolExecutionContext):
    if context.remote_runtime is None:
        raise ToolExecutionError('Local remote runtime is not available.')
    return context.remote_runtime


def _require_remote_trigger_runtime(context: ToolExecutionContext):
    if context.remote_trigger_runtime is None:
        raise ToolExecutionError('Local remote trigger runtime is not available.')
    return context.remote_trigger_runtime


def _require_plan_runtime(context: ToolExecutionContext):
    if context.plan_runtime is None:
        raise ToolExecutionError('Local plan runtime is not available.')
    return context.plan_runtime


def _require_task_runtime(context: ToolExecutionContext):
    if context.task_runtime is None:
        raise ToolExecutionError('Local task runtime is not available.')
    return context.task_runtime


def _require_team_runtime(context: ToolExecutionContext):
    if context.team_runtime is None:
        raise ToolExecutionError('Local team runtime is not available.')
    return context.team_runtime


def _require_workflow_runtime(context: ToolExecutionContext):
    if context.workflow_runtime is None or not context.workflow_runtime.has_workflows():
        raise ToolExecutionError('Local workflow runtime is not available.')
    return context.workflow_runtime


def _require_worktree_runtime(context: ToolExecutionContext):
    if context.worktree_runtime is None:
        raise ToolExecutionError('Local worktree runtime is not available.')
    return context.worktree_runtime


def _task_mutation_metadata(
    *,
    action: str,
    mutation,
    task_id: str | None = None,
    task_status: str | None = None,
    total_tasks: int,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError:
        relative_path = str(mutation.store_path)
    payload: dict[str, Any] = {
        'action': action,
        'path': relative_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'before_task_count': mutation.before_count,
        'after_task_count': mutation.after_count,
        'total_tasks': total_tasks,
    }
    if task_id is not None:
        payload['task_id'] = task_id
    if task_status is not None:
        payload['task_status'] = task_status
    return payload


def _config_mutation_metadata(
    mutation,
    value: Any,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError:
        relative_path = str(mutation.store_path)
    return {
        'action': 'config_set',
        'path': relative_path,
        'source_name': mutation.source_name,
        'key_path': mutation.key_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'effective_key_count': mutation.effective_key_count,
        'value_preview': _snapshot_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True)
            if not isinstance(value, str)
            else value
        ),
    }


def _plan_mutation_metadata(
    *,
    action: str,
    mutation,
    total_steps: int,
    sync_tasks: bool,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError:
        relative_path = str(mutation.store_path)
    payload: dict[str, Any] = {
        'action': action,
        'path': relative_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'before_plan_count': mutation.before_count,
        'after_plan_count': mutation.after_count,
        'total_steps': total_steps,
        'sync_tasks': sync_tasks,
    }
    if mutation.explanation is not None:
        payload['explanation'] = mutation.explanation
    if mutation.synced_task_store_path is not None:
        try:
            payload['synced_task_store_path'] = str(
                Path(mutation.synced_task_store_path).relative_to(Path.cwd())
            )
        except ValueError:
            payload['synced_task_store_path'] = mutation.synced_task_store_path
    if mutation.synced_task_sha256 is not None:
        payload['synced_task_sha256'] = mutation.synced_task_sha256
    if mutation.synced_tasks:
        payload['synced_tasks'] = mutation.synced_tasks
    return payload


_SENSITIVE_ENV_KEYWORDS = (
    'SECRET',
    'TOKEN',
    'PASSWORD',
    'PRIVATE_KEY',
    'API_KEY',
    'CREDENTIAL',
    'AUTH',
)


def _is_sensitive_env_var(name: str) -> bool:
    """Return True if the environment variable name likely contains a secret."""
    upper = name.upper()
    return any(keyword in upper for keyword in _SENSITIVE_ENV_KEYWORDS)


def _build_subprocess_env(context: ToolExecutionContext) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not _is_sensitive_env_var(key)
    }
    env.update(context.extra_env)
    return env


def _build_bash_invocation(
    command: str,
    cwd: Path,
    *,
    extra_env: dict[str, str],
) -> tuple[list[str], Path | None]:
    executable = _resolve_bash_executable()
    script_lines: list[str] = []
    if _is_wsl_bash(executable):
        wsl_cwd = _to_wsl_path(cwd)
        script_lines.append(f'cd {shlex.quote(wsl_cwd)}')
        working_directory = None
        command = _materialize_bash_env_references(command, extra_env)
    else:
        working_directory = cwd
    for key, value in sorted(extra_env.items()):
        script_lines.append(f'export {key}={shlex.quote(value)}')
    script_lines.append(command)
    return [executable, '-lc', '; '.join(script_lines)], working_directory


def _resolve_bash_executable() -> str:
    resolved = shutil.which('bash')
    if isinstance(resolved, str) and resolved.strip():
        return resolved
    if os.path.exists('/bin/bash'):
        return '/bin/bash'
    raise ToolExecutionError('bash executable was not found on this system')


def _is_wsl_bash(executable: str) -> bool:
    normalized = executable.replace('/', '\\').lower()
    return normalized.endswith('\\bash.exe') and (
        '\\windows\\system32\\' in normalized or '\\windowsapps\\' in normalized
    )


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(':').lower()
    if drive:
        tail = resolved.as_posix().split(':', 1)[1].lstrip('/')
        return f'/mnt/{drive}/{tail}'
    return resolved.as_posix()


def _materialize_bash_env_references(
    command: str,
    extra_env: dict[str, str],
) -> str:
    rendered = command
    for key, value in sorted(extra_env.items(), key=lambda item: len(item[0]), reverse=True):
        rendered = re.sub(rf'\$\{{{re.escape(key)}\}}', value, rendered)
        rendered = re.sub(rf'\${re.escape(key)}\b', value, rendered)
    return rendered


def _start_stream_reader(
    stream,
    stream_name: str,
    stream_queue: 'queue.Queue[tuple[str, str | None]]',
) -> threading.Thread:
    def _pump() -> None:
        try:
            while True:
                line = stream.readline()
                if line == '':
                    break
                stream_queue.put((stream_name, line))
        finally:
            stream_queue.put((stream_name, None))

    thread = threading.Thread(
        target=_pump,
        daemon=True,
        name=f'bash-{stream_name}-reader',
    )
    thread.start()
    return thread


def _stream_static_text_result(
    result: ToolExecutionResult,
    *,
    chunk_size: int = 400,
) -> Iterator[ToolStreamUpdate]:
    content = result.content
    if content:
        for start in range(0, len(content), chunk_size):
            yield ToolStreamUpdate(
                kind='delta',
                content=content[start:start + chunk_size],
                stream='tool',
            )
    metadata = dict(result.metadata)
    metadata.setdefault('streamed', True)
    yield ToolStreamUpdate(
        kind='result',
        result=ToolExecutionResult(
            name=result.name,
            ok=result.ok,
            content=result.content,
            metadata=metadata,
        ),
    )
