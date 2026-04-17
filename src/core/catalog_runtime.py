from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.core.permissions import ToolPermissionContext


@dataclass(frozen=True)
class Subsystem:
    name: str
    path: str
    file_count: int
    notes: str


@dataclass(frozen=True)
class PortingModule:
    name: str
    responsibility: str
    source_hint: str
    status: str = 'planned'


@dataclass(frozen=True)
class PermissionDenial:
    tool_name: str
    reason: str


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0

    def add_turn(self, prompt: str, output: str) -> 'UsageSummary':
        return UsageSummary(
            input_tokens=self.input_tokens + len(prompt.split()),
            output_tokens=self.output_tokens + len(output.split()),
        )


@dataclass
class PortingBacklog:
    title: str
    modules: list[PortingModule] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f'- {module.name} [{module.status}] — {module.responsibility} (from {module.source_hint})'
            for module in self.modules
        ]


COMMAND_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / 'reference_data' / 'commands_snapshot.json'
TOOL_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / 'reference_data' / 'tools_snapshot.json'


@dataclass(frozen=True)
class CommandExecution:
    name: str
    source_hint: str
    prompt: str
    handled: bool
    message: str


@lru_cache(maxsize=1)
def load_command_snapshot() -> tuple[PortingModule, ...]:
    raw_entries = json.loads(COMMAND_SNAPSHOT_PATH.read_text())
    return tuple(
        PortingModule(
            name=entry['name'],
            responsibility=entry['responsibility'],
            source_hint=entry['source_hint'],
            status='mirrored',
        )
        for entry in raw_entries
    )


PORTED_COMMANDS = load_command_snapshot()


@lru_cache(maxsize=1)
def built_in_command_names() -> frozenset[str]:
    return frozenset(module.name for module in PORTED_COMMANDS)


def build_command_backlog() -> PortingBacklog:
    return PortingBacklog(title='Command surface', modules=list(PORTED_COMMANDS))


def command_names() -> list[str]:
    return [module.name for module in PORTED_COMMANDS]


def get_command(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_COMMANDS:
        if module.name.lower() == needle:
            return module
    return None


def get_commands(
    cwd: str | None = None,
    include_plugin_commands: bool = True,
    include_skill_commands: bool = True,
) -> tuple[PortingModule, ...]:
    commands = list(PORTED_COMMANDS)
    if not include_plugin_commands:
        commands = [module for module in commands if 'plugin' not in module.source_hint.lower()]
    if not include_skill_commands:
        commands = [module for module in commands if 'skills' not in module.source_hint.lower()]
    return tuple(commands)


def find_commands(query: str, limit: int = 20) -> list[PortingModule]:
    needle = query.lower()
    matches = [module for module in PORTED_COMMANDS if needle in module.name.lower() or needle in module.source_hint.lower()]
    return matches[:limit]


def execute_command(name: str, prompt: str = '') -> CommandExecution:
    module = get_command(name)
    if module is None:
        return CommandExecution(name=name, source_hint='', prompt=prompt, handled=False, message=f'Unknown mirrored command: {name}')
    action = f"Mirrored command '{module.name}' from {module.source_hint} would handle prompt {prompt!r}."
    return CommandExecution(name=module.name, source_hint=module.source_hint, prompt=prompt, handled=True, message=action)


def render_command_index(limit: int = 20, query: str | None = None) -> str:
    modules = find_commands(query, limit) if query else list(PORTED_COMMANDS[:limit])
    lines = [f'Command entries: {len(PORTED_COMMANDS)}', '']
    if query:
        lines.append(f'Filtered by: {query}')
        lines.append('')
    lines.extend(f'- {module.name} — {module.source_hint}' for module in modules)
    return '\n'.join(lines)


@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


@lru_cache(maxsize=1)
def load_tool_snapshot() -> tuple[PortingModule, ...]:
    raw_entries = json.loads(TOOL_SNAPSHOT_PATH.read_text())
    return tuple(
        PortingModule(
            name=entry['name'],
            responsibility=entry['responsibility'],
            source_hint=entry['source_hint'],
            status='mirrored',
        )
        for entry in raw_entries
    )


PORTED_TOOLS = load_tool_snapshot()


def build_tool_backlog() -> PortingBacklog:
    return PortingBacklog(title='Tool surface', modules=list(PORTED_TOOLS))


def tool_names() -> list[str]:
    return [module.name for module in PORTED_TOOLS]


def get_tool(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_TOOLS:
        if module.name.lower() == needle:
            return module
    return None


def filter_tools_by_permission_context(
    tools: tuple[PortingModule, ...],
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    if permission_context is None:
        return tools
    return tuple(module for module in tools if not permission_context.blocks(module.name))


def get_tools(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    tools = list(PORTED_TOOLS)
    if simple_mode:
        tools = [module for module in tools if module.name in {'BashTool', 'FileReadTool', 'FileEditTool'}]
    if not include_mcp:
        tools = [module for module in tools if 'mcp' not in module.name.lower() and 'mcp' not in module.source_hint.lower()]
    return filter_tools_by_permission_context(tuple(tools), permission_context)


def find_tools(query: str, limit: int = 20) -> list[PortingModule]:
    needle = query.lower()
    matches = [module for module in PORTED_TOOLS if needle in module.name.lower() or needle in module.source_hint.lower()]
    return matches[:limit]


def execute_tool(name: str, payload: str = '') -> ToolExecution:
    module = get_tool(name)
    if module is None:
        return ToolExecution(name=name, source_hint='', payload=payload, handled=False, message=f'Unknown mirrored tool: {name}')
    action = f"Mirrored tool '{module.name}' from {module.source_hint} would handle payload {payload!r}."
    return ToolExecution(name=module.name, source_hint=module.source_hint, payload=payload, handled=True, message=action)


def render_tool_index(limit: int = 20, query: str | None = None) -> str:
    modules = find_tools(query, limit) if query else list(PORTED_TOOLS[:limit])
    lines = [f'Tool entries: {len(PORTED_TOOLS)}', '']
    if query:
        lines.append(f'Filtered by: {query}')
        lines.append('')
    lines.extend(f'- {module.name} — {module.source_hint}' for module in modules)
    return '\n'.join(lines)


DEFAULT_SRC_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PortManifest:
    src_root: Path
    total_python_files: int
    top_level_modules: tuple[Subsystem, ...]

    def to_markdown(self) -> str:
        lines = [
            f'Port root: `{self.src_root}`',
            f'Total Python files: **{self.total_python_files}**',
            '',
            'Top-level Python modules:',
        ]
        for module in self.top_level_modules:
            lines.append(f'- `{module.name}` ({module.file_count} files) — {module.notes}')
        return '\n'.join(lines)


def build_port_manifest(src_root: Path | None = None) -> PortManifest:
    root = src_root or DEFAULT_SRC_ROOT
    files = [path for path in root.rglob('*.py') if path.is_file()]
    counter = Counter(
        path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else path.name
        for path in files
        if path.name != '__pycache__'
    )
    notes = {
        '__init__.py': 'package export surface',
        'main.py': 'CLI entrypoint',
        'catalog_runtime.py': 'catalog dataclasses, snapshots, and indexes',
        'bootstrap_runtime.py': 'startup, context, and bootstrap orchestration',
        'query_engine.py': 'port orchestration summary layer',
        'task.py': 'task-level planning structures',
    }
    modules = tuple(
        Subsystem(name=name, path=f'src/{name}', file_count=count, notes=notes.get(name, 'Python port support module'))
        for name, count in counter.most_common()
    )
    return PortManifest(src_root=root, total_python_files=len(files), top_level_modules=modules)


@dataclass(frozen=True)
class CommandGraph:
    builtins: tuple[PortingModule, ...]
    plugin_like: tuple[PortingModule, ...]
    skill_like: tuple[PortingModule, ...]

    def flattened(self) -> tuple[PortingModule, ...]:
        return self.builtins + self.plugin_like + self.skill_like

    def as_markdown(self) -> str:
        lines = [
            '# Command Graph',
            '',
            f'Builtins: {len(self.builtins)}',
            f'Plugin-like commands: {len(self.plugin_like)}',
            f'Skill-like commands: {len(self.skill_like)}',
        ]
        return '\n'.join(lines)


def build_command_graph() -> CommandGraph:
    commands = get_commands()
    builtins = tuple(module for module in commands if 'plugin' not in module.source_hint.lower() and 'skills' not in module.source_hint.lower())
    plugin_like = tuple(module for module in commands if 'plugin' in module.source_hint.lower())
    skill_like = tuple(module for module in commands if 'skills' in module.source_hint.lower())
    return CommandGraph(builtins=builtins, plugin_like=plugin_like, skill_like=skill_like)


@dataclass(frozen=True)
class ToolPool:
    tools: tuple[PortingModule, ...]
    simple_mode: bool
    include_mcp: bool

    def as_markdown(self) -> str:
        lines = [
            '# Tool Pool',
            '',
            f'Simple mode: {self.simple_mode}',
            f'Include MCP: {self.include_mcp}',
            f'Tool count: {len(self.tools)}',
        ]
        lines.extend(f'- {tool.name} — {tool.source_hint}' for tool in self.tools[:15])
        return '\n'.join(lines)


def assemble_tool_pool(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> ToolPool:
    return ToolPool(
        tools=get_tools(
            simple_mode=simple_mode,
            include_mcp=include_mcp,
            permission_context=permission_context,
        ),
        simple_mode=simple_mode,
        include_mcp=include_mcp,
    )
