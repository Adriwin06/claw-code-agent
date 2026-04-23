from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


_DOTENV_NAME = '.env'
_DOTENV_KEY_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_BRACED_ENV_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?)-([^}]*))?\}')
_DOLLAR_ENV_RE = re.compile(r'(?<!\$)\$([A-Za-z_][A-Za-z0-9_]*)')
_PERCENT_ENV_RE = re.compile(r'%([A-Za-z_][A-Za-z0-9_]*)%')
_UNRESOLVED_ENV_RE = re.compile(
    r'\$\{[^}]+\}|(?<!\$)\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%'
)
_LOADED_ENV_FILES: set[Path] = set()


def load_workspace_env(
    cwd: Path,
    additional_working_directories: tuple[str | Path, ...] = (),
    *,
    override: bool = False,
) -> tuple[Path, ...]:
    loaded_files: list[Path] = []
    for path in discover_workspace_env_files(
        cwd,
        additional_working_directories=additional_working_directories,
    ):
        if path in _LOADED_ENV_FILES:
            loaded_files.append(path)
            continue
        values = parse_dotenv_file(path)
        for key, value in values.items():
            if override or key not in os.environ:
                os.environ[key] = value
        _LOADED_ENV_FILES.add(path)
        loaded_files.append(path)
    return tuple(loaded_files)


def discover_workspace_env_files(
    cwd: Path,
    additional_working_directories: tuple[str | Path, ...] = (),
) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for raw_path in (cwd, *additional_working_directories):
        root = Path(raw_path).resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        roots.append(root)

    discovered: list[Path] = []
    seen_files: set[Path] = set()
    for root in roots:
        current = root
        while True:
            candidate = (current / _DOTENV_NAME).resolve()
            if candidate not in seen_files and candidate.exists() and candidate.is_file():
                seen_files.add(candidate)
                discovered.append(candidate)
            if current.parent == current:
                break
            current = current.parent
    return tuple(discovered)


def parse_dotenv_file(path: Path) -> dict[str, str]:
    try:
        raw_text = path.read_text(encoding='utf-8')
    except OSError:
        return {}

    loaded: dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition('=')
        if separator != '=':
            continue
        key = name.strip()
        if not _DOTENV_KEY_RE.match(key):
            continue
        value = _parse_dotenv_value(raw_value)
        env = dict(os.environ)
        env.update(loaded)
        loaded[key] = expand_env_vars(value, env=env)
    return loaded


def expand_env_vars(value: str, *, env: Mapping[str, str] | None = None) -> str:
    environment = os.environ if env is None else env
    expanded = value
    for _ in range(5):
        previous = expanded
        expanded = _BRACED_ENV_RE.sub(
            lambda match: _expand_braced_var(match, environment),
            expanded,
        )
        expanded = _DOLLAR_ENV_RE.sub(
            lambda match: environment.get(match.group(1), match.group(0)),
            expanded,
        )
        expanded = _PERCENT_ENV_RE.sub(
            lambda match: environment.get(match.group(1), match.group(0)),
            expanded,
        )
        if expanded == previous:
            break
    return expanded


def has_unresolved_env_var(value: str) -> bool:
    return bool(_UNRESOLVED_ENV_RE.search(value))


def _parse_dotenv_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ''
    if value[0] == "'":
        return _parse_single_quoted_value(value)
    if value[0] == '"':
        return _parse_double_quoted_value(value)
    return _strip_unquoted_comment(value)


def _parse_single_quoted_value(value: str) -> str:
    closing_index = value.find("'", 1)
    if closing_index == -1:
        return value[1:]
    return value[1:closing_index]


def _parse_double_quoted_value(value: str) -> str:
    characters: list[str] = []
    escaped = False
    for character in value[1:]:
        if escaped:
            if character == 'n':
                characters.append('\n')
            elif character == 't':
                characters.append('\t')
            else:
                characters.append(character)
            escaped = False
            continue
        if character == '\\':
            escaped = True
            continue
        if character == '"':
            break
        characters.append(character)
    if escaped:
        characters.append('\\')
    return ''.join(characters)


def _strip_unquoted_comment(value: str) -> str:
    for index, character in enumerate(value):
        if character != '#':
            continue
        if index == 0 or value[index - 1].isspace():
            return value[:index].rstrip()
    return value.strip()


def _expand_braced_var(match: re.Match[str], env: Mapping[str, str]) -> str:
    name = match.group(1)
    default_operator = match.group(2)
    default_value = match.group(3)
    resolved = env.get(name)
    if default_value is not None:
        use_default = resolved is None or (default_operator == ':' and resolved == '')
        if use_default:
            return default_value
    if resolved is None:
        return match.group(0)
    return resolved
