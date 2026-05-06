from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


_ALWAYS_IGNORED_DIR_NAMES = {
    '.git',
    '.hg',
}
_NOISY_IGNORED_DIR_NAMES = {
    '.mypy_cache',
    '.port_sessions',
    '.pytest_cache',
    '.ruff_cache',
    '.tox',
    '.venv',
    '.venv-local',
    '.venv-linux',
    '__pycache__',
    'node_modules',
    'venv',
    'venv-local',
    'venv-linux',
}
_REFERENCE_TOKEN_PATTERN = re.compile(
    r'(^|[\s(\[{,])@(?:"([^"]+)"|\'([^\']+)\'|([^\s]+))'
)


@dataclass(frozen=True)
class WorkspaceReferenceQuery:
    query: str
    start_index: int
    end_index: int
    quoted: bool = False


@dataclass(frozen=True)
class WorkspacePathSuggestion:
    path: str
    kind: str
    size_bytes: int | None = None

    @property
    def label(self) -> str:
        suffix = '/' if self.kind == 'directory' else ''
        return f'@{self.path}{suffix}'

    @property
    def insertion_text(self) -> str:
        if any(character.isspace() for character in self.path):
            return f'@"{self.path}" '
        return f'@{self.path} '


@dataclass(frozen=True)
class WorkspacePathReference:
    path: str
    kind: str
    size_bytes: int | None = None


def extract_workspace_reference_query(
    text: str,
    cursor_position: int | None = None,
) -> WorkspaceReferenceQuery | None:
    cursor = len(text) if cursor_position is None else max(0, min(cursor_position, len(text)))
    prefix = text[:cursor]
    at_index = prefix.rfind('@')
    if at_index < 0:
        return None
    if at_index > 0 and not prefix[at_index - 1].isspace() and prefix[at_index - 1] not in '([{,':
        return None

    token = prefix[at_index + 1 :]
    if not token:
        return WorkspaceReferenceQuery('', at_index, cursor)
    if token[0] in {'"', "'"}:
        quote = token[0]
        body = token[1:]
        if quote in body:
            return None
        return WorkspaceReferenceQuery(body, at_index, cursor, quoted=True)
    if any(character.isspace() for character in token) or '"' in token or "'" in token:
        return None
    return WorkspaceReferenceQuery(token, at_index, cursor)


def build_workspace_path_suggestions(
    workspace: Path,
    *,
    hide_gitignored: bool = True,
    max_entries: int = 1500,
) -> tuple[WorkspacePathSuggestion, ...]:
    root = Path(workspace).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return ()

    suggestions: list[WorkspacePathSuggestion] = []
    raw_limit = max_entries * 4 if hide_gitignored else max_entries
    for directory, dir_names, file_names in _walk_workspace(
        root,
        hide_gitignored=hide_gitignored,
    ):
        relative_directory = _relative_workspace_path(directory, root)
        if relative_directory:
            suggestions.append(
                WorkspacePathSuggestion(
                    path=relative_directory,
                    kind='directory',
                )
            )
            if len(suggestions) >= raw_limit:
                break
        for file_name in sorted(file_names, key=str.lower):
            file_path = directory / file_name
            relative_file = _relative_workspace_path(file_path, root)
            if not relative_file:
                continue
            try:
                size = file_path.stat().st_size
            except OSError:
                size = None
            suggestions.append(
                WorkspacePathSuggestion(
                    path=relative_file,
                    kind='file',
                    size_bytes=size,
                )
            )
            if len(suggestions) >= raw_limit:
                break
        if len(suggestions) >= raw_limit:
            break

    if hide_gitignored:
        ignored_paths = _gitignored_workspace_paths(
            root,
            tuple(suggestion.path for suggestion in suggestions),
        )
        if ignored_paths:
            suggestions = [
                suggestion
                for suggestion in suggestions
                if suggestion.path not in ignored_paths
            ]

    return tuple(
        sorted(
            suggestions,
            key=lambda item: (0 if item.kind == 'directory' else 1, item.path.lower()),
        )[:max_entries]
    )


def filter_workspace_path_suggestions(
    query: str,
    suggestions: Iterable[WorkspacePathSuggestion],
    *,
    limit: int = 40,
) -> list[WorkspacePathSuggestion]:
    normalized_query = _normalize_reference_path(query).lower()
    if not normalized_query:
        return list(suggestions)[:limit]

    scored: list[tuple[int, str, WorkspacePathSuggestion]] = []
    for suggestion in suggestions:
        path = suggestion.path.lower()
        name = Path(suggestion.path).name.lower()
        if path == normalized_query:
            score = 0
        elif path.startswith(normalized_query):
            score = 1
        elif name.startswith(normalized_query):
            score = 2
        elif normalized_query in path:
            score = 3
        else:
            continue
        kind_score = '0' if suggestion.kind == 'directory' else '1'
        scored.append((score, f'{kind_score}:{path}', suggestion))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [suggestion for _, _, suggestion in scored[:limit]]


def extract_workspace_path_references(
    text: str,
    workspace: Path,
    *,
    max_references: int = 50,
) -> tuple[WorkspacePathReference, ...]:
    root = Path(workspace).expanduser().resolve()
    references: list[WorkspacePathReference] = []
    seen: set[str] = set()
    for match in _REFERENCE_TOKEN_PATTERN.finditer(text):
        raw_path = next(
            group for group in match.groups()[1:] if isinstance(group, str)
        )
        normalized = _normalize_reference_path(raw_path)
        if not normalized or normalized in seen:
            continue
        resolved = _resolve_workspace_reference(root, normalized)
        if resolved is None:
            continue
        seen.add(normalized)
        kind = 'directory' if resolved.is_dir() else 'file'
        size: int | None = None
        if resolved.is_file():
            try:
                size = resolved.stat().st_size
            except OSError:
                size = None
        references.append(
            WorkspacePathReference(
                path=normalized,
                kind=kind,
                size_bytes=size,
            )
        )
        if len(references) >= max_references:
            break
    return tuple(references)


def render_workspace_path_suggestion_detail(suggestion: WorkspacePathSuggestion) -> str:
    lines = [
        suggestion.label,
        '',
        'folder' if suggestion.kind == 'directory' else 'file',
        '',
        f'path={suggestion.path}',
    ]
    if suggestion.size_bytes is not None:
        lines.append(f'size={format_file_size(suggestion.size_bytes)}')
    lines.extend(
        [
            '',
            'Controls',
            '',
            'Up/Down: move selection',
            'Tab/Enter: insert path reference',
            'Mouse: click a path to insert it',
        ]
    )
    return '\n'.join(lines)


def format_file_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return 'unknown'
    value = float(max(size_bytes, 0))
    for unit in ('B', 'KB', 'MB', 'GB'):
        if value < 1024.0 or unit == 'GB':
            if unit == 'B':
                return f'{int(value)} {unit}'
            return f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} GB'


def _walk_workspace(root: Path, *, hide_gitignored: bool):
    gitignore = _GitIgnoreMatcher(root) if hide_gitignored else None
    for raw_directory, dir_names, file_names in os.walk(root):
        directory = Path(raw_directory)
        ignored_children = (
            gitignore.ignored_paths(
                tuple(
                    _relative_workspace_path(directory / name, root)
                    for name in (*dir_names, *file_names)
                )
            )
            if gitignore is not None
            else set()
        )
        hidden_dir_names = set(_ALWAYS_IGNORED_DIR_NAMES)
        if hide_gitignored:
            hidden_dir_names.update(_NOISY_IGNORED_DIR_NAMES)
        dir_names[:] = [
            name
            for name in sorted(dir_names, key=str.lower)
            if name not in hidden_dir_names
            and _relative_workspace_path(directory / name, root) not in ignored_children
        ]
        visible_file_names = [
            name
            for name in file_names
            if _relative_workspace_path(directory / name, root) not in ignored_children
        ]
        yield directory, dir_names, visible_file_names


def _relative_workspace_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return ''
    if not relative.parts:
        return ''
    return relative.as_posix()


def _normalize_reference_path(path: str) -> str:
    normalized = path.strip().strip('/\\').replace('\\', '/')
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return '' if normalized == '.' else normalized


def _resolve_workspace_reference(root: Path, relative_path: str) -> Path | None:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.exists():
        return None
    return candidate


def _gitignored_workspace_paths(root: Path, relative_paths: tuple[str, ...]) -> set[str]:
    if not relative_paths:
        return set()
    payload = ''.join(f'{path}\0' for path in relative_paths).encode('utf-8')
    try:
        result = subprocess.run(
            ['git', '-C', str(root), 'check-ignore', '--stdin', '-z'],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if result.returncode not in {0, 1}:
        return set()
    ignored: set[str] = set()
    for item in result.stdout.split(b'\0'):
        if not item:
            continue
        ignored.add(item.decode('utf-8', errors='replace').replace('\\', '/'))
    return ignored


class _GitIgnoreMatcher:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._enabled = True
        self._cache: dict[str, bool] = {}

    def ignored_paths(self, relative_paths: tuple[str, ...]) -> set[str]:
        if not self._enabled:
            return set()
        candidates = tuple(
            path
            for path in relative_paths
            if path and path not in self._cache
        )
        if candidates:
            ignored = _gitignored_workspace_paths(self._root, candidates)
            if not ignored and len(candidates) > 0:
                # Preserve the fallback behavior if this is not a git workspace
                # or git is unavailable.
                pass
            for path in candidates:
                self._cache[path] = path in ignored
        return {path for path in relative_paths if self._cache.get(path, False)}
