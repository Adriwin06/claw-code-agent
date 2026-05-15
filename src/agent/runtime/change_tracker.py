from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import os
from pathlib import Path
from typing import Iterable

from src.agent.models.types import ToolCall


_EXCLUDED_DIR_NAMES = {
    '.git',
    '.hg',
    '.svn',
    '.port_sessions',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.tox',
    '.venv',
    'venv',
    '__pycache__',
    'node_modules',
    'dist',
    'build',
}


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    sha256: str
    size: int
    text: str | None
    binary: bool
    truncated: bool
    line_count: int | None


@dataclass(frozen=True)
class WorkspaceChange:
    path: str
    status: str
    before: FileSnapshot | None
    after: FileSnapshot | None
    added_lines: int
    removed_lines: int
    diff: str
    diff_truncated: bool

    def to_event_file(self) -> dict[str, object]:
        payload: dict[str, object] = {
            'path': self.path,
            'status': self.status,
            'added_lines': self.added_lines,
            'removed_lines': self.removed_lines,
            'binary': bool(
                (self.before is not None and self.before.binary)
                or (self.after is not None and self.after.binary)
            ),
            'content_truncated': bool(
                (self.before is not None and self.before.truncated)
                or (self.after is not None and self.after.truncated)
            ),
            'diff': self.diff,
            'diff_truncated': self.diff_truncated,
        }
        if self.before is not None:
            payload.update(
                {
                    'before_sha256': self.before.sha256,
                    'before_size': self.before.size,
                }
            )
            if self.before.line_count is not None:
                payload['before_line_count'] = self.before.line_count
        if self.after is not None:
            payload.update(
                {
                    'after_sha256': self.after.sha256,
                    'after_size': self.after.size,
                }
            )
            if self.after.line_count is not None:
                payload['after_line_count'] = self.after.line_count
        return payload


class WorkspaceChangeTracker:
    """Track file changes made during one agent run without touching user Git state."""

    def __init__(
        self,
        root: Path,
        *,
        max_text_bytes: int = 512_000,
        max_diff_lines: int = 240,
        max_diff_chars: int = 18_000,
        max_event_files: int = 250,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.max_text_bytes = max_text_bytes
        self.max_diff_lines = max_diff_lines
        self.max_diff_chars = max_diff_chars
        self.max_event_files = max_event_files
        self._initial_baseline = self._snapshot_workspace()
        self._baseline = self._initial_baseline
        self._sequence = 0
        self._change_history: list[WorkspaceChange] = []

    def collect_tool_changes(
        self,
        *,
        tool_call: ToolCall,
        turn_index: int,
    ) -> dict[str, object] | None:
        current = self._snapshot_workspace()
        changes = self._diff_snapshots(self._baseline, current)
        self._baseline = current
        if not changes:
            return None

        self._change_history.extend(changes)
        self._sequence += 1
        event = self._changes_to_event(changes, event_type='workspace_change')
        event.update(
            {
                'sequence': self._sequence,
                'turn_index': turn_index,
                'tool_name': tool_call.name,
                'tool_call_id': tool_call.id,
            }
        )
        return event

    def collect_run_recap(
        self,
        *,
        event_type: str = 'workspace_change_recap',
    ) -> dict[str, object] | None:
        if not self._change_history:
            return None
        changes = self._aggregate_change_history()
        if not changes:
            return None
        return self._changes_to_event(
            changes,
            event_type=event_type,
        )

    def _aggregate_change_history(self) -> list[WorkspaceChange]:
        changes_by_path: dict[str, list[WorkspaceChange]] = {}
        for change in self._change_history:
            changes_by_path.setdefault(change.path, []).append(change)
        return [
            self._aggregate_path_changes(path, changes_by_path[path])
            for path in sorted(changes_by_path)
        ]

    def _aggregate_path_changes(
        self,
        path: str,
        changes: list[WorkspaceChange],
    ) -> WorkspaceChange:
        first = changes[0]
        last = changes[-1]
        before = first.before
        after = last.after
        if before is None and after is None:
            status = 'changed'
        elif before is None:
            status = 'added'
        elif after is None:
            status = 'deleted'
        else:
            status = 'modified'
        diff, diff_truncated = self._join_change_diffs(changes)
        return WorkspaceChange(
            path=path,
            status=status,
            before=before,
            after=after,
            added_lines=sum(change.added_lines for change in changes),
            removed_lines=sum(change.removed_lines for change in changes),
            diff=diff,
            diff_truncated=diff_truncated,
        )

    def _join_change_diffs(
        self,
        changes: list[WorkspaceChange],
    ) -> tuple[str, bool]:
        if len(changes) == 1:
            change = changes[0]
            return change.diff, change.diff_truncated

        rendered_lines: list[str] = []
        diff_truncated = any(change.diff_truncated for change in changes)
        for index, change in enumerate(changes, start=1):
            if not change.diff.strip():
                continue
            if rendered_lines:
                rendered_lines.append('')
            rendered_lines.append(
                f'# change {index}: {change.status} {change.path} '
                f'(+{change.added_lines} -{change.removed_lines})'
            )
            rendered_lines.extend(change.diff.rstrip().splitlines())
        if len(rendered_lines) > self.max_diff_lines:
            rendered_lines = rendered_lines[: self.max_diff_lines]
            diff_truncated = True
        rendered = '\n'.join(rendered_lines)
        if len(rendered) > self.max_diff_chars:
            rendered = rendered[: self.max_diff_chars].rstrip()
            diff_truncated = True
        if diff_truncated and rendered:
            rendered = rendered.rstrip() + '\n...[diff truncated]...'
        return rendered, diff_truncated

    def _changes_to_event(
        self,
        changes: list[WorkspaceChange],
        *,
        event_type: str,
    ) -> dict[str, object]:
        visible_changes = changes[: self.max_event_files]
        truncated_file_count = max(0, len(changes) - len(visible_changes))
        added_lines = sum(change.added_lines for change in changes)
        removed_lines = sum(change.removed_lines for change in changes)
        status_counts = {
            'added': sum(1 for change in changes if change.status == 'added'),
            'modified': sum(
                1
                for change in changes
                if change.status not in {'added', 'deleted'}
            ),
            'deleted': sum(1 for change in changes if change.status == 'deleted'),
        }
        changed_paths = [change.path for change in changes]
        return {
            'type': event_type,
            'file_count': len(changes),
            'changed_paths': changed_paths,
            'added_files': status_counts['added'],
            'modified_files': status_counts['modified'],
            'deleted_files': status_counts['deleted'],
            'added_lines': added_lines,
            'removed_lines': removed_lines,
            'files': [change.to_event_file() for change in visible_changes],
            'truncated_file_count': truncated_file_count,
            'summary': _render_change_summary(
                file_count=len(changes),
                added_files=status_counts['added'],
                modified_files=status_counts['modified'],
                deleted_files=status_counts['deleted'],
                added_lines=added_lines,
                removed_lines=removed_lines,
            ),
        }

    def _snapshot_workspace(self) -> dict[str, FileSnapshot]:
        snapshots: dict[str, FileSnapshot] = {}
        if not self.root.exists():
            return snapshots
        for path in self._iter_workspace_files():
            snapshot = self._snapshot_file(path)
            if snapshot is not None:
                snapshots[snapshot.path] = snapshot
        return snapshots

    def _iter_workspace_files(self) -> Iterable[Path]:
        for current_root, dir_names, file_names in os.walk(self.root):
            current = Path(current_root)
            dir_names[:] = [
                name
                for name in dir_names
                if name not in _EXCLUDED_DIR_NAMES
                and not (current / name).is_symlink()
            ]
            for file_name in file_names:
                path = current / file_name
                if path.is_symlink() or not path.is_file():
                    continue
                yield path

    def _snapshot_file(self, path: Path) -> FileSnapshot | None:
        try:
            relative = path.relative_to(self.root).as_posix()
            stat = path.stat()
            digest = _sha256_file(path)
            text: str | None = None
            binary = False
            truncated = stat.st_size > self.max_text_bytes
            line_count: int | None = None
            if not truncated:
                raw = path.read_bytes()
                binary = _looks_binary(raw)
                if not binary:
                    text = raw.decode('utf-8', errors='replace')
                    line_count = _line_count(text)
            return FileSnapshot(
                path=relative,
                sha256=digest,
                size=stat.st_size,
                text=text,
                binary=binary,
                truncated=truncated,
                line_count=line_count,
            )
        except OSError:
            return None

    def _diff_snapshots(
        self,
        before: dict[str, FileSnapshot],
        after: dict[str, FileSnapshot],
    ) -> list[WorkspaceChange]:
        changes: list[WorkspaceChange] = []
        for path in sorted(set(before) | set(after)):
            before_snapshot = before.get(path)
            after_snapshot = after.get(path)
            if (
                before_snapshot is not None
                and after_snapshot is not None
                and before_snapshot.sha256 == after_snapshot.sha256
                and before_snapshot.size == after_snapshot.size
            ):
                continue
            changes.append(
                self._build_change(path, before_snapshot, after_snapshot)
            )
        return changes

    def _build_change(
        self,
        path: str,
        before: FileSnapshot | None,
        after: FileSnapshot | None,
    ) -> WorkspaceChange:
        if before is None:
            status = 'added'
        elif after is None:
            status = 'deleted'
        else:
            status = 'modified'

        diff, added_lines, removed_lines, diff_truncated = self._build_unified_diff(
            path,
            before,
            after,
        )
        return WorkspaceChange(
            path=path,
            status=status,
            before=before,
            after=after,
            added_lines=added_lines,
            removed_lines=removed_lines,
            diff=diff,
            diff_truncated=diff_truncated,
        )

    def _build_unified_diff(
        self,
        path: str,
        before: FileSnapshot | None,
        after: FileSnapshot | None,
    ) -> tuple[str, int, int, bool]:
        before_text = before.text if before is not None else ''
        after_text = after.text if after is not None else ''
        if before_text is None or after_text is None:
            return '', 0, 0, False

        diff_lines = list(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=f'a/{path}',
                tofile=f'b/{path}',
                lineterm='',
            )
        )
        added_lines = sum(
            1
            for line in diff_lines
            if line.startswith('+') and not line.startswith('+++')
        )
        removed_lines = sum(
            1
            for line in diff_lines
            if line.startswith('-') and not line.startswith('---')
        )
        rendered_lines = diff_lines
        diff_truncated = False
        if len(rendered_lines) > self.max_diff_lines:
            rendered_lines = rendered_lines[: self.max_diff_lines]
            diff_truncated = True
        rendered = '\n'.join(rendered_lines)
        if len(rendered) > self.max_diff_chars:
            rendered = rendered[: self.max_diff_chars].rstrip()
            diff_truncated = True
        if diff_truncated:
            rendered = rendered.rstrip() + '\n...[diff truncated]...'
        return rendered, added_lines, removed_lines, diff_truncated


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_binary(raw: bytes) -> bool:
    if not raw:
        return False
    if b'\0' in raw[:4096]:
        return True
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        return True
    return False


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count('\n') + (0 if text.endswith('\n') else 1)


def _render_change_summary(
    *,
    file_count: int,
    added_files: int,
    modified_files: int,
    deleted_files: int,
    added_lines: int,
    removed_lines: int,
) -> str:
    return (
        f'{file_count} file(s): '
        f'{added_files} added, {modified_files} modified, {deleted_files} deleted; '
        f'+{added_lines} -{removed_lines}'
    )
