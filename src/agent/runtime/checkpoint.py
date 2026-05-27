"""Workspace file checkpointing for conversation rollback.

Captures full workspace snapshots at session initialization and after each
assistant turn so that ``/rewind`` can physically restore the filesystem
to a previous state.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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

DEFAULT_CHECKPOINTS_DIR_NAME = 'checkpoints'


def _checkpoints_root(session_dir: Path) -> Path:
    return session_dir / DEFAULT_CHECKPOINTS_DIR_NAME


def _checkpoint_dir(session_dir: Path, session_id: str, message_count: int) -> Path:
    return _checkpoints_root(session_dir) / session_id / str(message_count)


def _iter_workspace_files(workspace: Path) -> list[str]:
    """Return relative POSIX paths for all trackable files in *workspace*."""
    results: list[str] = []
    for current_root, dir_names, file_names in os.walk(workspace):
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
            results.append(path.relative_to(workspace).as_posix())
    return results


def create_checkpoint(
    workspace: Path,
    session_id: str,
    message_count: int,
    session_dir: Path,
) -> Path:
    """Snapshot every trackable file in *workspace* into a checkpoint folder.

    Returns the checkpoint directory path.
    """
    target = _checkpoint_dir(session_dir, session_id, message_count)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for rel_posix in _iter_workspace_files(workspace):
        src = workspace / rel_posix
        dst = target / rel_posix
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    return target


def restore_checkpoint(
    workspace: Path,
    session_id: str,
    message_count: int,
    session_dir: Path,
) -> bool:
    """Restore the workspace to the state saved in the given checkpoint.

    1. Delete all trackable files in *workspace* that are not in the
       checkpoint (files added after the checkpoint was taken).
    2. Overwrite all trackable files in *workspace* with the checkpoint
       versions (files modified after the checkpoint was taken).
    3. Copy back any files that exist in the checkpoint but are missing
       from the workspace (files deleted after the checkpoint was taken).

    Returns ``True`` if the checkpoint existed and was restored, ``False``
    otherwise.
    """
    source = _checkpoint_dir(session_dir, session_id, message_count)
    if not source.is_dir():
        return False

    checkpoint_files = set(_iter_workspace_files(source))
    workspace_files = set(_iter_workspace_files(workspace))

    # 1. Remove files that exist in workspace but not in checkpoint.
    for rel_posix in sorted(workspace_files - checkpoint_files):
        target = workspace / rel_posix
        if target.is_file():
            target.unlink()

    # 2+3. Copy/overwrite checkpoint files into workspace.
    for rel_posix in sorted(checkpoint_files):
        src = source / rel_posix
        dst = workspace / rel_posix
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    # Clean up empty directories left after deletions.
    _remove_empty_dirs(workspace)

    return True


def cleanup_subsequent_checkpoints(
    session_id: str,
    max_allowed_count: int,
    session_dir: Path,
) -> int:
    """Delete all checkpoints with ``message_count > max_allowed_count``.

    Returns the number of checkpoints deleted.
    """
    session_checkpoint_root = _checkpoints_root(session_dir) / session_id
    if not session_checkpoint_root.is_dir():
        return 0
    deleted = 0
    for entry in sorted(session_checkpoint_root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            count = int(entry.name)
        except ValueError:
            continue
        if count > max_allowed_count:
            shutil.rmtree(entry)
            deleted += 1
    return deleted


def checkpoint_exists(
    session_id: str,
    message_count: int,
    session_dir: Path,
) -> bool:
    """Return ``True`` if a checkpoint directory exists for the given point."""
    return _checkpoint_dir(session_dir, session_id, message_count).is_dir()


def list_checkpoints(session_id: str, session_dir: Path) -> list[int]:
    """Return sorted list of message counts that have checkpoints."""
    session_checkpoint_root = _checkpoints_root(session_dir) / session_id
    if not session_checkpoint_root.is_dir():
        return []
    counts: list[int] = []
    for entry in session_checkpoint_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            counts.append(int(entry.name))
        except ValueError:
            continue
    return sorted(counts)


def _remove_empty_dirs(root: Path) -> None:
    """Walk bottom-up removing empty directories, respecting exclusions."""
    for current_root, dir_names, file_names in os.walk(root, topdown=False):
        current = Path(current_root)
        if current == root:
            continue
        try:
            relative_parts = current.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _EXCLUDED_DIR_NAMES for part in relative_parts):
            continue
        try:
            if not any(current.iterdir()):
                current.rmdir()
        except OSError:
            pass
