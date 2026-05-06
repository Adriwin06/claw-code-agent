from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import mimetypes
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlparse
from uuid import uuid4

from .workspace_files import WorkspacePathReference, format_file_size


_IMAGE_EXTENSIONS = {
    '.apng',
    '.avif',
    '.bmp',
    '.gif',
    '.heic',
    '.heif',
    '.jpeg',
    '.jpg',
    '.png',
    '.tif',
    '.tiff',
    '.webp',
}


@dataclass(frozen=True)
class PromptAttachment:
    original_path: str
    workspace_path: str
    name: str
    kind: str
    size_bytes: int
    mime_type: str | None = None

    @property
    def label(self) -> str:
        return f'{self.name} ({self.kind}, {format_file_size(self.size_bytes)})'


def parse_pasted_file_paths(text: str) -> tuple[Path, ...]:
    candidates = _candidate_path_strings(text)
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = _path_from_candidate(candidate)
        if path is None or not path.exists() or not path.is_file():
            continue
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return tuple(paths)


def copy_external_attachment(
    source: Path,
    workspace: Path,
    *,
    batch_id: str | None = None,
) -> PromptAttachment | None:
    source_path = Path(source).expanduser().resolve()
    workspace_root = Path(workspace).expanduser().resolve()
    try:
        source_path.relative_to(workspace_root)
    except ValueError:
        pass
    else:
        return None

    target_root = workspace_root / '.port_sessions' / 'attachments'
    if batch_id:
        target_root = target_root / _safe_path_name(batch_id)
    else:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        target_root = target_root / f'{timestamp}-{uuid4().hex[:8]}'
    target_root.mkdir(parents=True, exist_ok=True)
    target = _unique_target_path(target_root, source_path.name)
    shutil.copy2(source_path, target)
    relative_target = target.relative_to(workspace_root).as_posix()
    mime_type = mimetypes.guess_type(str(source_path))[0]
    kind = 'image' if source_path.suffix.lower() in _IMAGE_EXTENSIONS else 'file'
    return PromptAttachment(
        original_path=str(source_path),
        workspace_path=relative_target,
        name=source_path.name,
        kind=kind,
        size_bytes=target.stat().st_size,
        mime_type=mime_type,
    )


def workspace_reference_for_pasted_path(path: Path, workspace: Path) -> str | None:
    workspace_root = Path(workspace).expanduser().resolve()
    try:
        relative = Path(path).expanduser().resolve().relative_to(workspace_root)
    except (OSError, ValueError):
        return None
    if not relative.parts:
        return None
    return relative.as_posix()


def build_prompt_with_references(
    prompt: str,
    *,
    attachments: tuple[PromptAttachment, ...] = (),
    workspace_references: tuple[WorkspacePathReference, ...] = (),
) -> str:
    context = render_prompt_reference_context(
        attachments=attachments,
        workspace_references=workspace_references,
    )
    if not context:
        return prompt
    return f'{prompt.rstrip()}\n\n{context}'


def build_display_prompt_with_attachments(
    prompt: str,
    attachments: tuple[PromptAttachment, ...],
) -> str:
    if not attachments:
        return prompt
    lines = [prompt.rstrip(), '', 'Attached files:']
    for attachment in attachments:
        lines.append(f'- {attachment.label} -> {attachment.workspace_path}')
    return '\n'.join(lines).strip()


def render_prompt_reference_context(
    *,
    attachments: tuple[PromptAttachment, ...] = (),
    workspace_references: tuple[WorkspacePathReference, ...] = (),
) -> str:
    if not attachments and not workspace_references:
        return ''
    lines = [
        '<system-reminder>',
        'Claw UI prompt context:',
    ]
    if attachments:
        lines.extend(['', 'External files copied into the workspace for this turn:'])
        for attachment in attachments:
            details = [
                f'kind={attachment.kind}',
                f'name={attachment.name}',
                f'copied_to={attachment.workspace_path}',
                f'original={attachment.original_path}',
                f'size={format_file_size(attachment.size_bytes)}',
            ]
            if attachment.mime_type:
                details.append(f'mime={attachment.mime_type}')
            lines.append('- ' + '; '.join(details))
        lines.append(
            'Use copied_to paths when calling workspace file tools; original paths may be outside the workspace.'
        )
    if workspace_references:
        lines.extend(['', 'Workspace paths referenced by the user with @:'])
        for reference in workspace_references:
            details = [
                f'path={reference.path}',
                f'kind={reference.kind}',
            ]
            if reference.size_bytes is not None:
                details.append(f'size={format_file_size(reference.size_bytes)}')
            lines.append('- ' + '; '.join(details))
    lines.append('</system-reminder>')
    return '\n'.join(lines)


def render_attachment_summary(attachments: tuple[PromptAttachment, ...]) -> str:
    if not attachments:
        return ''
    lines = ['External attachments']
    for index, attachment in enumerate(attachments, start=1):
        lines.append(
            f'{index}. {attachment.label} -> {attachment.workspace_path}'
        )
    return '\n'.join(lines)


def _candidate_path_strings(text: str) -> tuple[str, ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    candidates: list[str] = [stripped]
    candidates.extend(line.strip() for line in stripped.splitlines() if line.strip())
    candidates.extend(_split_quoted_candidates(stripped))
    return tuple(candidate for candidate in candidates if candidate)


def _split_quoted_candidates(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in text:
        if quote is not None:
            if character == quote:
                if current:
                    candidates.append(''.join(current))
                current = []
                quote = None
            else:
                current.append(character)
            continue
        if character in {'"', "'"}:
            if current:
                candidates.append(''.join(current).strip())
                current = []
            quote = character
            continue
        if character in {'\r', '\n', '\t'}:
            if current:
                candidates.append(''.join(current).strip())
                current = []
            continue
        current.append(character)
    if current:
        candidates.append(''.join(current).strip())
    return tuple(candidate for candidate in candidates if candidate)


def _path_from_candidate(candidate: str) -> Path | None:
    cleaned = candidate.strip()
    if cleaned.startswith('& '):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.strip('"\'')
    if cleaned.lower().startswith('file:'):
        cleaned = _path_from_file_uri(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return Path(cleaned).expanduser()


def _path_from_file_uri(value: str) -> str:
    parsed = urlparse(value)
    raw_path = unquote(parsed.path or '')
    if parsed.netloc and not raw_path.startswith('/'):
        raw_path = f'//{parsed.netloc}/{raw_path}'
    elif parsed.netloc:
        raw_path = f'//{parsed.netloc}{raw_path}'
    if re.match(r'^/[A-Za-z]:/', raw_path):
        raw_path = raw_path[1:]
    return raw_path


def _safe_path_name(name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._-')
    return safe or 'attachment'


def _unique_target_path(directory: Path, filename: str) -> Path:
    safe_name = _safe_path_name(Path(filename).stem)
    suffix = Path(filename).suffix
    candidate = directory / f'{safe_name}{suffix}'
    counter = 2
    while candidate.exists():
        candidate = directory / f'{safe_name}-{counter}{suffix}'
        counter += 1
    return candidate
