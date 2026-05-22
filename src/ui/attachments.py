from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import mimetypes
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
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
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024


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


@dataclass(frozen=True)
class PromptPathMatch:
    path: Path
    start_index: int
    end_index: int


def parse_pasted_file_paths(text: str) -> tuple[Path, ...]:
    return tuple(match.path for match in extract_prompt_file_path_matches(text))


def extract_prompt_file_path_matches(text: str) -> tuple[PromptPathMatch, ...]:
    candidates = _candidate_path_spans(text)
    matches: list[PromptPathMatch] = []
    seen_paths: set[str] = set()
    used_spans: list[tuple[int, int]] = []
    for candidate, start_index, end_index in candidates:
        if any(_spans_overlap((start_index, end_index), span) for span in used_spans):
            continue
        path = _existing_path_from_candidate(candidate)
        if path is None or not _path_exists(path) or not _path_is_file(path):
            continue
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        used_spans.append((start_index, end_index))
        matches.append(
            PromptPathMatch(
                path=path,
                start_index=start_index,
                end_index=end_index,
            )
        )
    return tuple(matches)


def extract_unresolved_prompt_file_path_candidates(text: str) -> tuple[str, ...]:
    resolved_spans = [
        (match.start_index, match.end_index)
        for match in extract_prompt_file_path_matches(text)
    ]
    unresolved: list[str] = []
    seen: set[str] = set()
    for candidate, start_index, end_index in _candidate_path_spans(text):
        if any(_spans_overlap((start_index, end_index), span) for span in resolved_spans):
            continue
        cleaned = _clean_candidate(candidate)
        if not cleaned or not _looks_like_file_path_candidate(cleaned):
            continue
        path = _existing_path_from_candidate(candidate)
        if path is not None and _path_exists(path) and _path_is_file(path):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        unresolved.append(cleaned)
    return tuple(unresolved)


def extract_prompt_file_paths(text: str) -> tuple[str, tuple[Path, ...]]:
    matches = extract_prompt_file_path_matches(text)
    if not matches:
        return text, ()
    cleaned_parts: list[str] = []
    cursor = 0
    for match in sorted(matches, key=lambda item: item.start_index):
        cleaned_parts.append(text[cursor:match.start_index])
        cursor = match.end_index
    cleaned_parts.append(text[cursor:])
    cleaned_prompt = _normalize_prompt_without_path_tokens(''.join(cleaned_parts))
    return cleaned_prompt, tuple(match.path for match in matches)


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

    target_root = _attachment_target_root(workspace_root, batch_id=batch_id)
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


def copy_clipboard_image_attachment(
    workspace: Path,
    *,
    batch_id: str | None = None,
) -> PromptAttachment | None:
    workspace_root = Path(workspace).expanduser().resolve()
    target_root = _attachment_target_root(workspace_root, batch_id=batch_id)
    target_root.mkdir(parents=True, exist_ok=True)
    target = _unique_target_path(target_root, 'clipboard-image.png')
    if not _write_clipboard_image(target):
        try:
            target.unlink()
        except OSError:
            pass
        return None
    return PromptAttachment(
        original_path='clipboard',
        workspace_path=target.relative_to(workspace_root).as_posix(),
        name=target.name,
        kind='image',
        size_bytes=target.stat().st_size,
        mime_type='image/png',
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


def build_prompt_image_blocks(
    attachments: tuple[PromptAttachment, ...],
    workspace: Path,
    *,
    max_inline_bytes: int = _MAX_INLINE_IMAGE_BYTES,
) -> tuple[dict[str, object], ...]:
    workspace_root = Path(workspace).expanduser().resolve()
    blocks: list[dict[str, object]] = []
    for attachment in attachments:
        if attachment.kind != 'image':
            continue
        image_path = (workspace_root / attachment.workspace_path).resolve()
        try:
            image_path.relative_to(workspace_root)
        except ValueError:
            continue
        try:
            payload = image_path.read_bytes()
        except OSError:
            continue
        if len(payload) > max_inline_bytes:
            continue
        mime_type = attachment.mime_type or mimetypes.guess_type(str(image_path))[0] or 'image/png'
        encoded = base64.b64encode(payload).decode('ascii')
        blocks.append(
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{mime_type};base64,{encoded}',
                },
            }
        )
    return tuple(blocks)


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
    return tuple(candidate for candidate, _, _ in _candidate_path_spans(text))


def _candidate_path_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    stripped = text.strip()
    if not stripped:
        return ()
    candidates: list[tuple[str, int, int]] = []
    text_start = len(text) - len(text.lstrip())
    text_end = len(text.rstrip())
    candidates.append((text[text_start:text_end], text_start, text_end))
    for match in re.finditer(r'[^\r\n]+', text):
        line_start = match.start()
        line_text = match.group(0)
        trimmed_start = len(line_text) - len(line_text.lstrip())
        trimmed_end = len(line_text.rstrip())
        if trimmed_start < trimmed_end:
            candidates.append(
                (
                    line_text[trimmed_start:trimmed_end],
                    line_start + trimmed_start,
                    line_start + trimmed_end,
                )
            )
    candidates.extend(_split_quoted_candidate_spans(text))
    candidates.extend(_split_unquoted_candidate_spans(text))
    seen: set[tuple[str, int, int]] = set()
    unique: list[tuple[str, int, int]] = []
    for candidate in candidates:
        if not candidate[0] or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return tuple(unique)


def _split_quoted_candidates(text: str) -> tuple[str, ...]:
    return tuple(candidate for candidate, _, _ in _split_quoted_candidate_spans(text))


def _split_quoted_candidate_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    current: list[str] = []
    quote: str | None = None
    token_start = 0
    content_start = 0
    spans: list[tuple[str, int, int]] = []
    for index, character in enumerate(text):
        if quote is not None:
            if character == quote:
                if current:
                    spans.append((''.join(current), token_start, index + 1))
                current = []
                quote = None
            else:
                current.append(character)
            continue
        if character in {'"', "'"}:
            if current:
                candidate = ''.join(current).strip()
                if candidate:
                    leading = len(''.join(current)) - len(''.join(current).lstrip())
                    trailing = len(''.join(current).rstrip())
                    spans.append(
                        (
                            candidate,
                            content_start + leading,
                            content_start + trailing,
                        )
                    )
                current = []
            quote = character
            token_start = index
            content_start = index + 1
            continue
        if character in {'\r', '\n', '\t'}:
            if current:
                raw = ''.join(current)
                candidate = raw.strip()
                if candidate:
                    leading = len(raw) - len(raw.lstrip())
                    trailing = len(raw.rstrip())
                    spans.append((candidate, content_start + leading, content_start + trailing))
                current = []
            continue
        if not current:
            content_start = index
        current.append(character)
    if current:
        raw = ''.join(current)
        candidate = raw.strip()
        if candidate:
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            spans.append((candidate, content_start + leading, content_start + trailing))
    return tuple(spans)


def _split_unquoted_candidate_spans(text: str) -> tuple[tuple[str, int, int], ...]:
    spans: list[tuple[str, int, int]] = []
    patterns = (
        r'(?<!\S)(?:&\s*)?file:///[^\s"\']+',
        r'(?<!\S)(?:&\s*)?[A-Za-z]:[\\/][^\s"\']+',
        r'(?<!\S)(?:&\s*)?/[^\s"\']+',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start_index = match.start()
            end_index = match.end()
            value = text[start_index:end_index].rstrip('.,;:')
            spans.append((value, start_index, start_index + len(value)))
    return tuple(spans)


def _normalize_prompt_without_path_tokens(text: str) -> str:
    lines = []
    for line in text.splitlines():
        normalized = re.sub(r'[ \t]{2,}', ' ', line).strip()
        if normalized:
            lines.append(normalized)
    return '\n'.join(lines)


def _spans_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _path_from_candidate(candidate: str) -> Path | None:
    cleaned = _clean_candidate(candidate)
    if cleaned.lower().startswith('file:'):
        cleaned = _path_from_file_uri(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return Path(cleaned).expanduser()


def _existing_path_from_candidate(candidate: str) -> Path | None:
    path = _path_from_candidate(candidate)
    if path is not None and _path_exists(path):
        return path
    mapped_path = _mapped_host_path_from_candidate(candidate)
    if mapped_path is not None and _path_exists(mapped_path):
        return mapped_path
    return path


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _clean_candidate(candidate: str) -> str:
    cleaned = candidate.strip()
    if cleaned.startswith('& '):
        cleaned = cleaned[2:].strip()
    return cleaned.strip('"\'')


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


def _mapped_host_path_from_candidate(candidate: str) -> Path | None:
    cleaned = _clean_candidate(candidate)
    if cleaned.lower().startswith('file:'):
        cleaned = _path_from_file_uri(cleaned)
    for host_root, container_root in _host_path_mappings():
        relative = _relative_to_host_root(cleaned, host_root)
        if relative is None:
            continue
        return Path(container_root).expanduser().joinpath(*relative)
    return None


def _host_path_mappings() -> tuple[tuple[str, str], ...]:
    mappings: list[tuple[str, str]] = []
    _append_host_mapping(
        mappings,
        os.environ.get('CLAW_HOST_WORKSPACE'),
        os.environ.get('CLAW_CONTAINER_WORKSPACE', '/workspace'),
    )
    _append_host_mapping(
        mappings,
        os.environ.get('CLAW_HOST_ATTACHMENTS_ROOT'),
        os.environ.get('CLAW_CONTAINER_HOST_ATTACHMENTS_ROOT', '/host-attachments'),
    )
    _append_host_mapping(
        mappings,
        os.environ.get('CLAW_HOST_ATTACHMENTS_MOUNT_ROOT'),
        os.environ.get('CLAW_CONTAINER_HOST_ATTACHMENTS_ROOT', '/host-attachments'),
    )
    _append_host_mapping(
        mappings,
        os.environ.get('CLAW_HOST_HOME'),
        os.environ.get('CLAW_CONTAINER_HOST_HOME', '/host-home'),
    )
    return tuple(mappings)


def _append_host_mapping(
    mappings: list[tuple[str, str]],
    host_root: str | None,
    container_root: str | None,
) -> None:
    if not host_root or not container_root:
        return
    normalized = (host_root.strip(), container_root.strip())
    if not normalized[0] or not normalized[1] or normalized in mappings:
        return
    mappings.append(normalized)


def _relative_to_host_root(value: str, host_root: str) -> tuple[str, ...] | None:
    if _is_windows_path(value) or _is_windows_path(host_root):
        return _relative_windows_path(value, host_root)
    return _relative_posix_path(value, host_root)


def _relative_windows_path(value: str, host_root: str) -> tuple[str, ...] | None:
    value_path = PureWindowsPath(value)
    root_path = PureWindowsPath(host_root)
    value_parts = tuple(value_path.parts)
    root_parts = tuple(root_path.parts)
    if len(value_parts) < len(root_parts):
        return None
    if tuple(part.casefold() for part in value_parts[: len(root_parts)]) != tuple(
        part.casefold() for part in root_parts
    ):
        return None
    return tuple(part for part in value_parts[len(root_parts) :] if part not in {'', '\\'})


def _relative_posix_path(value: str, host_root: str) -> tuple[str, ...] | None:
    value_path = PurePosixPath(value)
    root_path = PurePosixPath(host_root)
    value_parts = tuple(value_path.parts)
    root_parts = tuple(root_path.parts)
    if len(value_parts) < len(root_parts):
        return None
    if value_parts[: len(root_parts)] != root_parts:
        return None
    return tuple(part for part in value_parts[len(root_parts) :] if part not in {'', '/'})


def _is_windows_path(value: str) -> bool:
    return bool(re.match(r'^[A-Za-z]:[\\/]', value)) or value.startswith('\\\\')


def _looks_like_file_path_candidate(value: str) -> bool:
    if _contains_non_file_url(value):
        return False
    if '"' in value:
        return False
    if value.lower().startswith('file:'):
        return True
    if _is_windows_path(value):
        return True
    if value.startswith('/'):
        return '/' in value[1:] or bool(PurePosixPath(value).suffix)
    if value.startswith('~/'):
        return True
    return any(separator in value for separator in ('/', '\\')) and bool(Path(value).suffix)


def _contains_non_file_url(value: str) -> bool:
    for match in re.finditer(r'\b([A-Za-z][A-Za-z0-9+.-]*):\/\/', value):
        if match.group(1).casefold() != 'file':
            return True
    return False


def _safe_path_name(name: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._-')
    return safe or 'attachment'


def _attachment_target_root(workspace_root: Path, *, batch_id: str | None) -> Path:
    target_root = workspace_root / '.port_sessions' / 'attachments'
    if batch_id:
        return target_root / _safe_path_name(batch_id)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return target_root / f'{timestamp}-{uuid4().hex[:8]}'


def _unique_target_path(directory: Path, filename: str) -> Path:
    safe_name = _safe_path_name(Path(filename).stem)
    suffix = Path(filename).suffix
    candidate = directory / f'{safe_name}{suffix}'
    counter = 2
    while candidate.exists():
        candidate = directory / f'{safe_name}-{counter}{suffix}'
        counter += 1
    return candidate


def _write_clipboard_image(target: Path) -> bool:
    if os.name == 'nt':
        return _write_windows_clipboard_image(target)
    if shutil.which('pngpaste'):
        result = subprocess.run(
            ['pngpaste', str(target)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0 and target.exists()
    if shutil.which('wl-paste'):
        with target.open('wb') as output:
            result = subprocess.run(
                ['wl-paste', '--type', 'image/png'],
                stdout=output,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return result.returncode == 0 and target.exists() and target.stat().st_size > 0
    if shutil.which('xclip'):
        with target.open('wb') as output:
            result = subprocess.run(
                ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o'],
                stdout=output,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return result.returncode == 0 and target.exists() and target.stat().st_size > 0
    return False


def _write_windows_clipboard_image(target: Path) -> bool:
    powershell = shutil.which('powershell.exe') or shutil.which('powershell')
    if not powershell:
        return False
    script = (
        "$ErrorActionPreference='Stop';"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "Add-Type -AssemblyName System.Drawing;"
        "if (-not [System.Windows.Forms.Clipboard]::ContainsImage()) { exit 2 };"
        "$image=[System.Windows.Forms.Clipboard]::GetImage();"
        "$image.Save($args[0], [System.Drawing.Imaging.ImageFormat]::Png)"
    )
    result = subprocess.run(
        [
            powershell,
            '-NoProfile',
            '-STA',
            '-Command',
            script,
            str(target),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and target.exists() and target.stat().st_size > 0
