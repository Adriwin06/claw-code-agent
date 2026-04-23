from __future__ import annotations


def _preview_value(value: object, *, max_chars: int = 120) -> str:
    if value is None:
        return ''
    text = str(value)
    text = ' '.join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3] + '...'
    return text


def _preview_multiline(text: str, *, max_chars: int = 220, max_lines: int = 4) -> str:
    if not text:
        return ''
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ''
    clipped = lines[-max_lines:]
    rendered = '\n'.join(clipped)
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:]
        if not rendered.startswith('...'):
            rendered = '...' + rendered[3:]
    return rendered


def _friendly_stop_reason(stop_reason: str | None) -> str:
    if stop_reason is None:
        return 'completed'
    normalized = str(stop_reason).strip()
    if not normalized:
        return 'completed'
    if normalized.lower() in {'stop', 'completed', 'complete', 'done', 'end_turn'}:
        return 'completed'
    return normalized
