from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.agent.commands.slash import get_slash_command_specs


@dataclass(frozen=True)
class SlashCommandSuggestion:
    primary_name: str
    aliases: tuple[str, ...]
    description: str
    insertion_text_override: str | None = None
    label_override: str | None = None
    detail_lines: tuple[str, ...] = ()
    kind: str = 'command'

    @property
    def insertion_text(self) -> str:
        if self.insertion_text_override is not None:
            return self.insertion_text_override
        return f'/{self.primary_name} '

    @property
    def label(self) -> str:
        if self.label_override is not None:
            return self.label_override
        return f'/{self.primary_name}'


def build_slash_command_suggestions() -> tuple[SlashCommandSuggestion, ...]:
    return tuple(
        SlashCommandSuggestion(
            primary_name=spec.names[0],
            aliases=spec.names,
            description=spec.description,
        )
        for spec in get_slash_command_specs()
    )


def extract_slash_command_query(text: str) -> str | None:
    trimmed = text.lstrip()
    if not trimmed.startswith('/'):
        return None
    without_slash = trimmed[1:]
    if any(character.isspace() for character in without_slash):
        return None
    return without_slash.lower()


def extract_rewind_message_query(text: str) -> str | None:
    trimmed = text.lstrip()
    if not trimmed.startswith('/'):
        return None
    without_slash = trimmed[1:]
    command, separator, rest = without_slash.partition(' ')
    if separator == '':
        return None
    if command.lower() not in {'rewind', 'checkpoint'}:
        return None
    query = rest.lstrip()
    if any(character.isspace() for character in query):
        return None
    return query.lower()


def filter_slash_command_suggestions(
    text: str,
    suggestions: tuple[SlashCommandSuggestion, ...] | None = None,
    messages: Sequence[Any] | None = None,
) -> list[SlashCommandSuggestion]:
    query = extract_slash_command_query(text)
    if query is not None:
        if suggestions is None:
            suggestions = build_slash_command_suggestions()
        if not query:
            return list(suggestions)

        scored: list[tuple[int, int, SlashCommandSuggestion]] = []
        for index, suggestion in enumerate(suggestions):
            aliases = tuple(alias.lower() for alias in suggestion.aliases)
            if any(alias == query for alias in aliases):
                score = 0
            elif any(alias.startswith(query) for alias in aliases):
                score = 1
            elif any(query in alias for alias in aliases):
                score = 2
            else:
                continue
            scored.append((score, index, suggestion))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [suggestion for _, _, suggestion in scored]

    rewind_query = extract_rewind_message_query(text)
    if rewind_query is None:
        return []
    return build_rewind_message_suggestions(messages or (), query=rewind_query)


def build_rewind_message_suggestions(
    messages: Sequence[Any],
    *,
    query: str = '',
    limit: int = 40,
) -> list[SlashCommandSuggestion]:
    scored: list[tuple[int, int, SlashCommandSuggestion]] = []
    lowered_query = query.lower()
    for index, message in enumerate(messages):
        message_id = _message_identifier(message)
        if not message_id:
            continue
        role = _message_role(message).upper()
        preview = _message_preview(message, max_chars=64)
        searchable = f'{message_id.lower()} {role.lower()} {preview.lower()}'
        if lowered_query:
            lowered_id = message_id.lower()
            if lowered_id == lowered_query:
                score = 0
            elif lowered_id.startswith(lowered_query):
                score = 1
            elif lowered_query in searchable:
                score = 2
            else:
                continue
        else:
            score = 1
        label = f'{message_id} [{role}]'
        if preview:
            label += f' {preview}'
        suggestion = SlashCommandSuggestion(
            primary_name=f'rewind-message-{index}',
            aliases=(message_id,),
            description=(
                f'Rewind to message id `{message_id}` at transcript index {index}. '
                'Executing it deletes every later message and restores a matching '
                'workspace checkpoint when one exists.'
            ),
            insertion_text_override=f'/rewind {message_id}',
            label_override=label,
            detail_lines=(
                f'index={index}',
                f'role={role.lower()}',
                f'message_id={message_id}',
            ),
            kind='rewind_message',
        )
        scored.append((score, -index, suggestion))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [suggestion for _, _, suggestion in scored[:limit]]


def render_slash_command_suggestion_detail(suggestion: SlashCommandSuggestion) -> str:
    lines = [
        suggestion.label,
        '',
        suggestion.description,
    ]
    if suggestion.detail_lines:
        lines.extend(['', *suggestion.detail_lines])
    if suggestion.kind == 'command' and len(suggestion.aliases) > 1:
        aliases = ', '.join(f'/{alias}' for alias in suggestion.aliases[1:])
        lines.extend(
            [
                '',
                f'aliases={aliases}',
            ]
        )
    lines.extend(
        [
            '',
            'Controls',
            '',
            'Up/Down: move selection',
            'Tab: insert command or target',
            'Enter: insert when incomplete, run when exact',
            'Mouse: click an item to insert it',
        ]
    )
    return '\n'.join(lines)


def _message_identifier(message: Any) -> str | None:
    if isinstance(message, dict):
        message_id = message.get('message_id')
        if isinstance(message_id, str) and message_id:
            return message_id
        metadata = message.get('metadata')
        if isinstance(metadata, dict):
            lineage_id = metadata.get('lineage_id')
            if isinstance(lineage_id, str) and lineage_id:
                return lineage_id
        return None

    message_id = getattr(message, 'message_id', None)
    if isinstance(message_id, str) and message_id:
        return message_id
    metadata = getattr(message, 'metadata', None)
    if isinstance(metadata, dict):
        lineage_id = metadata.get('lineage_id')
        if isinstance(lineage_id, str) and lineage_id:
            return lineage_id
    return None


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get('role')
    else:
        role = getattr(message, 'role', None)
    return role if isinstance(role, str) and role else 'message'


def _message_preview(message: Any, *, max_chars: int) -> str:
    if isinstance(message, dict):
        content = message.get('content')
    else:
        content = getattr(message, 'content', None)
    if not isinstance(content, str):
        return ''
    preview = ' '.join(content.split())
    if len(preview) <= max_chars:
        return preview
    return preview[: max_chars - 3].rstrip() + '...'
