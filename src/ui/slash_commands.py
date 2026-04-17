from __future__ import annotations

from dataclasses import dataclass

from src.agent.agent_slash_commands import get_slash_command_specs


@dataclass(frozen=True)
class SlashCommandSuggestion:
    primary_name: str
    aliases: tuple[str, ...]
    description: str

    @property
    def insertion_text(self) -> str:
        return f'/{self.primary_name} '

    @property
    def label(self) -> str:
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


def filter_slash_command_suggestions(
    text: str,
    suggestions: tuple[SlashCommandSuggestion, ...] | None = None,
) -> list[SlashCommandSuggestion]:
    query = extract_slash_command_query(text)
    if query is None:
        return []
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


def render_slash_command_suggestion_detail(suggestion: SlashCommandSuggestion) -> str:
    lines = [
        suggestion.label,
        '',
        suggestion.description,
    ]
    if len(suggestion.aliases) > 1:
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
            'Tab: insert command',
            'Enter: insert when incomplete, run when exact',
            'Mouse: click a command to insert it',
        ]
    )
    return '\n'.join(lines)
