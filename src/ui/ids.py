from __future__ import annotations


def should_route_key_to_prompt(
    *,
    character: str | None,
    prompt_focused: bool,
    busy: bool,
) -> bool:
    if busy or prompt_focused:
        return False
    if not isinstance(character, str) or len(character) != 1:
        return False
    return character.isprintable()


def build_working_section_id(conversation_id: str, turn_id: str) -> str:
    return f'working-{conversation_id}-{turn_id}'


def build_working_section_instance_id(
    conversation_id: str,
    turn_id: str,
    *,
    section_index: int,
    section_count: int,
) -> str:
    base_id = build_working_section_id(conversation_id, turn_id)
    if section_count <= 1:
        return base_id
    return f'{base_id}-section-{section_index}'
