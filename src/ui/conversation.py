from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from .formatting import _preview_value, sanitize_assistant_display_text


_UI_PROMPT_CONTEXT_PATTERN = re.compile(
    r'\n{0,2}<system-reminder>\nClaw UI prompt context:.*?</system-reminder>\s*',
    re.DOTALL,
)


@dataclass
class ConversationTurn:
    turn_id: str
    user_prompt: str
    assistant_response: str = ''
    assistant_status: str = 'Queued'
    phase_label: str = 'Queued'
    tool_count: int = 0
    restored: bool = False
    stop_reason: str | None = None
    session_id: str | None = None
    entries: list['ConversationEntry'] = field(default_factory=list)

    def prompt_preview(self, max_chars: int = 44) -> str:
        return _preview_value(self.user_prompt, max_chars=max_chars) or '(empty prompt)'

    def assistant_preview(self, max_chars: int = 60) -> str:
        if not self.assistant_response.strip():
            return self.assistant_status
        return _preview_value(self.assistant_response, max_chars=max_chars)


@dataclass(frozen=True)
class ConversationHistoryItem:
    turn_id: str
    label: str
    prompt_preview: str
    assistant_preview: str
    status: str


@dataclass
class ConversationThread:
    conversation_id: str
    title: str
    turns: tuple[ConversationTurn, ...] = ()
    session_id: str | None = None
    expanded: bool = True


@dataclass(frozen=True)
class SidebarItem:
    option_id: str
    label: str
    kind: str
    conversation_id: str
    turn_id: str | None = None


@dataclass
class ActivityItem:
    key: str
    label: str
    detail: str = ''
    status: str = 'info'


@dataclass
class ConversationEntry:
    entry_id: str
    kind: str
    title: str
    content: str = ''
    status: str = 'info'
    merge_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _should_hide_session_message(payload: dict[str, object]) -> bool:
    role = str(payload.get('role', ''))
    if role == 'system':
        return True
    content = '' if payload.get('content') is None else str(payload.get('content', ''))
    message_id = payload.get('message_id')
    metadata = payload.get('metadata')
    if isinstance(message_id, str) and (
        message_id.startswith('user_context_')
        or message_id.startswith('plugin_tool_runtime_')
    ):
        return True
    if isinstance(metadata, dict):
        kind = metadata.get('kind')
        if isinstance(kind, str) and kind.endswith('_runtime'):
            return True
    return content.lstrip().startswith('<system-reminder>')


def sanitize_user_prompt_display_text(content: str) -> str:
    return _UI_PROMPT_CONTEXT_PATTERN.sub('', content).strip()


def restore_conversation_turns(
    messages: Sequence[dict[str, object]],
) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    current_turn: ConversationTurn | None = None
    restored_index = 0

    for payload in messages:
        if not isinstance(payload, dict) or _should_hide_session_message(payload):
            continue
        role = str(payload.get('role', ''))
        content = '' if payload.get('content') is None else str(payload.get('content', ''))
        stop_reason = (
            str(payload['stop_reason'])
            if isinstance(payload.get('stop_reason'), str)
            else None
        )
        if role == 'user':
            display_content = sanitize_user_prompt_display_text(content)
            restored_index += 1
            current_turn = ConversationTurn(
                turn_id=f'restored-{restored_index}',
                user_prompt=display_content or '(empty prompt)',
                assistant_status='Restored',
                phase_label='Restored',
                restored=True,
                stop_reason=stop_reason,
            )
            turns.append(current_turn)
            continue
        if role == 'assistant':
            content = sanitize_assistant_display_text(content)
            if current_turn is None:
                restored_index += 1
                current_turn = ConversationTurn(
                    turn_id=f'restored-{restored_index}',
                    user_prompt='(conversation resumed)',
                    assistant_status='Restored',
                    phase_label='Restored',
                    restored=True,
                )
                turns.append(current_turn)
            if content:
                if current_turn.assistant_response:
                    current_turn.assistant_response += '\n\n' + content
                else:
                    current_turn.assistant_response = content
                current_turn.entries.append(
                    ConversationEntry(
                        entry_id=f'{current_turn.turn_id}-assistant-{len(current_turn.entries) + 1}',
                        kind='assistant',
                        title='Assistant',
                        content=content,
                        status='ok',
                    )
                )
            current_turn.assistant_status = 'Ready'
            current_turn.phase_label = 'Restored'
            current_turn.stop_reason = stop_reason or current_turn.stop_reason
            continue
        if role == 'tool' and current_turn is not None:
            current_turn.tool_count += 1
            current_turn.entries.append(
                ConversationEntry(
                    entry_id=f'{current_turn.turn_id}-tool-{len(current_turn.entries) + 1}',
                    kind='tool',
                    title='Tool Result',
                    content=content,
                    status='info',
                )
            )

    return tuple(turns)


def build_conversation_history_items(
    turns: Sequence[ConversationTurn],
) -> tuple[ConversationHistoryItem, ...]:
    items: list[ConversationHistoryItem] = []
    for index, turn in enumerate(turns, start=1):
        prompt_preview = turn.prompt_preview(max_chars=34)
        items.append(
            ConversationHistoryItem(
                turn_id=turn.turn_id,
                label=f'{index:02d}. {prompt_preview}',
                prompt_preview=prompt_preview,
                assistant_preview=turn.assistant_preview(max_chars=56),
                status=turn.assistant_status,
            )
        )
    return tuple(items)
