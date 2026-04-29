from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

from .conversation import ConversationEntry, ConversationThread, ConversationTurn


CONVERSATION_HISTORY_VERSION = 1


@dataclass(frozen=True)
class ConversationHistorySnapshot:
    conversations: tuple[ConversationThread, ...]
    active_conversation_id: str | None = None


def default_claw_code_home() -> Path:
    configured = os.environ.get('CLAW_CODE_HOME')
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()
    return (Path.home() / '.claw-code').resolve()


def workspace_history_key(workspace: Path) -> str:
    resolved = Path(workspace).expanduser().resolve()
    normalized = str(resolved)
    if os.name == 'nt':
        normalized = normalized.lower()
    digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '-', resolved.name).strip('.-')
    if not slug:
        slug = 'workspace'
    return f'{slug}-{digest}'


class ConversationHistoryStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_claw_code_home()).expanduser().resolve()

    @property
    def conversations_dir(self) -> Path:
        return self.root / 'conversations'

    def workspace_path(self, workspace: Path) -> Path:
        return self.conversations_dir / f'{workspace_history_key(workspace)}.json'

    def load_workspace_conversations(
        self,
        workspace: Path,
    ) -> tuple[ConversationThread, ...]:
        return self.load_workspace(workspace).conversations

    def load_workspace(self, workspace: Path) -> ConversationHistorySnapshot:
        path = self.workspace_path(workspace)
        if not path.exists():
            return ConversationHistorySnapshot(())
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return ConversationHistorySnapshot(())
        if not isinstance(payload, dict):
            return ConversationHistorySnapshot(())
        conversations = payload.get('conversations')
        if not isinstance(conversations, list):
            return ConversationHistorySnapshot(())
        restored = [
            _conversation_from_payload(conversation)
            for conversation in conversations
            if isinstance(conversation, dict)
        ]
        active_conversation_id = _string_value(payload.get('active_conversation_id'))
        return ConversationHistorySnapshot(
            tuple(conversation for conversation in restored if conversation is not None),
            active_conversation_id=active_conversation_id,
        )

    def save_workspace_conversations(
        self,
        workspace: Path,
        conversations: Sequence[ConversationThread],
        *,
        active_conversation_id: str | None = None,
    ) -> Path:
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        resolved_workspace = Path(workspace).expanduser().resolve()
        path = self.workspace_path(resolved_workspace)
        payload = {
            'version': CONVERSATION_HISTORY_VERSION,
            'workspace': str(resolved_workspace),
            'workspace_key': workspace_history_key(resolved_workspace),
            'active_conversation_id': active_conversation_id,
            'updated_at': _utc_now(),
            'conversations': [
                _conversation_to_payload(conversation)
                for conversation in conversations
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf-8')
        return path

    def delete_workspace_conversation(
        self,
        workspace: Path,
        conversation_id: str,
        *,
        active_conversation_id: str | None = None,
    ) -> bool:
        conversations = [
            conversation
            for conversation in self.load_workspace_conversations(workspace)
            if conversation.conversation_id != conversation_id
        ]
        path = self.workspace_path(workspace)
        if not conversations:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            except OSError:
                return False
            return True
        self.save_workspace_conversations(
            workspace,
            conversations,
            active_conversation_id=active_conversation_id,
        )
        return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _conversation_to_payload(conversation: ConversationThread) -> dict[str, Any]:
    return {
        'conversation_id': conversation.conversation_id,
        'title': conversation.title,
        'session_id': conversation.session_id,
        'expanded': conversation.expanded,
        'turns': [_turn_to_payload(turn) for turn in conversation.turns],
    }


def _turn_to_payload(turn: ConversationTurn) -> dict[str, Any]:
    return {
        'turn_id': turn.turn_id,
        'user_prompt': turn.user_prompt,
        'assistant_response': turn.assistant_response,
        'assistant_status': turn.assistant_status,
        'phase_label': turn.phase_label,
        'tool_count': turn.tool_count,
        'restored': turn.restored,
        'stop_reason': turn.stop_reason,
        'session_id': turn.session_id,
        'entries': [asdict(entry) for entry in turn.entries],
    }


def _conversation_from_payload(payload: dict[str, Any]) -> ConversationThread | None:
    conversation_id = _string_value(payload.get('conversation_id'))
    if not conversation_id:
        return None
    turns_payload = payload.get('turns')
    turns: list[ConversationTurn] = []
    if isinstance(turns_payload, list):
        turns = [
            turn
            for turn in (
                _turn_from_payload(turn_payload)
                for turn_payload in turns_payload
                if isinstance(turn_payload, dict)
            )
            if turn is not None
        ]
    return ConversationThread(
        conversation_id=conversation_id,
        title=_string_value(payload.get('title')) or 'Conversation',
        turns=tuple(turns),
        session_id=_string_value(payload.get('session_id')),
        expanded=bool(payload.get('expanded', True)),
    )


def _turn_from_payload(payload: dict[str, Any]) -> ConversationTurn | None:
    turn_id = _string_value(payload.get('turn_id'))
    if not turn_id:
        return None
    entries_payload = payload.get('entries')
    entries: list[ConversationEntry] = []
    if isinstance(entries_payload, list):
        entries = [
            entry
            for entry in (
                _entry_from_payload(entry_payload)
                for entry_payload in entries_payload
                if isinstance(entry_payload, dict)
            )
            if entry is not None
        ]
    return ConversationTurn(
        turn_id=turn_id,
        user_prompt=_string_value(payload.get('user_prompt')) or '',
        assistant_response=_string_value(payload.get('assistant_response')) or '',
        assistant_status=_string_value(payload.get('assistant_status')) or 'Restored',
        phase_label=_string_value(payload.get('phase_label')) or 'Restored',
        tool_count=_int_value(payload.get('tool_count')),
        restored=bool(payload.get('restored', True)),
        stop_reason=_string_value(payload.get('stop_reason')),
        session_id=_string_value(payload.get('session_id')),
        entries=entries,
    )


def _entry_from_payload(payload: dict[str, Any]) -> ConversationEntry | None:
    entry_id = _string_value(payload.get('entry_id'))
    kind = _string_value(payload.get('kind'))
    title = _string_value(payload.get('title'))
    if not entry_id or not kind or title is None:
        return None
    return ConversationEntry(
        entry_id=entry_id,
        kind=kind,
        title=title,
        content=_string_value(payload.get('content')) or '',
        status=_string_value(payload.get('status')) or 'info',
        merge_key=_string_value(payload.get('merge_key')),
    )


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
