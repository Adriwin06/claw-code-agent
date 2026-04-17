from __future__ import annotations

import json
from typing import Any

from src.agent.agent_types import AssistantTurn, OutputSchemaConfig, StreamEvent
from src.llm.parsers import (
    build_response_format,
    coerce_finish_reason,
    iter_stream_events,
    normalize_content,
    parse_tool_calls_from_message,
    parse_usage,
)


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def read(self) -> bytes:
        if isinstance(self.payload, (dict, list)):
            encoded = json.dumps(self.payload)
        else:
            encoded = str(self.payload)
        return encoded.encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeStreamingHTTPResponse:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.lines: list[bytes] = []
        for payload in payloads:
            chunk = f'data: {json.dumps(payload)}\n\n'
            self.lines.extend(
                part.encode('utf-8') for part in chunk.splitlines(keepends=True)
            )
        done_chunk = 'data: [DONE]\n\n'
        self.lines.extend(part.encode('utf-8') for part in done_chunk.splitlines(keepends=True))

    def readline(self) -> bytes:
        if not self.lines:
            return b''
        return self.lines.pop(0)

    def __enter__(self) -> 'FakeStreamingHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ScriptedLLMClient:
    """Minimal scripted client implementing LocalCodingAgent's complete/stream interface."""

    def __init__(
        self,
        *,
        completion_responses: list[dict[str, object]] | None = None,
        stream_responses: list[list[dict[str, object]]] | None = None,
        recorded_payloads: list[dict[str, object]] | None = None,
    ) -> None:
        self._completion_responses = list(completion_responses or [])
        self._stream_responses = [list(item) for item in (stream_responses or [])]
        self._recorded_payloads = recorded_payloads

    def _record_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: OutputSchemaConfig | None,
        stream: bool,
    ) -> None:
        if self._recorded_payloads is None:
            return
        payload: dict[str, object] = {
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'stream': stream,
        }
        response_format = build_response_format(output_schema)
        if response_format is not None:
            payload['response_format'] = response_format
        self._recorded_payloads.append(payload)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        self._record_request(
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            stream=False,
        )
        if not self._completion_responses:
            raise AssertionError('No scripted completion response is available')
        payload = self._completion_responses.pop(0)
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise AssertionError('Scripted completion payload is missing choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise AssertionError('Scripted completion choice must be a dict')
        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise AssertionError('Scripted completion choice must include a message dict')
        return AssistantTurn(
            content=normalize_content(message.get('content')),
            tool_calls=tuple(
                parse_tool_calls_from_message(
                    message,
                    strict=True,
                    include_legacy_function_call=True,
                )
            ),
            finish_reason=coerce_finish_reason(first_choice.get('finish_reason')),
            raw_message=message,
            usage=parse_usage(payload.get('usage')),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ):
        self._record_request(
            messages=messages,
            tools=tools,
            output_schema=output_schema,
            stream=True,
        )
        if not self._stream_responses:
            raise AssertionError('No scripted streaming response is available')
        payloads = self._stream_responses.pop(0)
        yield StreamEvent(type='message_start')
        for payload in payloads:
            if not isinstance(payload, dict):
                raise AssertionError('Scripted streaming payload entries must be dicts')
            yield from iter_stream_events(payload, normalize_list_content=True)


def _is_http_request_obj(value: object) -> bool:
    return hasattr(value, 'data') or hasattr(value, 'full_url')


def _decode_request_payload(request_obj: Any) -> dict[str, object]:
    raw = getattr(request_obj, 'data', None)
    if isinstance(raw, bytes):
        text = raw.decode('utf-8', errors='replace')
    elif isinstance(raw, str):
        text = raw
    else:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def make_urlopen_side_effect(responses: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]
    scripted_client = ScriptedLLMClient(completion_responses=responses)

    def _fake_urlopen(request_obj, timeout=None, **kwargs):  # noqa: ANN001
        if _is_http_request_obj(request_obj):
            return queued.pop(0)
        return scripted_client

    return _fake_urlopen


def make_recording_urlopen_side_effect(
    responses: list[dict[str, object]],
    recorded_payloads: list[dict[str, object]],
):
    queued = [FakeHTTPResponse(payload) for payload in responses]
    scripted_client = ScriptedLLMClient(
        completion_responses=responses,
        recorded_payloads=recorded_payloads,
    )

    def _fake_urlopen(request_obj, timeout=None, **kwargs):  # noqa: ANN001
        if _is_http_request_obj(request_obj):
            recorded_payloads.append(_decode_request_payload(request_obj))
            return queued.pop(0)
        return scripted_client

    return _fake_urlopen


def make_streaming_urlopen_side_effect(
    responses: list[list[dict[str, object]]],
):
    queued = [FakeStreamingHTTPResponse(payloads) for payloads in responses]
    scripted_client = ScriptedLLMClient(stream_responses=responses)

    def _fake_urlopen(request_obj, timeout=None, **kwargs):  # noqa: ANN001
        if _is_http_request_obj(request_obj):
            return queued.pop(0)
        return scripted_client

    return _fake_urlopen


def make_recording_streaming_urlopen_side_effect(
    responses: list[list[dict[str, object]]],
    recorded_payloads: list[dict[str, object]],
):
    queued = [FakeStreamingHTTPResponse(payloads) for payloads in responses]
    scripted_client = ScriptedLLMClient(
        stream_responses=responses,
        recorded_payloads=recorded_payloads,
    )

    def _fake_urlopen(request_obj, timeout=None, **kwargs):  # noqa: ANN001
        if _is_http_request_obj(request_obj):
            recorded_payloads.append(_decode_request_payload(request_obj))
            return queued.pop(0)
        return scripted_client

    return _fake_urlopen


def make_build_llm_client_side_effect(responses: list[dict[str, object]]):
    scripted_client = ScriptedLLMClient(completion_responses=responses)

    def _fake_builder(*args, **kwargs):  # noqa: ANN001
        return scripted_client

    return _fake_builder


def make_recording_build_llm_client_side_effect(
    responses: list[dict[str, object]],
    recorded_payloads: list[dict[str, object]],
):
    scripted_client = ScriptedLLMClient(
        completion_responses=responses,
        recorded_payloads=recorded_payloads,
    )

    def _fake_builder(*args, **kwargs):  # noqa: ANN001
        return scripted_client

    return _fake_builder


def make_streaming_build_llm_client_side_effect(
    responses: list[list[dict[str, object]]],
):
    scripted_client = ScriptedLLMClient(stream_responses=responses)

    def _fake_builder(*args, **kwargs):  # noqa: ANN001
        return scripted_client

    return _fake_builder


def make_recording_streaming_build_llm_client_side_effect(
    responses: list[list[dict[str, object]]],
    recorded_payloads: list[dict[str, object]],
):
    scripted_client = ScriptedLLMClient(
        stream_responses=responses,
        recorded_payloads=recorded_payloads,
    )

    def _fake_builder(*args, **kwargs):  # noqa: ANN001
        return scripted_client

    return _fake_builder


def make_urlopen_side_effect_legacy(responses: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        return queued.pop(0)

    return _fake_urlopen


__all__ = [
    'FakeHTTPResponse',
    'FakeStreamingHTTPResponse',
    'ScriptedLLMClient',
    'make_build_llm_client_side_effect',
    'make_recording_build_llm_client_side_effect',
    'make_recording_streaming_urlopen_side_effect',
    'make_recording_streaming_build_llm_client_side_effect',
    'make_recording_urlopen_side_effect',
    'make_streaming_build_llm_client_side_effect',
    'make_streaming_urlopen_side_effect',
    'make_urlopen_side_effect_legacy',
    'make_urlopen_side_effect',
]
