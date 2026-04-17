from __future__ import annotations

import json
from typing import Any


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

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        return queued.pop(0)

    return _fake_urlopen


def make_recording_urlopen_side_effect(
    responses: list[dict[str, object]],
    recorded_payloads: list[dict[str, object]],
):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        recorded_payloads.append(_decode_request_payload(request_obj))
        return queued.pop(0)

    return _fake_urlopen


def make_streaming_urlopen_side_effect(
    responses: list[list[dict[str, object]]],
):
    queued = [FakeStreamingHTTPResponse(payloads) for payloads in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        return queued.pop(0)

    return _fake_urlopen


def make_recording_streaming_urlopen_side_effect(
    responses: list[list[dict[str, object]]],
    recorded_payloads: list[dict[str, object]],
):
    queued = [FakeStreamingHTTPResponse(payloads) for payloads in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        recorded_payloads.append(_decode_request_payload(request_obj))
        return queued.pop(0)

    return _fake_urlopen


__all__ = [
    'FakeHTTPResponse',
    'FakeStreamingHTTPResponse',
    'make_recording_streaming_urlopen_side_effect',
    'make_recording_urlopen_side_effect',
    'make_streaming_urlopen_side_effect',
    'make_urlopen_side_effect',
]
