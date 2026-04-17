from __future__ import annotations

import json
from typing import Any, Iterator
from urllib import error, request

from src.agent.agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
)
from src.llm.parsers import (
    LLMBackendError,
    build_response_format as _build_response_format,
    iter_stream_events,
    join_url as _join_url,
    normalize_content as _normalize_content,
    optional_int as _optional_int,
    parse_tool_arguments as _parse_tool_arguments,
    parse_tool_calls_from_message,
    parse_usage as _parse_usage,
)


OpenAICompatError = LLMBackendError


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client for local model servers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        payload = self._request_json(
            self._build_payload(
                messages=messages,
                tools=tools,
                stream=False,
                output_schema=output_schema,
            )
        )
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatError('Local model backend returned no choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatError('Local model backend returned malformed choice data')

        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise OpenAICompatError('Local model backend returned no assistant message')

        content = _normalize_content(message.get('content'))
        tool_calls = self._parse_tool_calls_from_message(message)

        finish_reason = first_choice.get('finish_reason')
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        return AssistantTurn(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            raw_message=message,
            usage=_parse_usage(payload.get('usage')),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=True,
            output_schema=output_schema,
        )
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                yield StreamEvent(type='message_start')
                for event_payload in self._iter_sse_payloads(response):
                    yield from self._parse_stream_payload(event_payload)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(
                f'HTTP {exc.code} from local model backend: {detail}'
            ) from exc
        except error.URLError as exc:
            raise OpenAICompatError(
                f'Unable to reach local model backend at {self.config.base_url}: {exc.reason}'
            ) from exc

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode('utf-8')
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=body,
            headers={
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(
                f'HTTP {exc.code} from local model backend: {detail}'
            ) from exc
        except error.URLError as exc:
            raise OpenAICompatError(
                f'Unable to reach local model backend at {self.config.base_url}: {exc.reason}'
            ) from exc

        try:
            payload = json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise OpenAICompatError('Local model backend returned invalid JSON') from exc
        if not isinstance(payload, dict):
            raise OpenAICompatError('Local model backend returned malformed JSON payload')
        return payload

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool,
        output_schema: OutputSchemaConfig | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            'model': self.config.model,
            'messages': messages,
            'tools': tools,
            'tool_choice': 'auto',
            'temperature': self.config.temperature,
            'stream': stream,
        }
        if stream:
            payload['stream_options'] = {'include_usage': True}
        response_format = _build_response_format(output_schema)
        if response_format is not None:
            payload['response_format'] = response_format
        return payload

    def _parse_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        return parse_tool_calls_from_message(
            message,
            strict=True,
            include_legacy_function_call=True,
        )

    def _iter_sse_payloads(self, response: Any) -> Iterator[dict[str, Any]]:
        buffer: list[str] = []
        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode('utf-8', errors='replace')
            else:
                text = str(line)
            stripped = text.strip()
            if not stripped:
                if not buffer:
                    continue
                joined = '\n'.join(buffer)
                buffer.clear()
                if joined == '[DONE]':
                    break
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed SSE payload from model backend')
                yield payload
                continue
            if stripped.startswith('data:'):
                buffer.append(stripped[5:].strip())

        if buffer:
            joined = '\n'.join(buffer)
            if joined != '[DONE]':
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid trailing JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed trailing SSE payload from model backend')
                yield payload

    def _parse_stream_payload(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        yield from iter_stream_events(payload, normalize_list_content=False)
