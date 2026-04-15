from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse
from uuid import uuid4


MCP_PROTOCOL_VERSION = '2025-11-25'


@dataclass(frozen=True)
class MathTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class SageMathBackend:
    def __init__(self) -> None:
        try:
            from sage.all import diff, integral, latex, solve, var  # type: ignore
            from sage.all import sage_eval  # type: ignore
        except Exception as exc:  # pragma: no cover - only relevant outside Sage
            self._import_error = exc
            self.diff = None
            self.integral = None
            self.latex = None
            self.solve = None
            self.var = None
            self.sage_eval = None
            return
        self._import_error = None
        self.diff = diff
        self.integral = integral
        self.latex = latex
        self.solve = solve
        self.var = var
        self.sage_eval = sage_eval

    def _require_backend(self) -> None:
        if self._import_error is not None:
            raise RuntimeError(
                'SageMath backend is not available inside this container'
            ) from self._import_error

    def _symbol_context(self, variables: list[str]) -> dict[str, Any]:
        self._require_backend()
        context: dict[str, Any] = {}
        if not variables:
            variables = ['x']
        unique_names: list[str] = []
        for item in variables:
            text = str(item).strip()
            if text and text not in unique_names:
                unique_names.append(text)
        if not unique_names:
            unique_names = ['x']
        symbols = self.var(' '.join(unique_names))
        if isinstance(symbols, tuple):
            for name, symbol in zip(unique_names, symbols):
                context[name] = symbol
        else:
            context[unique_names[0]] = symbols
        return context

    def _parse_expression(self, expression: str, *, variables: list[str] | None = None) -> Any:
        self._require_backend()
        return self.sage_eval(
            expression,
            locals=self._symbol_context(list(variables or [])),
        )

    def _render_result(self, result: Any) -> dict[str, Any]:
        self._require_backend()
        return {
            'text': str(result),
            'latex': str(self.latex(result)),
        }

    def evaluate_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = _require_string(arguments, 'expression')
        variables = _string_list(arguments.get('variables'))
        numeric = bool(arguments.get('numeric', False))
        expression_value = self._parse_expression(expression, variables=variables)
        result = expression_value.n() if numeric and hasattr(expression_value, 'n') else expression_value
        return {
            'operation': 'evaluate_expression',
            'input': expression,
            **self._render_result(result),
        }

    def simplify_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = _require_string(arguments, 'expression')
        variables = _string_list(arguments.get('variables'))
        result = self._parse_expression(expression, variables=variables).simplify_full()
        return {
            'operation': 'simplify_expression',
            'input': expression,
            **self._render_result(result),
        }

    def differentiate_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = _require_string(arguments, 'expression')
        variable_name = _optional_string(arguments.get('variable')) or 'x'
        order = _coerce_positive_int(arguments.get('order'), default=1)
        result = self.diff(
            self._parse_expression(expression, variables=[variable_name]),
            self._symbol_context([variable_name])[variable_name],
            order,
        )
        return {
            'operation': 'differentiate_expression',
            'input': expression,
            'variable': variable_name,
            'order': order,
            **self._render_result(result),
        }

    def integrate_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expression = _require_string(arguments, 'expression')
        variable_name = _optional_string(arguments.get('variable')) or 'x'
        lower_bound = _optional_string(arguments.get('lower_bound'))
        upper_bound = _optional_string(arguments.get('upper_bound'))
        symbol = self._symbol_context([variable_name])[variable_name]
        parsed_expression = self._parse_expression(expression, variables=[variable_name])
        if lower_bound is not None or upper_bound is not None:
            if lower_bound is None or upper_bound is None:
                raise ValueError('lower_bound and upper_bound must both be provided')
            parsed_lower = self._parse_expression(lower_bound, variables=[variable_name])
            parsed_upper = self._parse_expression(upper_bound, variables=[variable_name])
            result = self.integral(parsed_expression, symbol, parsed_lower, parsed_upper)
        else:
            result = self.integral(parsed_expression, symbol)
        payload = {
            'operation': 'integrate_expression',
            'input': expression,
            'variable': variable_name,
            **self._render_result(result),
        }
        if lower_bound is not None and upper_bound is not None:
            payload['lower_bound'] = lower_bound
            payload['upper_bound'] = upper_bound
        return payload

    def solve_equation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        equation = _require_string(arguments, 'equation')
        variable_name = _optional_string(arguments.get('variable')) or 'x'
        symbol = self._symbol_context([variable_name])[variable_name]
        left_text, right_text = _split_equation(equation)
        lhs = self._parse_expression(left_text, variables=[variable_name])
        rhs = self._parse_expression(right_text, variables=[variable_name])
        solutions = self.solve(lhs == rhs, symbol)
        rendered_solutions = [str(item) for item in solutions]
        return {
            'operation': 'solve_equation',
            'input': equation,
            'variable': variable_name,
            'solutions': rendered_solutions,
            'text': '\n'.join(rendered_solutions) if rendered_solutions else 'No solutions found.',
        }


def _build_tools(backend: SageMathBackend) -> tuple[MathTool, ...]:
    return (
        MathTool(
            name='evaluate_expression',
            description='Evaluate a SageMath expression, optionally numerically.',
            input_schema={
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string'},
                    'variables': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'numeric': {'type': 'boolean'},
                },
                'required': ['expression'],
            },
            handler=backend.evaluate_expression,
        ),
        MathTool(
            name='simplify_expression',
            description='Simplify a symbolic SageMath expression.',
            input_schema={
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string'},
                    'variables': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                },
                'required': ['expression'],
            },
            handler=backend.simplify_expression,
        ),
        MathTool(
            name='differentiate_expression',
            description='Differentiate a symbolic expression with respect to one variable.',
            input_schema={
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string'},
                    'variable': {'type': 'string'},
                    'order': {'type': 'integer', 'minimum': 1},
                },
                'required': ['expression'],
            },
            handler=backend.differentiate_expression,
        ),
        MathTool(
            name='integrate_expression',
            description='Integrate a symbolic expression, either indefinitely or between bounds.',
            input_schema={
                'type': 'object',
                'properties': {
                    'expression': {'type': 'string'},
                    'variable': {'type': 'string'},
                    'lower_bound': {'type': 'string'},
                    'upper_bound': {'type': 'string'},
                },
                'required': ['expression'],
            },
            handler=backend.integrate_expression,
        ),
        MathTool(
            name='solve_equation',
            description='Solve an equation for a given variable.',
            input_schema={
                'type': 'object',
                'properties': {
                    'equation': {'type': 'string'},
                    'variable': {'type': 'string'},
                },
                'required': ['equation'],
            },
            handler=backend.solve_equation,
        ),
    )


def _build_resources() -> tuple[dict[str, Any], ...]:
    return (
        {
            'uri': 'mcp://sagemath/server-info',
            'name': 'SageMath Server Info',
            'mimeType': 'text/plain',
        },
        {
            'uri': 'mcp://sagemath/tool-guide',
            'name': 'SageMath Tool Guide',
            'mimeType': 'text/plain',
        },
    )


def _resource_text(uri: str) -> str:
    if uri == 'mcp://sagemath/server-info':
        return (
            'SageMath MCP server\n'
            '- transport: streamable-http\n'
            '- operations: evaluate, simplify, differentiate, integrate, solve\n'
            '- intended use: trusted local/container traffic for symbolic maths'
        )
    if uri == 'mcp://sagemath/tool-guide':
        return (
            'Suggested usage:\n'
            '- evaluate_expression for direct evaluation\n'
            '- simplify_expression before presenting algebraic results\n'
            '- differentiate_expression for derivatives\n'
            '- integrate_expression for indefinite or definite integrals\n'
            '- solve_equation for equations written with or without "="'
        )
    raise KeyError(uri)


class SageMathMCPHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, SageMathMCPHandler)
        self.sessions: dict[str, dict[str, Any]] = {}
        self.backend = SageMathBackend()
        self.tools = {tool.name: tool for tool in _build_tools(self.backend)}
        self.resources = _build_resources()


class SageMathMCPHandler(BaseHTTPRequestHandler):
    server_version = 'SageMathMCP/1.0'

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/health':
            self._send_json(
                200,
                {'status': 'ok', 'service': 'sagemath-mcp', 'protocol': MCP_PROTOCOL_VERSION},
                content_type='application/json',
            )
            return
        if parsed.path == '/mcp':
            self.send_response(405)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != '/mcp':
            self.send_response(404)
            self.end_headers()
            return
        origin = self.headers.get('Origin')
        if origin and not self._origin_allowed(origin):
            self._send_json(
                403,
                {
                    'jsonrpc': '2.0',
                    'error': {'code': -32000, 'message': 'Origin not allowed'},
                },
            )
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(
                400,
                {
                    'jsonrpc': '2.0',
                    'error': {'code': -32700, 'message': str(exc)},
                },
            )
            return
        method = payload.get('method')
        if not isinstance(method, str) or not method.strip():
            self._send_json(
                400,
                {
                    'jsonrpc': '2.0',
                    'id': payload.get('id'),
                    'error': {'code': -32600, 'message': 'Invalid JSON-RPC request'},
                },
            )
            return
        if method == 'initialize':
            session_id = uuid4().hex
            self.server.sessions[session_id] = {'created_at': uuid4().hex}
            self._send_json(
                200,
                {
                    'jsonrpc': '2.0',
                    'id': payload.get('id'),
                    'result': {
                        'protocolVersion': MCP_PROTOCOL_VERSION,
                        'capabilities': {'resources': {}, 'tools': {}},
                        'serverInfo': {'name': 'sagemath-mcp', 'version': '1.0.0'},
                    },
                },
                headers={'Mcp-Session-Id': session_id},
            )
            return

        session_id = self.headers.get('Mcp-Session-Id')
        if not session_id:
            self._send_json(
                400,
                {
                    'jsonrpc': '2.0',
                    'id': payload.get('id'),
                    'error': {'code': -32001, 'message': 'Missing Mcp-Session-Id header'},
                },
            )
            return
        if session_id not in self.server.sessions:
            self.send_response(404)
            self.end_headers()
            return
        if method == 'notifications/initialized':
            self.send_response(202)
            self.end_headers()
            return
        try:
            result = self._handle_request(method, payload.get('params', {}))
        except KeyError:
            self._send_json(
                404,
                {
                    'jsonrpc': '2.0',
                    'id': payload.get('id'),
                    'error': {'code': -32601, 'message': f'Method not found: {method}'},
                },
            )
            return
        except Exception as exc:
            self._send_json(
                200,
                {
                    'jsonrpc': '2.0',
                    'id': payload.get('id'),
                    'error': {'code': -32000, 'message': str(exc)},
                },
            )
            return
        self._send_json(
            200,
            {
                'jsonrpc': '2.0',
                'id': payload.get('id'),
                'result': result,
            },
        )

    def _handle_request(self, method: str, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError('params must be a JSON object')
        if method == 'resources/list':
            return {'resources': list(self.server.resources)}
        if method == 'resources/read':
            uri = _require_string(params, 'uri')
            return {
                'contents': [
                    {
                        'uri': uri,
                        'mimeType': 'text/plain',
                        'text': _resource_text(uri),
                    }
                ]
            }
        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': tool.name,
                        'description': tool.description,
                        'inputSchema': tool.input_schema,
                    }
                    for tool in self.server.tools.values()
                ]
            }
        if method == 'tools/call':
            name = _require_string(params, 'name')
            tool = self.server.tools.get(name)
            if tool is None:
                raise ValueError(f'Unknown tool: {name}')
            arguments = params.get('arguments', {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError('arguments must be a JSON object')
            structured = tool.handler(arguments)
            return {
                'content': [
                    {
                        'type': 'text',
                        'text': structured.get('text', json.dumps(structured, ensure_ascii=True)),
                    }
                ],
                'structuredContent': structured,
                'isError': False,
            }
        raise KeyError(method)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Request body must be valid JSON') from exc
        if not isinstance(payload, dict):
            raise ValueError('Request body must be a JSON object')
        return payload

    def _origin_allowed(self, origin: str) -> bool:
        parsed = urlparse(origin)
        host = (parsed.hostname or '').lower()
        return host in {'127.0.0.1', 'localhost', 'sagemath'}

    def _send_json(
        self,
        status_code: int,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        content_type: str = 'application/json',
    ) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be a non-empty string')
    return value.strip()


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value >= 1:
        return value
    return default


def _split_equation(equation: str) -> tuple[str, str]:
    if '=' not in equation:
        return equation, '0'
    left, right = equation.split('=', 1)
    return left.strip(), right.strip()


def main() -> int:
    host = os.environ.get('SAGEMATH_MCP_HOST', '0.0.0.0')
    port = int(os.environ.get('SAGEMATH_MCP_PORT', '8000'))
    server = SageMathMCPHTTPServer((host, port))
    print(f'SageMath MCP server listening on http://{host}:{port}/mcp', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
