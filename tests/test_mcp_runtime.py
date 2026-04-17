from __future__ import annotations

import json
import os
import threading
import sys
import tempfile
import unittest
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src.agent.agent_runtime import LocalCodingAgent
from src.agent.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent.agent_types import AgentRuntimeConfig, ModelConfig
from src.features.integration.mcp_runtime import MCPRuntime
from tests.test_helpers import make_urlopen_side_effect


class MCPRuntimeTests(unittest.TestCase):
    def _write_fake_stdio_server(self, workspace: Path) -> Path:
        server_path = workspace / 'fake_mcp_server.py'
        server_path.write_text(
            (
                'import json, sys\n'
                'RESOURCES = [{"uri": "mcp://remote/notes", "name": "Remote Notes", "mimeType": "text/plain"}]\n'
                'TOOLS = [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]\n'
                'for raw in sys.stdin:\n'
                '    raw = raw.strip()\n'
                '    if not raw:\n'
                '        continue\n'
                '    message = json.loads(raw)\n'
                '    method = message.get("method")\n'
                '    if method == "initialize":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"protocolVersion": "2025-11-25", "capabilities": {"resources": {}, "tools": {}}, "serverInfo": {"name": "fake-remote", "version": "1.0.0"}}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "notifications/initialized":\n'
                '        continue\n'
                '    if method == "resources/list":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"resources": RESOURCES}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "resources/read":\n'
                '        uri = message.get("params", {}).get("uri")\n'
                '        text = "remote notes via stdio" if uri == "mcp://remote/notes" else "unknown resource"\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "tools/list":\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"tools": TOOLS}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    if method == "tools/call":\n'
                '        params = message.get("params", {})\n'
                '        text = params.get("arguments", {}).get("text", "")\n'
                '        response = {"jsonrpc": "2.0", "id": message.get("id"), "result": {"content": [{"type": "text", "text": "echo:" + text}], "isError": False}}\n'
                '        print(json.dumps(response), flush=True)\n'
                '        continue\n'
                '    response = {"jsonrpc": "2.0", "id": message.get("id"), "error": {"code": -32601, "message": "Method not found"}}\n'
                '    print(json.dumps(response), flush=True)\n'
            ),
            encoding='utf-8',
        )
        return server_path

    def _start_fake_streamable_http_server(self, *, mode: str = 'json'):
        class ServerState:
            def __init__(self) -> None:
                self.mode = mode
                self.session_id = 'session-http-123'
                self.request_log: list[dict[str, str | None]] = []

        state = ServerState()

        class Handler(BaseHTTPRequestHandler):
            server_version = 'FakeMCPHTTP/1.0'

            def log_message(self, format, *args):  # noqa: A003, ANN001
                return

            def do_GET(self) -> None:
                self.send_response(405)
                self.end_headers()

            def do_POST(self) -> None:
                if self.path != '/mcp':
                    self.send_response(404)
                    self.end_headers()
                    return
                content_length = int(self.headers.get('Content-Length', '0'))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode('utf-8'))
                method = payload.get('method')
                state.request_log.append(
                    {
                        'method': str(method) if method is not None else None,
                        'session': self.headers.get('Mcp-Session-Id'),
                        'protocol': self.headers.get('MCP-Protocol-Version'),
                    }
                )
                if method == 'initialize':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Mcp-Session-Id', state.session_id)
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                'jsonrpc': '2.0',
                                'id': payload.get('id'),
                                'result': {
                                    'protocolVersion': '2025-11-25',
                                    'capabilities': {'resources': {}, 'tools': {}},
                                    'serverInfo': {'name': 'fake-http', 'version': '1.0.0'},
                                },
                            }
                        ).encode('utf-8')
                    )
                    return
                expected_session = state.session_id
                if self.headers.get('Mcp-Session-Id') != expected_session:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'missing session header')
                    return
                if self.headers.get('MCP-Protocol-Version') != '2025-11-25':
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'missing protocol version header')
                    return
                if method == 'notifications/initialized':
                    self.send_response(202)
                    self.end_headers()
                    return
                if method == 'resources/list':
                    response_payload = {
                        'jsonrpc': '2.0',
                        'id': payload.get('id'),
                        'result': {
                            'resources': [
                                {
                                    'uri': 'mcp://http/notes',
                                    'name': 'HTTP Notes',
                                    'mimeType': 'text/plain',
                                }
                            ]
                        },
                    }
                    self._write_transport_response(response_payload)
                    return
                if method == 'resources/read':
                    uri = payload.get('params', {}).get('uri')
                    response_payload = {
                        'jsonrpc': '2.0',
                        'id': payload.get('id'),
                        'result': {
                            'contents': [
                                {
                                    'uri': uri,
                                    'mimeType': 'text/plain',
                                    'text': 'remote notes via http',
                                }
                            ]
                        },
                    }
                    self._write_transport_response(response_payload)
                    return
                if method == 'tools/list':
                    response_payload = {
                        'jsonrpc': '2.0',
                        'id': payload.get('id'),
                        'result': {
                            'tools': [
                                {
                                    'name': 'echo',
                                    'description': 'Echo text over HTTP',
                                    'inputSchema': {
                                        'type': 'object',
                                        'properties': {'text': {'type': 'string'}},
                                    },
                                }
                            ]
                        },
                    }
                    self._write_transport_response(response_payload)
                    return
                if method == 'tools/call':
                    text = payload.get('params', {}).get('arguments', {}).get('text', '')
                    response_payload = {
                        'jsonrpc': '2.0',
                        'id': payload.get('id'),
                        'result': {
                            'content': [{'type': 'text', 'text': 'echo:' + str(text)}],
                            'isError': False,
                        },
                    }
                    self._write_transport_response(response_payload)
                    return
                self.send_response(404)
                self.end_headers()

            def _write_transport_response(self, response_payload: dict[str, object]) -> None:
                if state.mode == 'sse':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    self.wfile.write(b'id: evt-1\ndata:\n\n')
                    self.wfile.write(
                        (
                            'event: message\n'
                            f'data: {json.dumps(response_payload)}\n\n'
                        ).encode('utf-8')
                    )
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/mcp'
        return server, thread, url, state

    def test_runtime_discovers_and_reads_local_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"},'
                    '{"uri":"mcp://workspace/inline","name":"Inline","text":"inline body"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            self.assertEqual(len(runtime.resources), 2)
            self.assertIn('Local MCP resources: 2', runtime.render_summary())
            self.assertEqual(runtime.read_resource('mcp://workspace/inline'), 'inline body')
            self.assertIn('mcp notes', runtime.read_resource('mcp://workspace/notes'))

    def test_runtime_discovers_stdio_server_and_remote_resources_and_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            resources = runtime.list_resources()
            tools = runtime.list_tools()
            self.assertEqual(len(runtime.servers), 1)
            self.assertTrue(runtime.has_transport_servers())
            self.assertIn('Configured MCP servers: 1', runtime.render_summary())
            self.assertEqual(len(resources), 1)
            self.assertEqual(resources[0].uri, 'mcp://remote/notes')
            self.assertIn('remote notes via stdio', runtime.read_resource('mcp://remote/notes'))
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, 'echo')
            rendered, metadata = runtime.call_tool('echo', arguments={'text': 'hello'})
            self.assertIn('echo:hello', rendered)
            self.assertEqual(metadata.get('server_name'), 'remote')

    def test_runtime_discovers_streamable_http_server_and_remote_resources_and_tools(self) -> None:
        server, thread, url, state = self._start_fake_streamable_http_server()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                workspace = Path(tmp_dir)
                (workspace / '.claw-mcp.json').write_text(
                    json.dumps(
                        {
                            'mcpServers': {
                                'remote-http': {
                                    'transport': 'streamable-http',
                                    'url': url,
                                }
                            }
                        }
                    ),
                    encoding='utf-8',
                )
                runtime = MCPRuntime.from_workspace(workspace)
                resources = runtime.list_resources()
                tools = runtime.list_tools()
                rendered, metadata = runtime.call_tool('echo', arguments={'text': 'hello'})
                self.assertEqual(len(runtime.servers), 1)
                self.assertTrue(runtime.has_transport_servers())
                self.assertIn('streamable-http: 1 server(s)', runtime.render_summary())
                self.assertEqual(len(resources), 1)
                self.assertEqual(resources[0].uri, 'mcp://http/notes')
                self.assertIn('remote notes via http', runtime.read_resource('mcp://http/notes'))
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0].name, 'echo')
                self.assertIn('echo:hello', rendered)
                self.assertEqual(metadata.get('server_name'), 'remote-http')
                methods = [entry['method'] for entry in state.request_log]
                self.assertEqual(methods.count('initialize'), 1)
                self.assertIn('notifications/initialized', methods)
                for entry in state.request_log[1:]:
                    self.assertEqual(entry['session'], state.session_id)
                    self.assertEqual(entry['protocol'], '2025-11-25')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_runtime_supports_streamable_http_sse_responses(self) -> None:
        server, thread, url, _state = self._start_fake_streamable_http_server(mode='sse')
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                workspace = Path(tmp_dir)
                (workspace / '.claw-mcp.json').write_text(
                    json.dumps(
                        {
                            'mcpServers': {
                                'remote-http': {
                                    'transport': 'http',
                                    'url': url,
                                }
                            }
                        }
                    ),
                    encoding='utf-8',
                )
                runtime = MCPRuntime.from_workspace(workspace)
                tools = runtime.list_tools()
                rendered, metadata = runtime.call_tool('echo', arguments={'text': 'sse'})
                self.assertEqual(len(tools), 1)
                self.assertEqual(tools[0].name, 'echo')
                self.assertIn('echo:sse', rendered)
                self.assertEqual(metadata.get('server_name'), 'remote-http')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_streamable_http_server_url_can_expand_from_environment(self) -> None:
        server, thread, url, _state = self._start_fake_streamable_http_server()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                workspace = Path(tmp_dir)
                manifest_path = workspace / '.claw-mcp.json'
                manifest_path.write_text(
                    json.dumps(
                        {
                            'mcpServers': {
                                'remote-http': {
                                    'transport': 'streamable-http',
                                    'url': '${MCP_TEST_URL}',
                                }
                            }
                        }
                    ),
                    encoding='utf-8',
                )
                with patch.dict(os.environ, {'MCP_TEST_URL': url}, clear=False):
                    runtime = MCPRuntime.from_workspace(workspace)
                    self.assertEqual(len(runtime.servers), 1)
                    self.assertEqual(runtime.servers[0].url, url)
                    self.assertEqual(len(runtime.list_tools()), 1)
                with patch.dict(os.environ, {}, clear=True):
                    runtime = MCPRuntime.from_workspace(workspace)
                    self.assertEqual(len(runtime.servers), 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_mcp_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                mcp_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'mcp_list_resources',
                {},
                context,
            )
            read_result = execute_tool(
                default_tool_registry(),
                'mcp_read_resource',
                {'uri': 'mcp://workspace/notes'},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('mcp://workspace/notes', list_result.content)
        self.assertTrue(read_result.ok)
        self.assertIn('mcp notes', read_result.content)

    def test_mcp_transport_tools_execute_against_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                mcp_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'mcp_list_tools',
                {},
                context,
            )
            call_result = execute_tool(
                default_tool_registry(),
                'mcp_call_tool',
                {'tool_name': 'echo', 'arguments': {'text': 'tool-run'}},
                context,
            )

        self.assertTrue(list_result.ok)
        self.assertIn('echo', list_result.content)
        self.assertTrue(call_result.ok)
        self.assertIn('echo:tool-run', call_result.content)
        self.assertEqual(call_result.metadata.get('action'), 'mcp_call_tool')

    def test_mcp_list_tools_reports_unknown_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            runtime = MCPRuntime.from_workspace(workspace)
            context = build_tool_context(
                AgentRuntimeConfig(cwd=workspace),
                mcp_runtime=runtime,
            )
            list_result = execute_tool(
                default_tool_registry(),
                'mcp_list_tools',
                {'server': 'missing'},
                context,
            )

        self.assertFalse(list_result.ok)
        self.assertIn('Unknown MCP server: missing', list_result.content)
        self.assertIn('Unknown MCP server: missing', runtime.render_tool_index(server_name='missing'))

    def test_mcp_cli_subprocess_smoke_works_with_stdio_server(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )

            commands = [
                (
                    [
                        sys.executable,
                        '-m',
                        'src.main',
                        'mcp-status',
                        '--cwd',
                        str(workspace),
                    ],
                    ('Configured MCP servers: 1', 'stdio: 1 server(s)'),
                ),
                (
                    [
                        sys.executable,
                        '-m',
                        'src.main',
                        'mcp-tools',
                        '--cwd',
                        str(workspace),
                    ],
                    ('echo ; server=remote',),
                ),
                (
                    [
                        sys.executable,
                        '-m',
                        'src.main',
                        'mcp-call-tool',
                        'echo',
                        '--arguments-json',
                        '{"text":"hello"}',
                        '--cwd',
                        str(workspace),
                    ],
                    ('echo:hello',),
                ),
            ]

            for command, expected_snippets in commands:
                try:
                    result = subprocess.run(
                        command,
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                except subprocess.TimeoutExpired as exc:
                    self.fail(
                        f'Subprocess timed out for {command!r}\n'
                        f'stdout:\n{exc.stdout or ""}\n'
                        f'stderr:\n{exc.stderr or ""}'
                    )
                if result.returncode != 0:
                    self.fail(
                        f'Subprocess failed for {command!r}\n'
                        f'stdout:\n{result.stdout}\n'
                        f'stderr:\n{result.stderr}'
                    )
                for snippet in expected_snippets:
                    self.assertIn(snippet, result.stdout)

    def test_mcp_cli_subprocess_smoke_works_with_streamable_http_server(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        server, thread, url, _state = self._start_fake_streamable_http_server()
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                workspace = Path(tmp_dir)
                (workspace / '.claw-mcp.json').write_text(
                    json.dumps(
                        {
                            'mcpServers': {
                                'remote-http': {
                                    'transport': 'streamable-http',
                                    'url': url,
                                }
                            }
                        }
                    ),
                    encoding='utf-8',
                )

                commands = [
                    (
                        [
                            sys.executable,
                            '-m',
                            'src.main',
                            'mcp-status',
                            '--cwd',
                            str(workspace),
                        ],
                        ('Configured MCP servers: 1', 'streamable-http: 1 server(s)'),
                    ),
                    (
                        [
                            sys.executable,
                            '-m',
                            'src.main',
                            'mcp-tools',
                            '--cwd',
                            str(workspace),
                        ],
                        ('echo ; server=remote-http',),
                    ),
                    (
                        [
                            sys.executable,
                            '-m',
                            'src.main',
                            'mcp-call-tool',
                            'echo',
                            '--server',
                            'remote-http',
                            '--arguments-json',
                            '{"text":"hello-http"}',
                            '--cwd',
                            str(workspace),
                        ],
                        ('echo:hello-http',),
                    ),
                ]

                for command, expected_snippets in commands:
                    try:
                        result = subprocess.run(
                            command,
                            cwd=repo_root,
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                    except subprocess.TimeoutExpired as exc:
                        self.fail(
                            f'Subprocess timed out for {command!r}\n'
                            f'stdout:\n{exc.stdout or ""}\n'
                            f'stderr:\n{exc.stderr or ""}'
                        )
                    if result.returncode != 0:
                        self.fail(
                            f'Subprocess failed for {command!r}\n'
                            f'stdout:\n{result.stdout}\n'
                            f'stderr:\n{result.stderr}'
                        )
                    for snippet in expected_snippets:
                        self.assertIn(snippet, result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_agent_can_use_mcp_tools_in_model_loop(self) -> None:
        responses = [
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'I will inspect the MCP resource.',
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'mcp_read_resource',
                                        'arguments': '{"uri": "mcp://workspace/notes"}',
                                    },
                                }
                            ],
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': {'prompt_tokens': 8, 'completion_tokens': 3},
            },
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'The MCP resource says mcp notes.',
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 6, 'completion_tokens': 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'notes.txt').write_text('mcp notes\n', encoding='utf-8')
            (workspace / '.claw-mcp.json').write_text(
                (
                    '{"servers":[{"name":"workspace","resources":['
                    '{"uri":"mcp://workspace/notes","name":"Notes","path":"notes.txt"}'
                    ']}]}'
                ),
                encoding='utf-8',
            )
            with patch('src.openai_compat.request.urlopen', side_effect=make_urlopen_side_effect(responses)):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='Qwen/Qwen3-Coder-30B-A3B-Instruct',
                        base_url='http://127.0.0.1:8000/v1',
                    ),
                    runtime_config=AgentRuntimeConfig(cwd=workspace),
                )
                result = agent.run('Read the MCP notes resource')

        self.assertEqual(result.final_output, 'The MCP resource says mcp notes.')
        self.assertEqual(result.tool_calls, 1)
        tool_message = next(
            message
            for message in result.transcript
            if message.get('role') == 'tool'
        )
        self.assertIn('mcp notes', tool_message.get('content', ''))

    def test_agent_can_use_transport_backed_mcp_call_tool_in_model_loop(self) -> None:
        responses = [
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'I will call the remote MCP tool.',
                            'tool_calls': [
                                {
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'mcp_call_tool',
                                        'arguments': '{"tool_name": "echo", "server": "remote", "arguments": {"text": "agent-call"}}',
                                    },
                                }
                            ],
                        },
                        'finish_reason': 'tool_calls',
                    }
                ],
                'usage': {'prompt_tokens': 8, 'completion_tokens': 3},
            },
            {
                'choices': [
                    {
                        'message': {
                            'role': 'assistant',
                            'content': 'The remote MCP tool replied with echo:agent-call.',
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {'prompt_tokens': 6, 'completion_tokens': 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            server_path = self._write_fake_stdio_server(workspace)
            (workspace / '.claw-mcp.json').write_text(
                json.dumps(
                    {
                        'mcpServers': {
                            'remote': {
                                'command': sys.executable,
                                'args': ['-u', str(server_path)],
                            }
                        }
                    }
                ),
                encoding='utf-8',
            )
            with patch('src.openai_compat.request.urlopen', side_effect=make_urlopen_side_effect(responses)):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(
                        model='Qwen/Qwen3-Coder-30B-A3B-Instruct',
                        base_url='http://127.0.0.1:8000/v1',
                    ),
                    runtime_config=AgentRuntimeConfig(cwd=workspace),
                )
                result = agent.run('Call the remote MCP echo tool')

        self.assertEqual(result.final_output, 'The remote MCP tool replied with echo:agent-call.')
        self.assertEqual(result.tool_calls, 1)
        tool_message = next(
            message
            for message in result.transcript
            if message.get('role') == 'tool'
        )
        self.assertIn('echo:agent-call', tool_message.get('content', ''))
