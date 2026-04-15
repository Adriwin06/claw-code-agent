from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRunResult, AgentRuntimeConfig, ModelConfig, UsageStats
from src.textual_ui import (
    AgentTuiEventBridge,
    AgentTuiState,
    build_slash_command_suggestions,
    extract_slash_command_query,
    filter_slash_command_suggestions,
    render_slash_command_suggestion_detail,
)


class TextualUiTests(unittest.TestCase):
    def test_extract_slash_command_query_only_matches_active_command_token(self) -> None:
        self.assertEqual(extract_slash_command_query('/con'), 'con')
        self.assertEqual(extract_slash_command_query('   /config'), 'config')
        self.assertEqual(extract_slash_command_query('/'), '')
        self.assertIsNone(extract_slash_command_query('plain prompt'))
        self.assertIsNone(extract_slash_command_query('/context now'))
        self.assertIsNone(extract_slash_command_query('/context '))

    def test_filter_slash_command_suggestions_matches_prefix_and_alias(self) -> None:
        suggestions = build_slash_command_suggestions()

        prefix_matches = filter_slash_command_suggestions('/con', suggestions=suggestions)
        alias_matches = filter_slash_command_suggestions('/usage', suggestions=suggestions)

        self.assertTrue(any(item.primary_name == 'context' for item in prefix_matches))
        self.assertTrue(any(item.primary_name == 'config' for item in prefix_matches))
        self.assertEqual(alias_matches[0].primary_name, 'context')

    def test_render_slash_command_suggestion_detail_includes_description_and_aliases(self) -> None:
        suggestion = next(
            item for item in build_slash_command_suggestions() if item.primary_name == 'context'
        )
        rendered = render_slash_command_suggestion_detail(suggestion)

        self.assertIn('/context', rendered)
        self.assertIn('Show estimated session context usage', rendered)
        self.assertIn('/usage', rendered)
        self.assertIn('Tab: insert command', rendered)

    def test_state_from_agent_renders_core_runtime_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='demo-model'),
                runtime_config=AgentRuntimeConfig(
                    cwd=Path(tmp_dir),
                    permissions=AgentPermissions(
                        allow_file_write=True,
                        allow_shell_commands=True,
                    ),
                ),
            )

        state = AgentTuiState.from_agent(agent)
        rendered = state.render()

        self.assertIn('workspace=', rendered)
        self.assertIn('model=demo-model', rendered)
        self.assertIn('permissions=write, shell', rendered)
        self.assertIn('status=Idle', rendered)

    def test_event_bridge_streams_updates_and_final_usage(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Inspect repo')
        bridge.handle_event({'type': 'message_start'})
        bridge.handle_event({'type': 'content_delta', 'delta': 'Hello '})
        bridge.handle_event({'type': 'content_delta', 'delta': 'world'})
        bridge.handle_event({'type': 'message_stop', 'finish_reason': 'tool_calls'})
        bridge.handle_event(
            {
                'type': 'tool_start',
                'tool_name': 'bash',
                'arguments': {'command': 'pwd'},
            }
        )
        bridge.handle_event(
            {
                'type': 'tool_delta',
                'tool_name': 'bash',
                'tool_call_id': 'call-1',
                'stream': 'stdout',
                'delta': '/workspace\n',
            }
        )
        bridge.handle_event(
            {
                'type': 'tool_result',
                'tool_name': 'bash',
                'ok': True,
                'metadata': {'action': 'bash', 'exit_code': 0},
            }
        )
        bridge.complete(
            AgentRunResult(
                final_output='Hello world',
                turns=1,
                tool_calls=1,
                transcript=(),
                usage=UsageStats(input_tokens=8, output_tokens=4),
                total_cost_usd=0.00125,
                stop_reason='stop',
                session_id='session-1',
                scratchpad_directory='C:/workspace/.port_sessions/scratchpad/session-1',
            )
        )

        rendered = ''.join(chunks)
        self.assertIn('[user] Inspect repo', rendered)
        self.assertIn('[assistant] Hello world', rendered)
        self.assertIn('[command] pwd', rendered)
        self.assertIn('[tool-output:bash:stdout]', rendered)
        self.assertIn('/workspace', rendered)
        self.assertIn('[usage] total_tokens=12', rendered)
        self.assertIn('[session] session_id=session-1', rendered)
        self.assertFalse(state.busy)
        self.assertEqual(state.last_tool, 'bash')
        self.assertEqual(state.last_stop_reason, 'stop')
        self.assertEqual(state.total_tokens, 12)

    def test_event_bridge_emits_final_output_when_response_is_not_streamed(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Summarize the repo')
        bridge.complete(
            AgentRunResult(
                final_output='Completed without streaming.',
                turns=1,
                tool_calls=0,
                transcript=(),
                usage=UsageStats(input_tokens=3, output_tokens=2),
                total_cost_usd=0.0,
                stop_reason='stop',
                session_id='session-2',
            )
        )

        rendered = ''.join(chunks)
        self.assertIn('[assistant] Completed without streaming.', rendered)
        self.assertEqual(state.session_id, 'session-2')


if __name__ == '__main__':
    unittest.main()
