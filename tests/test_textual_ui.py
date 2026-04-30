from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent.agent_runtime import LocalCodingAgent
from src.agent.agent_types import AgentPermissions, AgentRunResult, AgentRuntimeConfig, ModelConfig, UsageStats
from src.textual_ui import (
    ActivityItem,
    AgentTuiEventBridge,
    AgentTuiState,
    ConversationEntry,
    ConversationThread,
    ConversationTurn,
    build_working_section_id,
    build_working_section_instance_id,
    build_slash_command_suggestions,
    build_conversation_history_items,
    extract_slash_command_query,
    filter_slash_command_suggestions,
    render_details_panel,
    render_slash_command_suggestion_detail,
    restore_conversation_turns,
    should_route_key_to_prompt,
)
from src.ui.conversation_store import ConversationHistoryStore, workspace_history_key


try:
    TEXTUAL_AVAILABLE = importlib.util.find_spec('textual.app') is not None
except ModuleNotFoundError:
    TEXTUAL_AVAILABLE = False


class TextualUiTests(unittest.TestCase):
    def test_build_working_section_id_uses_textual_safe_identifier(self) -> None:
        identifier = build_working_section_id('conversation-1', 'turn-1')

        self.assertEqual(identifier, 'working-conversation-1-turn-1')
        self.assertNotIn(':', identifier)

    def test_build_working_section_instance_id_adds_section_suffix_only_when_needed(self) -> None:
        single_section = build_working_section_instance_id(
            'conversation-1',
            'turn-1',
            section_index=1,
            section_count=1,
        )
        multi_section = build_working_section_instance_id(
            'conversation-1',
            'turn-1',
            section_index=2,
            section_count=2,
        )

        self.assertEqual(single_section, 'working-conversation-1-turn-1')
        self.assertEqual(multi_section, 'working-conversation-1-turn-1-section-2')
        self.assertNotIn(':', multi_section)

    def test_should_route_key_to_prompt_only_for_printable_chars_when_prompt_unfocused(self) -> None:
        self.assertTrue(
            should_route_key_to_prompt(character='a', prompt_focused=False, busy=False)
        )
        self.assertTrue(
            should_route_key_to_prompt(character='/', prompt_focused=False, busy=False)
        )
        self.assertFalse(
            should_route_key_to_prompt(character=None, prompt_focused=False, busy=False)
        )
        self.assertFalse(
            should_route_key_to_prompt(character='\n', prompt_focused=False, busy=False)
        )
        self.assertFalse(
            should_route_key_to_prompt(character='a', prompt_focused=True, busy=False)
        )
        self.assertFalse(
            should_route_key_to_prompt(character='a', prompt_focused=False, busy=True)
        )

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
        self.assertIn('phase=Idle', rendered)
        self.assertIn('streaming=False', rendered)
        self.assertIn('search_enabled=', rendered)
        self.assertIn('search_context_size=', rendered)

    def test_render_details_panel_highlights_actions_and_selected_turn(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='write, shell',
            status='Ready',
            phase='Ready',
            session_id='session-1',
            prompt_count=2,
            total_tokens=42,
            total_cost_usd=0.125,
            search_enabled=False,
            search_context_size='high',
            search_default_max_results=8,
            search_provider_count=2,
            search_manifest_count=1,
            search_active_provider='local-search (searxng)',
        )
        turn = ConversationTurn(
            turn_id='turn-1',
            user_prompt='Inspect the repo',
            assistant_response='Done.',
            tool_count=2,
            stop_reason='stop',
        )
        rendered = render_details_panel(
            state,
            turn,
            (ActivityItem(key='latest', label='Tool finished'),),
            auto_follow=True,
        )

        self.assertIn('Run', rendered)
        self.assertIn('Selected Turn', rendered)
        self.assertIn('Search', rendered)
        self.assertIn('enabled=False', rendered)
        self.assertIn('context_size=high', rendered)
        self.assertIn('default_max_results=8', rendered)
        self.assertIn('active_provider=local-search (searxng)', rendered)
        self.assertIn('workspace_identity=', rendered)
        self.assertIn('history_key=', rendered)
        self.assertIn('stop_reason=completed', rendered)
        self.assertIn('last_activity=Tool finished', rendered)
        self.assertIn('Ctrl+U: reuse selected prompt', rendered)
        self.assertIn('Ctrl+T: rerun selected turn', rendered)

    def test_restore_conversation_turns_skips_internal_messages_and_tracks_tools(self) -> None:
        turns = restore_conversation_turns(
            (
                {'role': 'system', 'content': 'system prompt'},
                {
                    'role': 'user',
                    'content': '<system-reminder>\nctx',
                    'message_id': 'user_context_0',
                },
                {'role': 'user', 'content': 'First question', 'message_id': 'user_0'},
                {'role': 'assistant', 'content': 'First answer'},
                {'role': 'tool', 'content': 'pwd', 'tool_call_id': 'call-1'},
                {
                    'role': 'user',
                    'content': 'plugin runtime note',
                    'message_id': 'plugin_tool_runtime_call-1',
                },
                {'role': 'user', 'content': 'Second question', 'message_id': 'user_1'},
                {'role': 'assistant', 'content': 'Second answer'},
            )
        )

        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].user_prompt, 'First question')
        self.assertEqual(turns[0].assistant_response, 'First answer')
        self.assertEqual(turns[0].tool_count, 1)
        self.assertTrue(turns[0].restored)
        self.assertEqual([entry.kind for entry in turns[0].entries], ['assistant', 'tool'])
        self.assertEqual(turns[1].user_prompt, 'Second question')
        history = build_conversation_history_items(turns)
        self.assertEqual(history[0].label, '01. First question')
        self.assertEqual(history[1].assistant_preview, 'Second answer')

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
        self.assertEqual(state.phase, 'Ready')
        self.assertEqual(len(bridge.turns), 1)
        self.assertEqual(bridge.turns[0].assistant_response, 'Hello world')
        self.assertEqual(bridge.turns[0].tool_count, 1)
        self.assertEqual(
            [entry.kind for entry in bridge.turns[0].entries],
            ['assistant', 'tool', 'tool_output', 'tool_result'],
        )
        self.assertEqual(bridge.turns[0].entries[1].title, 'Tool Call: bash')
        self.assertEqual(bridge.turns[0].entries[3].title, 'Tool Result: bash')
        self.assertIn('arguments:', bridge.turns[0].entries[1].content)
        self.assertIn('metadata:', bridge.turns[0].entries[3].content)
        self.assertTrue(any(item.label == 'Tool finished' for item in bridge.activity_items))
        self.assertEqual(bridge.history_items[0].assistant_preview, 'Hello world')

    def test_event_bridge_renders_web_search_result_details(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('What is today\'s date?')
        bridge.handle_event(
            {
                'type': 'tool_result',
                'tool_name': 'web_search',
                'ok': True,
                'metadata': {
                    'action': 'web_search',
                    'query': 'What is the current date?',
                    'result_count': 1,
                    'top_urls': ['https://example.com/date'],
                },
                'content_preview': '# Web Search ...',
            }
        )

        rendered = ''.join(chunks)
        self.assertIn(
            '[search] ok=True query=What is the current date? results=1 top=https://example.com/date',
            rendered,
        )
        self.assertEqual(bridge.turns[0].entries[0].title, 'Tool Result: web_search')
        self.assertIn('results=1', bridge.turns[0].entries[0].content)

    def test_event_bridge_renders_failed_mcp_result_with_error_details(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Evaluate expression')
        bridge.handle_event(
            {
                'type': 'tool_result',
                'tool_name': 'mcp_call_tool',
                'tool_call_id': 'call-42',
                'ok': False,
                'content': 'MCP call failed: server unavailable',
                'content_preview': 'MCP call failed: server unavailable',
                'metadata': {
                    'action': 'mcp_call_tool',
                    'tool_name': 'evaluate_expression',
                    'requested_server': 'sagemath',
                },
            }
        )

        rendered = ''.join(chunks)
        self.assertIn('[mcp] server=sagemath tool=evaluate_expression ok=False', rendered)
        self.assertIn('error=MCP call failed: server unavailable', rendered)
        self.assertIn('content:', bridge.turns[0].entries[0].content)
        self.assertIn('server unavailable', bridge.turns[0].entries[0].content)
        self.assertIn('metadata:', bridge.turns[0].entries[0].content)

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
        self.assertEqual(bridge.turns[0].assistant_response, 'Completed without streaming.')
        self.assertEqual(bridge.turns[0].assistant_status, 'Ready')
        self.assertEqual(bridge.turns[0].entries[0].kind, 'assistant')

    def test_event_bridge_treats_completed_alias_as_ready(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=lambda _text: None)

        bridge.begin_prompt('Summarize the repo')
        bridge.complete(
            AgentRunResult(
                final_output='Completed alias.',
                turns=1,
                tool_calls=0,
                transcript=(),
                usage=UsageStats(input_tokens=3, output_tokens=2),
                total_cost_usd=0.0,
                stop_reason='completed',
                session_id='session-3',
            )
        )

        self.assertEqual(bridge.turns[0].assistant_status, 'Ready')
        self.assertEqual(state.last_stop_reason, 'completed')

    def test_event_bridge_renders_cancelled_run_as_stopped(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Generate a long reply')
        bridge.handle_event({'type': 'message_start'})
        bridge.handle_event({'type': 'content_delta', 'delta': 'Partial answer'})
        bridge.request_cancel('Stop requested by user')
        bridge.cancel('Stopped by user')

        self.assertFalse(state.busy)
        self.assertEqual(state.last_stop_reason, 'cancelled')
        self.assertEqual(bridge.turns[0].assistant_status, 'Stopped')
        self.assertEqual(bridge.turns[0].stop_reason, 'cancelled')
        self.assertIn('Stop Requested', [entry.title for entry in bridge.turns[0].entries])
        self.assertIn('stop_reason=cancelled', ''.join(chunks))

    def test_event_bridge_strips_leaked_channel_marker_from_assistant_text(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Update the CSS')
        bridge.handle_event({'type': 'content_delta', 'delta': 'I will update style.css.<channel>|>'})

        self.assertEqual(bridge.turns[0].assistant_response, 'I will update style.css.')
        self.assertEqual(bridge.turns[0].entries[-1].content, 'I will update style.css.')
        self.assertNotIn('<channel>', ''.join(chunks))

    def test_event_bridge_failed_partial_run_gets_error_stop_reason(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=lambda _text: None)

        bridge.begin_prompt('Update the CSS')
        bridge.handle_event({'type': 'content_delta', 'delta': 'Partial answer'})
        bridge.fail(RuntimeError('backend timed out'))

        self.assertFalse(state.busy)
        self.assertEqual(state.status, 'Error')
        self.assertEqual(state.last_stop_reason, 'RuntimeError')
        self.assertEqual(bridge.turns[0].assistant_status, 'Error')
        self.assertEqual(bridge.turns[0].stop_reason, 'RuntimeError')
        self.assertIn('Error', [entry.title for entry in bridge.turns[0].entries])

    def test_event_bridge_renders_live_delegate_activity(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)

        bridge.begin_prompt('Use a sub-agent')
        bridge.handle_event(
            {
                'type': 'delegate_subtask_start',
                'label': 'scan',
                'index': 1,
                'batch_index': 1,
                'depends_on': [],
                'prompt_preview': 'Inspect the repository.',
            }
        )
        bridge.handle_event(
            {
                'type': 'delegate_subtask_event',
                'label': 'scan',
                'batch_index': 1,
                'child_event_type': 'message_start',
            }
        )
        bridge.handle_event(
            {
                'type': 'delegate_subtask_event',
                'label': 'scan',
                'batch_index': 1,
                'child_event_type': 'tool_start',
                'tool_name': 'bash',
                'arguments': {'command': 'pwd'},
            }
        )
        bridge.handle_event(
            {
                'type': 'delegate_subtask_event',
                'label': 'scan',
                'batch_index': 1,
                'child_event_type': 'tool_result',
                'tool_name': 'bash',
                'ok': True,
                'metadata': {'action': 'bash', 'exit_code': 0},
            }
        )
        bridge.handle_event(
            {
                'type': 'delegate_subtask_result',
                'label': 'scan',
                'batch_index': 1,
                'session_id': 'child-session',
                'turns': 1,
                'tool_calls': 1,
                'stop_reason': 'stop',
                'output_preview': 'Repository scan complete.',
            }
        )

        titles = [entry.title for entry in bridge.turns[0].entries]
        self.assertIn('Sub-Agent Started: scan', titles)
        self.assertIn('Sub-Agent Live: scan', titles)
        self.assertIn('Sub-Agent: scan', titles)
        live_entry = next(
            entry
            for entry in bridge.turns[0].entries
            if entry.title == 'Sub-Agent Live: scan'
        )
        self.assertIn('model call started', live_entry.content)
        self.assertIn('tool started: bash pwd', live_entry.content)
        self.assertIn('tool finished: bash ok=True exit_code=0', live_entry.content)
        result_entry = bridge.turns[0].entries[-1]
        self.assertIn('output_preview=Repository scan complete.', result_entry.content)
        self.assertEqual(state.phase, 'Delegating')
        self.assertTrue(
            any(item.label == 'Sub-agent finished' for item in bridge.activity_items)
        )
        self.assertIn('[delegate] started scan', ''.join(chunks))

    def test_event_bridge_compacts_delegate_tool_start_details(self) -> None:
        chunks: list[str] = []
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=chunks.append)
        long_prompt = 'Create a complete website. ' * 80

        bridge.begin_prompt('Delegate planning')
        bridge.handle_event(
            {
                'type': 'tool_start',
                'tool_name': 'Agent',
                'tool_call_id': 'call-agent',
                'arguments': {
                    'subagent_type': 'Plan',
                    'prompt': long_prompt,
                },
            }
        )

        entry = bridge.turns[0].entries[-1]
        self.assertEqual(entry.title, 'Sub-Agent Requested: Plan')
        self.assertIn('prompt_preview=', entry.content)
        self.assertNotIn('arguments:', entry.content)
        self.assertLess(len(entry.content), 900)
        self.assertIn('[delegate] subagent=Plan', ''.join(chunks))

    def test_event_bridge_compacts_web_fetch_result_details(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=lambda _text: None)
        html = (
            '<!doctype html>\n'
            '<html><head><script>large()</script></head></html>'
        ) * 80

        bridge.begin_prompt('Fetch page')
        bridge.handle_event(
            {
                'type': 'tool_result',
                'tool_name': 'web_fetch',
                'tool_call_id': 'call-fetch',
                'ok': True,
                'content': html,
                'content_preview': '<!doctype html> <html>...',
                'metadata': {
                    'action': 'web_fetch',
                    'url': 'https://example.com',
                    'fetched_chars': len(html),
                    'truncated': False,
                },
            }
        )

        entry = bridge.turns[0].entries[-1]
        self.assertIn('preview=<!doctype html> <html>...', entry.content)
        self.assertIn('url=https://example.com', entry.content)
        self.assertNotIn('<script>large()</script>', entry.content)
        self.assertLess(len(entry.content), 600)

    def test_event_bridge_marks_max_turns_completion_as_stopped(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=lambda _text: None)

        bridge.begin_prompt('Do a long task')
        bridge.complete(
            AgentRunResult(
                final_output='Stopped before final answer.',
                turns=12,
                tool_calls=3,
                transcript=(),
                usage=UsageStats(input_tokens=10, output_tokens=5),
                stop_reason='max_turns',
            )
        )

        self.assertEqual(bridge.turns[0].assistant_status, 'Stopped')
        self.assertEqual(bridge.turns[0].phase_label, 'Stopped')
        self.assertIn('Run Stopped', [entry.title for entry in bridge.turns[0].entries])
        self.assertIn('last run stopped: max_turns', state.phase_detail)

    def test_event_bridge_restore_history_populates_turns_and_activity(self) -> None:
        state = AgentTuiState(
            workspace='C:/workspace',
            model='demo-model',
            permissions='read-only',
        )
        bridge = AgentTuiEventBridge(state, emit_data=lambda _text: None)

        bridge.restore_history(
            restore_conversation_turns(
                (
                    {'role': 'user', 'content': 'Resume me'},
                    {'role': 'assistant', 'content': 'Restored answer'},
                )
            )
        )

        self.assertEqual(state.conversation_turns, 1)
        self.assertEqual(bridge.turns[0].assistant_response, 'Restored answer')
        self.assertEqual(bridge.turns[0].entries[0].kind, 'assistant')
        self.assertEqual(bridge.activity_items[0].label, 'Conversation restored')

    def test_restore_conversation_turns_strips_leaked_channel_marker(self) -> None:
        turns = restore_conversation_turns(
            (
                {'role': 'user', 'content': 'Update animations'},
                {'role': 'assistant', 'content': 'I will edit style.css.<channel>|>'},
            )
        )

        self.assertEqual(turns[0].assistant_response, 'I will edit style.css.')
        self.assertEqual(turns[0].entries[0].content, 'I will edit style.css.')

    def test_conversation_history_store_round_trips_by_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / 'home' / '.claw-code'
            workspace_a = Path(tmp_dir) / 'workspace-a'
            workspace_b = Path(tmp_dir) / 'workspace-b'
            workspace_a.mkdir()
            workspace_b.mkdir()
            store = ConversationHistoryStore(root)
            conversation = ConversationThread(
                conversation_id='conversation-3',
                title='Inspect repo',
                session_id='session-a',
                turns=(
                    ConversationTurn(
                        turn_id='turn-1',
                        user_prompt='Inspect repo',
                        assistant_response='Done.',
                        assistant_status='Ready',
                        phase_label='Completed',
                        session_id='session-a',
                        entries=[
                            ConversationEntry(
                                entry_id='turn-1-entry-1',
                                kind='assistant',
                                title='Assistant',
                                content='Done.',
                                status='ok',
                            )
                        ],
                    ),
                ),
            )

            path = store.save_workspace_conversations(
                workspace_a,
                (conversation,),
                active_conversation_id='conversation-3',
            )
            snapshot_a = store.load_workspace(workspace_a)
            snapshot_b = store.load_workspace(workspace_b)

            self.assertEqual(path.parent, root / 'conversations')
            self.assertIn(workspace_history_key(workspace_a), path.name)
            self.assertEqual(snapshot_a.active_conversation_id, 'conversation-3')
            self.assertEqual(len(snapshot_a.conversations), 1)
            self.assertEqual(snapshot_a.conversations[0].session_id, 'session-a')
            self.assertEqual(snapshot_a.conversations[0].turns[0].user_prompt, 'Inspect repo')
            self.assertEqual(snapshot_a.conversations[0].turns[0].entries[0].content, 'Done.')
            self.assertEqual(snapshot_b.conversations, ())

    def test_conversation_history_store_uses_host_workspace_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / '.claw-code'
            runtime_workspace = Path(tmp_dir) / 'workspace'
            runtime_workspace.mkdir()
            store = ConversationHistoryStore(root)

            with mock.patch.dict(
                'os.environ',
                {'CLAW_HOST_WORKSPACE': r'C:\Users\Ada\project-one'},
            ):
                path_one = store.save_workspace_conversations(
                    runtime_workspace,
                    (ConversationThread('conversation-1', 'One'),),
                    active_conversation_id='conversation-1',
                )
                key_one = workspace_history_key(runtime_workspace)

            with mock.patch.dict(
                'os.environ',
                {'CLAW_HOST_WORKSPACE': r'C:\Users\Ada\project-two'},
            ):
                path_two = store.save_workspace_conversations(
                    runtime_workspace,
                    (ConversationThread('conversation-1', 'Two'),),
                    active_conversation_id='conversation-1',
                )
                snapshot_two = store.load_workspace(runtime_workspace)

            with mock.patch.dict(
                'os.environ',
                {'CLAW_HOST_WORKSPACE': r'C:\Users\Ada\project-one'},
            ):
                snapshot_one = store.load_workspace(runtime_workspace)

            self.assertNotEqual(path_one, path_two)
            self.assertIn(key_one, path_one.name)
            self.assertEqual(snapshot_one.conversations[0].title, 'One')
            self.assertEqual(snapshot_two.conversations[0].title, 'Two')

    def test_conversation_history_store_deletes_workspace_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / '.claw-code'
            workspace = Path(tmp_dir) / 'workspace'
            workspace.mkdir()
            store = ConversationHistoryStore(root)
            store.save_workspace_conversations(
                workspace,
                (
                    ConversationThread('conversation-1', 'One'),
                    ConversationThread('conversation-2', 'Two'),
                ),
                active_conversation_id='conversation-2',
            )

            deleted = store.delete_workspace_conversation(
                workspace,
                'conversation-1',
                active_conversation_id='conversation-2',
            )
            snapshot = store.load_workspace(workspace)

            self.assertTrue(deleted)
            self.assertEqual(
                [conversation.conversation_id for conversation in snapshot.conversations],
                ['conversation-2'],
            )

    @unittest.skipUnless(TEXTUAL_AVAILABLE, 'textual is not installed')
    def test_agent_tui_prompt_accepts_typing_when_idle(self) -> None:
        from textual.app import App
        from src.textual_ui import run_agent_tui

        captured: dict[str, App] = {}
        original_run = App.run

        def fake_run(app: App, *args, **kwargs) -> None:
            captured['app'] = app

        App.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='demo-model'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
                )
                run_agent_tui(
                    agent,
                    history_store=ConversationHistoryStore(Path(tmp_dir) / '.claw-code-test'),
                )
        finally:
            App.run = original_run

        app = captured['app']

        async def exercise() -> None:
            async with app.run_test() as pilot:
                prompt = app.query_one('#prompt')
                self.assertFalse(prompt.disabled)
                await pilot.press('a', 'b')
                self.assertEqual(prompt.value, 'ab')

        asyncio.run(exercise())

    @unittest.skipUnless(TEXTUAL_AVAILABLE, 'textual is not installed')
    def test_agent_tui_chat_layout_keeps_markdown_inside_turn_card_and_scrolls(self) -> None:
        from textual.app import App
        from src.textual_ui import run_agent_tui

        captured: dict[str, App] = {}
        original_run = App.run

        def fake_run(app: App, *args, **kwargs) -> None:
            captured['app'] = app

        App.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='demo-model'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
                )
                run_agent_tui(
                    agent,
                    history_store=ConversationHistoryStore(Path(tmp_dir) / '.claw-code-test'),
                )
        finally:
            App.run = original_run

        app = captured['app']

        async def exercise() -> None:
            async with app.run_test() as pilot:
                for command in ('/help', '/config', '/mcp', '/skills'):
                    await pilot.click('#prompt')
                    await pilot.press(*list(command), 'enter')
                    await pilot.pause(0.2)

                turn_cards = list(app.query('.turn-card'))
                self.assertGreaterEqual(len(turn_cards), 4)

                first_card = turn_cards[0]
                assistant = app.query_one('.turn-assistant')
                self.assertLessEqual(assistant.region.bottom, first_card.region.bottom)

                scroll = app.query_one('#conversation-scroll')
                self.assertGreater(getattr(scroll, 'max_scroll_y', 0), 0)

        asyncio.run(exercise())

    @unittest.skipUnless(TEXTUAL_AVAILABLE, 'textual is not installed')
    def test_agent_tui_can_create_and_navigate_conversations(self) -> None:
        from textual.app import App
        from src.textual_ui import run_agent_tui

        captured: dict[str, App] = {}
        original_run = App.run

        def fake_run(app: App, *args, **kwargs) -> None:
            captured['app'] = app

        App.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='demo-model'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
                )
                run_agent_tui(
                    agent,
                    history_store=ConversationHistoryStore(Path(tmp_dir) / '.claw-code-test'),
                )
        finally:
            App.run = original_run

        app = captured['app']

        async def exercise() -> None:
            async with app.run_test() as pilot:
                await pilot.click('#new-conversation-button')
                await pilot.pause(0.2)

                self.assertEqual(app._active_conversation_id, 'conversation-2')
                self.assertEqual(len(app._conversations), 2)

                app.action_previous_conversation()
                await pilot.pause(0.2)
                self.assertEqual(app._active_conversation_id, 'conversation-1')

                app.action_next_conversation()
                await pilot.pause(0.2)
                self.assertEqual(app._active_conversation_id, 'conversation-2')

        asyncio.run(exercise())

    @unittest.skipUnless(TEXTUAL_AVAILABLE, 'textual is not installed')
    def test_agent_tui_loads_and_deletes_workspace_history(self) -> None:
        from textual.app import App
        from src.textual_ui import run_agent_tui

        captured: dict[str, App] = {}
        original_run = App.run

        def fake_run(app: App, *args, **kwargs) -> None:
            captured['app'] = app

        App.run = fake_run
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        try:
            tmp_dir = Path(tmp.name)
            workspace = tmp_dir / 'workspace'
            other_workspace = tmp_dir / 'other'
            workspace.mkdir()
            other_workspace.mkdir()
            store = ConversationHistoryStore(tmp_dir / '.claw-code-test')
            store.save_workspace_conversations(
                workspace,
                (
                    ConversationThread(
                        conversation_id='conversation-5',
                        title='Saved task',
                        session_id='session-5',
                        turns=(
                            ConversationTurn(
                                turn_id='turn-1',
                                user_prompt='Saved task',
                                assistant_response='Saved answer',
                                assistant_status='Ready',
                            ),
                        ),
                    ),
                ),
                active_conversation_id='conversation-5',
            )
            store.save_workspace_conversations(
                other_workspace,
                (ConversationThread('conversation-1', 'Other workspace'),),
            )
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='demo-model'),
                runtime_config=AgentRuntimeConfig(cwd=workspace),
            )
            run_agent_tui(agent, history_store=store)
        finally:
            App.run = original_run

        app = captured['app']

        async def exercise() -> None:
            async with app.run_test() as pilot:
                self.assertEqual(app._active_conversation_id, 'conversation-5')
                self.assertEqual(len(app._conversations), 1)
                self.assertEqual(app._conversation_turns[0].user_prompt, 'Saved task')

                app.action_delete_conversation()
                await pilot.pause(0.2)

                self.assertEqual(app._active_conversation_id, 'conversation-1')
                self.assertEqual(app._conversation_turns, ())
                self.assertEqual(store.load_workspace(workspace).conversations, ())
                self.assertEqual(len(store.load_workspace(other_workspace).conversations), 1)

        asyncio.run(exercise())

    @unittest.skipUnless(TEXTUAL_AVAILABLE, 'textual is not installed')
    def test_agent_tui_renders_split_working_sections_with_unique_ids(self) -> None:
        from textual.app import App
        from src.textual_ui import run_agent_tui

        captured: dict[str, App] = {}
        original_run = App.run

        def fake_run(app: App, *args, **kwargs) -> None:
            captured['app'] = app

        App.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='demo-model'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
                )
                run_agent_tui(
                    agent,
                    history_store=ConversationHistoryStore(Path(tmp_dir) / '.claw-code-test'),
                )
        finally:
            App.run = original_run

        app = captured['app']

        async def exercise() -> None:
            async with app.run_test() as pilot:
                app._conversations[0].turns = (
                    ConversationTurn(
                        turn_id='turn-1',
                        user_prompt='Reproduce duplicate working sections',
                        assistant_response='Done.',
                        assistant_status='Ready',
                        phase_label='Completed',
                        tool_count=1,
                        entries=[
                            ConversationEntry(
                                entry_id='turn-1-entry-1',
                                kind='thinking',
                                title='Thinking',
                                content='Planning the tool call.',
                            ),
                            ConversationEntry(
                                entry_id='turn-1-entry-2',
                                kind='assistant',
                                title='Assistant',
                                content='Done.',
                                status='ok',
                            ),
                            ConversationEntry(
                                entry_id='turn-1-entry-3',
                                kind='tool_result',
                                title='Tool Result',
                                content='[tool] bash ok=True',
                                status='ok',
                            ),
                        ],
                    ),
                )
                app._switch_to_conversation('conversation-1')
                await pilot.pause(0.2)

                working_sections = list(app.query('.turn-working'))
                working_ids = {widget.id for widget in working_sections}

                self.assertEqual(len(working_sections), 2)
                self.assertEqual(
                    working_ids,
                    {
                        'working-conversation-1-turn-1-section-1',
                        'working-conversation-1-turn-1-section-2',
                    },
                )

        asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
