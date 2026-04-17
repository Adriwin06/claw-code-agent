from __future__ import annotations

import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.agent.agent_runtime import LocalCodingAgent
from src.agent.agent_types import AgentPermissions, AgentRunResult, AgentRuntimeConfig, ModelConfig, UsageStats
from src.textual_ui import (
    ActivityItem,
    AgentTuiEventBridge,
    AgentTuiState,
    ConversationEntry,
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
        self.assertTrue(any(item.label == 'Tool finished' for item in bridge.activity_items))
        self.assertEqual(bridge.history_items[0].assistant_preview, 'Hello world')

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
                run_agent_tui(agent)
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
                run_agent_tui(agent)
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
                run_agent_tui(agent)
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
                run_agent_tui(agent)
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
