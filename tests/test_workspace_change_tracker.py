from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent.models.types import ToolCall
from src.agent.runtime.change_tracker import WorkspaceChangeTracker


class WorkspaceChangeTrackerTests(unittest.TestCase):
    def test_collect_tool_changes_reports_added_modified_and_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'keep.txt').write_text('one\ntwo\n', encoding='utf-8')
            (workspace / 'delete.txt').write_text('remove me\n', encoding='utf-8')
            tracker = WorkspaceChangeTracker(workspace)

            (workspace / 'keep.txt').write_text('one\nTWO\nthree\n', encoding='utf-8')
            (workspace / 'delete.txt').unlink()
            (workspace / 'new.txt').write_text('created\n', encoding='utf-8')
            event = tracker.collect_tool_changes(
                tool_call=ToolCall(id='call-1', name='bash', arguments={}),
                turn_index=1,
            )

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event['type'], 'workspace_change')
        self.assertEqual(event['file_count'], 3)
        self.assertEqual(event['added_files'], 1)
        self.assertEqual(event['modified_files'], 1)
        self.assertEqual(event['deleted_files'], 1)
        self.assertEqual(
            event['changed_paths'],
            ['delete.txt', 'keep.txt', 'new.txt'],
        )
        files = {file['path']: file for file in event['files']}
        self.assertEqual(files['new.txt']['status'], 'added')
        self.assertEqual(files['delete.txt']['status'], 'deleted')
        self.assertIn('+TWO', files['keep.txt']['diff'])
        self.assertIn('-two', files['keep.txt']['diff'])

    def test_collect_tool_changes_ignores_runtime_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tracker = WorkspaceChangeTracker(workspace)

            session_file = workspace / '.port_sessions' / 'agent' / 'session.json'
            session_file.parent.mkdir(parents=True)
            session_file.write_text('{}\n', encoding='utf-8')
            event = tracker.collect_tool_changes(
                tool_call=ToolCall(id='call-1', name='write_file', arguments={}),
                turn_index=1,
            )

        self.assertIsNone(event)

    def test_collect_run_recap_reports_net_change_from_initial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            tracker = WorkspaceChangeTracker(workspace)

            target = workspace / 'notes.txt'
            target.write_text('one\n', encoding='utf-8')
            tracker.collect_tool_changes(
                tool_call=ToolCall(id='call-1', name='write_file', arguments={}),
                turn_index=1,
            )
            target.write_text('one\ntwo\n', encoding='utf-8')
            tracker.collect_tool_changes(
                tool_call=ToolCall(id='call-2', name='edit_file', arguments={}),
                turn_index=1,
            )
            (workspace / 'outside-tool.txt').write_text('ignore\n', encoding='utf-8')
            recap = tracker.collect_run_recap()

        self.assertIsNotNone(recap)
        assert recap is not None
        self.assertEqual(recap['type'], 'workspace_change_recap')
        self.assertEqual(recap['file_count'], 1)
        self.assertEqual(recap['changed_paths'], ['notes.txt'])
        self.assertEqual(recap['added_lines'], 2)
        self.assertEqual(recap['removed_lines'], 0)
        self.assertEqual(recap['files'][0]['path'], 'notes.txt')


if __name__ == '__main__':
    unittest.main()
