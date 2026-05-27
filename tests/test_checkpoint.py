"""Tests for src.agent.runtime.checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.agent.agent_runtime import LocalCodingAgent
from src.agent.agent_types import AgentRuntimeConfig, ModelConfig
from src.session.session_store import load_agent_session
from src.agent.runtime.checkpoint import (
    checkpoint_exists,
    cleanup_subsequent_checkpoints,
    create_checkpoint,
    list_checkpoints,
    restore_checkpoint,
)
from tests.test_helpers import ScriptedLLMClient


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "hello.py").write_text("print('hello')\n")
    (ws / "sub").mkdir()
    (ws / "sub" / "data.txt").write_text("some data\n")
    return ws


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".port_sessions" / "agent"
    d.mkdir(parents=True)
    return d


SESSION_ID = "test_session_abc"


class TestCreateCheckpoint:
    def test_creates_checkpoint_directory(self, workspace: Path, session_dir: Path) -> None:
        cp = create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert cp.is_dir()
        assert (cp / "hello.py").read_text() == "print('hello')\n"
        assert (cp / "sub" / "data.txt").read_text() == "some data\n"

    def test_excludes_excluded_directories(self, workspace: Path, session_dir: Path) -> None:
        (workspace / "__pycache__").mkdir()
        (workspace / "__pycache__" / "cached.pyc").write_bytes(b"\x00")
        (workspace / ".git").mkdir()
        (workspace / ".git" / "config").write_text("git config")

        cp = create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert not (cp / "__pycache__").exists()
        assert not (cp / ".git").exists()

    def test_overwrites_existing_checkpoint(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        (workspace / "hello.py").write_text("print('updated')\n")
        cp = create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert (cp / "hello.py").read_text() == "print('updated')\n"


class TestRestoreCheckpoint:
    def test_restores_modified_files(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        (workspace / "hello.py").write_text("print('modified')\n")
        assert restore_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert (workspace / "hello.py").read_text() == "print('hello')\n"

    def test_removes_added_files(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        (workspace / "new_file.txt").write_text("I am new")
        assert restore_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert not (workspace / "new_file.txt").exists()

    def test_restores_deleted_files(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        os.remove(workspace / "sub" / "data.txt")
        assert restore_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert (workspace / "sub" / "data.txt").read_text() == "some data\n"

    def test_preserves_excluded_directory_trees(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        excluded_empty_dir = workspace / ".git" / "objects" / "pack"
        excluded_empty_dir.mkdir(parents=True)
        (workspace / "new_file.txt").write_text("I am new")

        assert restore_checkpoint(workspace, SESSION_ID, 0, session_dir)

        assert excluded_empty_dir.is_dir()
        assert not (workspace / "new_file.txt").exists()

    def test_returns_false_for_missing_checkpoint(self, workspace: Path, session_dir: Path) -> None:
        assert not restore_checkpoint(workspace, SESSION_ID, 99, session_dir)

    def test_roundtrip_with_multiple_checkpoints(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        original_content = (workspace / "hello.py").read_text()

        # Simulate changes after turn 1.
        (workspace / "hello.py").write_text("print('turn 1')\n")
        (workspace / "turn1_file.txt").write_text("new")
        create_checkpoint(workspace, SESSION_ID, 2, session_dir)

        # Simulate changes after turn 2.
        (workspace / "hello.py").write_text("print('turn 2')\n")
        os.remove(workspace / "sub" / "data.txt")
        create_checkpoint(workspace, SESSION_ID, 4, session_dir)

        # Rewind to checkpoint 0.
        assert restore_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert (workspace / "hello.py").read_text() == original_content
        assert not (workspace / "turn1_file.txt").exists()
        assert (workspace / "sub" / "data.txt").read_text() == "some data\n"

        # Rewind to checkpoint 2.
        restore_checkpoint(workspace, SESSION_ID, 2, session_dir)
        assert (workspace / "hello.py").read_text() == "print('turn 1')\n"
        assert (workspace / "turn1_file.txt").read_text() == "new"
        assert (workspace / "sub" / "data.txt").read_text() == "some data\n"


class TestCleanupSubsequentCheckpoints:
    def test_deletes_subsequent_checkpoints(self, workspace: Path, session_dir: Path) -> None:
        for i in range(5):
            create_checkpoint(workspace, SESSION_ID, i, session_dir)

        deleted = cleanup_subsequent_checkpoints(SESSION_ID, 2, session_dir)
        assert deleted == 2  # Checkpoints 3 and 4

        remaining = list_checkpoints(SESSION_ID, session_dir)
        assert remaining == [0, 1, 2]

    def test_no_op_when_nothing_to_delete(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert cleanup_subsequent_checkpoints(SESSION_ID, 0, session_dir) == 0

    def test_handles_missing_session(self, session_dir: Path) -> None:
        assert cleanup_subsequent_checkpoints("nonexistent", 0, session_dir) == 0


class TestCheckpointExists:
    def test_returns_true_for_existing(self, workspace: Path, session_dir: Path) -> None:
        create_checkpoint(workspace, SESSION_ID, 0, session_dir)
        assert checkpoint_exists(SESSION_ID, 0, session_dir)

    def test_returns_false_for_missing(self, session_dir: Path) -> None:
        assert not checkpoint_exists(SESSION_ID, 0, session_dir)


class TestListCheckpoints:
    def test_lists_sorted_counts(self, workspace: Path, session_dir: Path) -> None:
        for i in [3, 0, 2, 5]:
            create_checkpoint(workspace, SESSION_ID, i, session_dir)
        assert list_checkpoints(SESSION_ID, session_dir) == [0, 2, 3, 5]

    def test_returns_empty_for_missing_session(self, session_dir: Path) -> None:
        assert list_checkpoints("nonexistent", session_dir) == []


class TestAgentCheckpointIntegration:
    def test_run_creates_initial_and_final_checkpoints(self, workspace: Path, session_dir: Path) -> None:
        agent = LocalCodingAgent(
            model_config=ModelConfig(model="test-model"),
            runtime_config=AgentRuntimeConfig(
                cwd=workspace,
                session_directory=session_dir,
            ),
        )
        agent.client = ScriptedLLMClient(
            completion_responses=[
                {
                    "choices": [
                        {
                            "message": {"content": "done"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ]
        )

        result = agent.run("hello")
        assert result.session_id is not None
        stored = load_agent_session(result.session_id, directory=session_dir)

        checkpoints = list_checkpoints(result.session_id, session_dir)
        assert 0 in checkpoints
        assert any(0 < count < len(stored.messages) for count in checkpoints)
        assert len(stored.messages) in checkpoints
