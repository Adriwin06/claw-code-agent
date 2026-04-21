from __future__ import annotations

import unittest
from unittest.mock import patch

from src.cli.agent_cli_config import _build_model_config, _default_model_from_env
from src.cli.parser_build import build_parser


class AgentCliConfigTests(unittest.TestCase):
    def test_default_model_from_env_prefixes_ollama_model_for_openai_compatible_base_url(self) -> None:
        env = {
            'OPENAI_BASE_URL': 'http://127.0.0.1:11434/v1',
            'OPENAI_MODEL': 'gemma4:e4b',
            'LLM_PROVIDER': 'ollama_chat',
        }

        with patch.dict('os.environ', env, clear=False):
            self.assertEqual(_default_model_from_env(), 'openai/gemma4:e4b')

    def test_build_model_config_prefixes_ollama_model_for_openai_base_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'agent',
                'hello',
                '--model',
                'gemma4:e4b',
                '--base-url',
                'http://127.0.0.1:11434/v1',
                '--cwd',
                '.',
            ]
        )

        with patch.dict('os.environ', {'LLM_PROVIDER': 'ollama_chat'}, clear=False):
            config = _build_model_config(args)

        self.assertEqual(config.model, 'openai/gemma4:e4b')

    def test_build_model_config_keeps_provider_prefix_for_non_openai_provider(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'agent',
                'hello',
                '--model',
                'claude-sonnet-4',
                '--base-url',
                'https://example.invalid/v1',
                '--cwd',
                '.',
            ]
        )

        with patch.dict('os.environ', {'LLM_PROVIDER': 'anthropic'}, clear=False):
            config = _build_model_config(args)

        self.assertEqual(config.model, 'anthropic/claude-sonnet-4')

    def test_build_model_config_uses_ollama_prefix_without_openai_compatible_base_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'agent',
                'hello',
                '--model',
                'gemma4:e4b',
                '--cwd',
                '.',
            ]
        )

        with patch.dict('os.environ', {'LLM_PROVIDER': 'ollama_chat'}, clear=False):
            config = _build_model_config(args)

        self.assertEqual(config.model, 'ollama/gemma4:e4b')


if __name__ == '__main__':
    unittest.main()
