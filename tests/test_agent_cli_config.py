from __future__ import annotations

import unittest
from unittest.mock import patch

from src.cli.agent_cli_config import _build_model_config, _default_model_from_env
from src.cli.parser_build import build_parser


class AgentCliConfigTests(unittest.TestCase):
    def test_default_model_from_env_prefixes_ollama_model_for_ollama_base_url(self) -> None:
        env = {
            'LLM_API_BASE': 'http://127.0.0.1:11434/v1',
            'LLM_MODEL': 'gemma4:e4b',
            'LLM_PROVIDER': 'ollama_chat',
        }

        with patch.dict('os.environ', env, clear=False):
            self.assertEqual(_default_model_from_env(), 'openai/gemma4:e4b')

    def test_build_model_config_prefixes_ollama_model_for_ollama_base_url(self) -> None:
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

    def test_build_model_config_uses_ollama_prefix_without_ollama_base_url(self) -> None:
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

    def test_build_model_config_uses_openai_provider_key_when_no_llm_api_key_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['agent', 'hello', '--cwd', '.'])

        with patch.dict(
            'os.environ',
            {
                'LLM_PROVIDER': 'openai',
                'OPENAI_API_KEY': 'sk-openai',
                'LLM_API_KEY': '',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'sk-openai')

    def test_build_model_config_uses_anthropic_provider_key(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['agent', 'hello', '--model', 'claude-sonnet-4', '--cwd', '.'])

        with patch.dict(
            'os.environ',
            {
                'LLM_PROVIDER': 'anthropic',
                'ANTHROPIC_API_KEY': 'sk-anthropic',
                'LLM_API_KEY': '',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'sk-anthropic')

    def test_build_model_config_uses_gemini_provider_key(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['agent', 'hello', '--model', 'gemini-2.5-pro', '--cwd', '.'])

        with patch.dict(
            'os.environ',
            {
                'LLM_PROVIDER': 'gemini',
                'GEMINI_API_KEY': 'sk-gemini',
                'LLM_API_KEY': '',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'sk-gemini')

    def test_build_model_config_uses_provider_key_inferred_from_model_prefix(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ['agent', 'hello', '--model', 'openai/gpt-4o-mini', '--cwd', '.']
        )

        with patch.dict(
            'os.environ',
            {
                'OPENAI_API_KEY': 'sk-openai',
                'LLM_API_KEY': '',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'sk-openai')

    def test_build_model_config_prefers_llm_api_key_override_over_provider_specific_key(self) -> None:
        parser = build_parser()
        args = parser.parse_args(['agent', 'hello', '--cwd', '.'])

        with patch.dict(
            'os.environ',
            {
                'LLM_PROVIDER': 'openai',
                'LLM_API_KEY': 'override-key',
                'OPENAI_API_KEY': 'sk-openai',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'override-key')

    def test_build_model_config_uses_openrouter_key_inferred_from_base_url(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                'agent',
                'hello',
                '--base-url',
                'https://openrouter.ai/api/v1',
                '--model',
                'openai/gpt-4o-mini',
                '--cwd',
                '.',
            ]
        )

        with patch.dict(
            'os.environ',
            {
                'OPENROUTER_API_KEY': 'sk-openrouter',
                'LLM_API_KEY': '',
            },
            clear=False,
        ):
            config = _build_model_config(args)

        self.assertEqual(config.api_key, 'sk-openrouter')


if __name__ == '__main__':
    unittest.main()
