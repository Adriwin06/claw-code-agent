from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agent.agent_types import ModelConfig
from src.llm.factory import (
    build_llm_client,
    resolve_llm_backend,
)
from src.llm.parsers import LLMBackendError
from src.llm.litellm_backend import LiteLLMClient


class LLMFactoryTests(unittest.TestCase):
    def test_resolve_aliases(self) -> None:
        self.assertEqual(resolve_llm_backend('lite-llm'), 'litellm')
        self.assertEqual(resolve_llm_backend('lite_llm'), 'litellm')

    def test_resolve_backend_from_env_default(self) -> None:
        with patch.dict('os.environ', {'CLAW_LLM_BACKEND': 'litellm'}, clear=False):
            self.assertEqual(resolve_llm_backend(None), 'litellm')

    def test_resolve_invalid_backend_raises(self) -> None:
        with self.assertRaises(LLMBackendError):
            resolve_llm_backend('unknown-backend')

    def test_build_litellm_alias_uses_litellm_client(self) -> None:
        client = build_llm_client(
            ModelConfig(model='test-model'),
            backend='lite-llm',
        )
        self.assertIsInstance(client, LiteLLMClient)

    def test_build_litellm_client(self) -> None:
        client = build_llm_client(
            ModelConfig(model='test-model', llm_backend='litellm'),
        )
        self.assertIsInstance(client, LiteLLMClient)


if __name__ == '__main__':
    unittest.main()
