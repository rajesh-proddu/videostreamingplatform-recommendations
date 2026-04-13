"""Tests for LLM provider factory."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.provider import LLMProvider


class TestGetLlmProvider:
    def setup_method(self):
        """Reset the singleton before each test."""
        import src.llm.provider as mod
        mod._provider_instance = None

    def test_get_ollama_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")

        mock_ollama_cls = MagicMock(spec=LLMProvider)
        with patch("src.llm.provider.config") as mock_config:
            mock_config.llm_provider = "ollama"
            with patch.dict("sys.modules", {"src.llm.ollama": MagicMock(OllamaProvider=mock_ollama_cls)}):
                import src.llm.provider as mod
                mod._provider_instance = None
                result = mod.get_llm_provider()
                assert result == mock_ollama_cls.return_value

    def test_get_bedrock_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")

        mock_bedrock_cls = MagicMock(spec=LLMProvider)
        with patch("src.llm.provider.config") as mock_config:
            mock_config.llm_provider = "bedrock"
            with patch.dict("sys.modules", {"src.llm.bedrock": MagicMock(BedrockProvider=mock_bedrock_cls)}):
                import src.llm.provider as mod
                mod._provider_instance = None
                result = mod.get_llm_provider()
                assert result == mock_bedrock_cls.return_value

    def test_unknown_provider_raises(self):
        with patch("src.llm.provider.config") as mock_config:
            mock_config.llm_provider = "unknown"
            import src.llm.provider as mod
            mod._provider_instance = None
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                mod.get_llm_provider()
