"""Tests for the Ollama LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.ollama import OllamaProvider


@pytest.fixture
def provider():
    with patch("src.llm.ollama.config") as mock_config:
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_model = "llama3.1"
        p = OllamaProvider()
        p.client = AsyncMock()
        yield p


@pytest.mark.asyncio
async def test_generate_without_system_prompt(provider):
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "Hello!"}}
    mock_response.raise_for_status = MagicMock()
    provider.client.post.return_value = mock_response

    result = await provider.generate("Hi there")

    assert result == "Hello!"
    call_args = provider.client.post.call_args
    body = call_args[1]["json"]
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_generate_with_system_prompt(provider):
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "response"}}
    mock_response.raise_for_status = MagicMock()
    provider.client.post.return_value = mock_response

    await provider.generate("question", system_prompt="You are helpful")

    call_args = provider.client.post.call_args
    body = call_args[1]["json"]
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "You are helpful"


@pytest.mark.asyncio
async def test_embed(provider):
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()
    provider.client.post.return_value = mock_response

    result = await provider.embed("test text")

    assert result == [0.1, 0.2, 0.3]
    call_args = provider.client.post.call_args
    assert "/api/embeddings" in call_args[0][0]
    body = call_args[1]["json"]
    assert body["prompt"] == "test text"
