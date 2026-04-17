"""Tests for the Bedrock LLM provider."""

import json
import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def provider():
    # Stub boto3 before importing BedrockProvider
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        with patch("src.llm.bedrock.config") as mock_config:
            mock_config.bedrock_region = "us-east-1"
            mock_config.bedrock_model_id = "anthropic.claude-3-sonnet"
            from src.llm.bedrock import BedrockProvider
            p = BedrockProvider()
            yield p


@pytest.mark.asyncio
async def test_generate_basic(provider):
    provider.bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "Generated text"}]}}
    }

    result = await provider.generate("Hello")

    assert result == "Generated text"
    call_kwargs = provider.bedrock.converse.call_args[1]
    assert call_kwargs["modelId"] == "anthropic.claude-3-sonnet"
    assert "system" not in call_kwargs


@pytest.mark.asyncio
async def test_generate_with_system_prompt(provider):
    provider.bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "reply"}]}}
    }

    await provider.generate("question", system_prompt="Be concise")

    call_kwargs = provider.bedrock.converse.call_args[1]
    assert call_kwargs["system"] == [{"text": "Be concise"}]


@pytest.mark.asyncio
async def test_embed(provider):
    embedding_response = json.dumps({"embedding": [0.5, 0.6, 0.7]}).encode()
    provider.bedrock.invoke_model.return_value = {
        "body": BytesIO(embedding_response),
    }

    result = await provider.embed("test text")

    assert result == [0.5, 0.6, 0.7]
    call_kwargs = provider.bedrock.invoke_model.call_args[1]
    assert call_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(call_kwargs["body"])
    assert body["inputText"] == "test text"
