"""Anthropic API LLM provider for direct Claude inference."""

import logging
from typing import Optional

import anthropic

from src.config import config
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider using the Anthropic API directly."""

    def __init__(self):
        token = config.anthropic_api_key
        if token.startswith("sk-ant-oat"):
            self.client = anthropic.AsyncAnthropic(
                api_key=None,
                auth_token=token,
                default_headers={"anthropic-beta": "oauth-2025-04-20"},
            )
        else:
            self.client = anthropic.AsyncAnthropic(api_key=token)
        self.model = config.anthropic_model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using the Anthropic API."""
        # No temperature pin here: claude-opus-4-7 and later reject sampling
        # params (400), so this provider's ranking is not deterministic.
        kwargs = {
            "model": self.model,
            "max_tokens": 16000,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        logger.debug(
            "LLM generate input (anthropic model=%s): system=%s prompt=%s",
            self.model,
            system_prompt,
            prompt,
        )

        response = await self.client.messages.create(**kwargs)

        output = next((b.text for b in response.content if b.type == "text"), "")
        logger.debug("LLM generate output (anthropic model=%s): %s", self.model, output)
        return output

    async def embed(self, text: str) -> list[float]:
        """Anthropic API does not offer embeddings."""
        raise NotImplementedError(
            "AnthropicProvider does not support embed(). "
            "Use ollama or bedrock for the embedding batch job."
        )
