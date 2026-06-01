"""LLM provider abstraction."""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from src.config import config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts concurrently. Fan-out is capped by
        config.max_concurrent_embeds so we don't swamp Ollama or hit Bedrock
        throughput limits. Providers with a native batch API can override."""
        if not texts:
            return []
        sem = asyncio.Semaphore(config.max_concurrent_embeds)

        async def _one(text: str) -> list[float]:
            async with sem:
                return await self.embed(text)

        return await asyncio.gather(*(_one(t) for t in texts))


_provider_instance: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider singleton."""
    global _provider_instance
    if _provider_instance is None:
        if config.llm_provider == "ollama":
            from src.llm.ollama import OllamaProvider
            _provider_instance = OllamaProvider()
        elif config.llm_provider == "bedrock":
            from src.llm.bedrock import BedrockProvider
            _provider_instance = BedrockProvider()
        elif config.llm_provider == "anthropic":
            from src.llm.anthropic import AnthropicProvider
            _provider_instance = AnthropicProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
    return _provider_instance
