"""LLM provider abstraction."""

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
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
    return _provider_instance
