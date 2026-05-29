"""Ollama LLM provider for local development."""

import logging
from typing import Optional

import httpx

from src.config import config
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider using Ollama for local inference."""

    def __init__(self):
        self.base_url = config.ollama_base_url
        self.model = config.ollama_model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info("LLM generate input (ollama model=%s): %s", self.model, messages)

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        output = response.json()["message"]["content"]
        logger.info("LLM generate output (ollama model=%s): %s", self.model, output)
        return output

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Ollama."""
        logger.info("LLM embed input (ollama model=%s): %s", self.model, text)
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        logger.info("LLM embed output (ollama model=%s): dim=%d", self.model, len(embedding))
        return embedding
