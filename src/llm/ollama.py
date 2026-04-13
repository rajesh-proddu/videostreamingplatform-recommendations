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

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using Ollama."""
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        return response.json()["embedding"]
