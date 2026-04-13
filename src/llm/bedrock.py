"""AWS Bedrock LLM provider for production."""

import json
import logging
from typing import Optional

from src.config import config
from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """LLM provider using AWS Bedrock."""

    def __init__(self):
        import boto3
        self.bedrock = boto3.client(
            "bedrock-runtime",
            region_name=config.bedrock_region,
        )
        self.model_id = config.bedrock_model_id

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text using AWS Bedrock."""
        import asyncio

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0.7},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        # Run synchronous boto3 call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.converse(**kwargs),
        )

        return response["output"]["message"]["content"][0]["text"]

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using AWS Bedrock Titan."""
        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": text}),
            ),
        )

        result = json.loads(response["body"].read())
        return result["embedding"]
