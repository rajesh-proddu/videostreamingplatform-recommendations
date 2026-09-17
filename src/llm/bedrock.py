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

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            # temperature 0: ranking must be reproducible for evals to mean anything.
            "inferenceConfig": {"maxTokens": 4096, "temperature": 0},
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        logger.debug(
            "LLM generate input (bedrock model=%s): system=%s messages=%s",
            self.model_id,
            system_prompt,
            messages,
        )

        # Run synchronous boto3 call in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.converse(**kwargs),
        )

        output = response["output"]["message"]["content"][0]["text"]
        logger.debug("LLM generate output (bedrock model=%s): %s", self.model_id, output)
        return output

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings using AWS Bedrock Titan."""
        import asyncio

        embed_model_id = "amazon.titan-embed-text-v2:0"
        logger.info("LLM embed input (bedrock model=%s): %s", embed_model_id, text)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.invoke_model(
                modelId=embed_model_id,
                body=json.dumps({"inputText": text}),
            ),
        )

        result = json.loads(response["body"].read())
        embedding = result["embedding"]
        logger.info("LLM embed output (bedrock model=%s): dim=%d", embed_model_id, len(embedding))
        return embedding
