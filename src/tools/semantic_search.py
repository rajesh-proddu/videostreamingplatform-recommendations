"""Semantic search tool — embed the query and find nearest videos by ANN.

Complements the keyword ES search:
  - ES guarantees every video is findable by exact terms (ES population is
    synchronous from the kafka-es-consumer).
  - Semantic search adds relevance where embeddings exist (embedding population
    is lazy through the kafka → pgvector embeddings consumer).

Both run side-by-side when state.query is set.
"""

import logging

from src.db import get_pool
from src.embeddings.store import find_similar_in_pool
from src.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)


async def semantic_search(query: str, limit: int = 20) -> list[dict]:
    """Embed the query and return videos by vector similarity."""
    if not query:
        return []
    try:
        llm = get_llm_provider()
        query_vec = await llm.embed(query)
        pool = await get_pool()
        return await find_similar_in_pool(pool, query_vec, limit=limit)
    except NotImplementedError:
        # AnthropicProvider has no embeddings — skip silently, ES still covers the query.
        logger.info("LLM provider does not support embeddings; skipping semantic search")
        return []
    except Exception:
        logger.warning("Failed semantic search for query: %s", query)
        return []
