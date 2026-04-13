"""Batch job to compute and store video embeddings."""

import asyncio
import logging

from elasticsearch import AsyncElasticsearch

from src.config import config
from src.embeddings.store import EmbeddingStore
from src.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)


async def embed_all_videos():
    """Fetch all videos from ES and compute/store embeddings."""
    store = EmbeddingStore()
    await store.initialize()

    es = AsyncElasticsearch(config.elasticsearch_url)
    llm = get_llm_provider()

    try:
        # Scroll through all videos in ES
        response = await es.search(
            index=config.es_video_index,
            body={"query": {"match_all": {}}, "size": 100},
            scroll="2m",
        )

        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        total = 0

        while hits:
            for hit in hits:
                video_id = hit["_id"]
                source = hit["_source"]
                title = source.get("title", "")
                description = source.get("description", "")

                text = f"{title}. {description}"
                try:
                    embedding = await llm.embed(text)
                    await store.store_embedding(video_id, title, description, embedding)
                    total += 1
                except Exception:
                    logger.warning(f"Failed to embed video {video_id}")

            response = await es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

        logger.info(f"Embedded {total} videos")

    finally:
        await es.close()
        await store.close()


if __name__ == "__main__":
    asyncio.run(embed_all_videos())
