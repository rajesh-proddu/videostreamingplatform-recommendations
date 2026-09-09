"""Tool for searching videos in Elasticsearch."""

import logging
from typing import Optional

from elasticsearch import AsyncElasticsearch

from src.config import config

logger = logging.getLogger(__name__)

_client: Optional[AsyncElasticsearch] = None


def get_es_client() -> AsyncElasticsearch:
    """Get or create the shared Elasticsearch client."""
    global _client
    if _client is None:
        _client = AsyncElasticsearch(config.elasticsearch_url)
    return _client


async def close_es_client() -> None:
    """Close the shared Elasticsearch client."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def search_videos(query: str, limit: int = 20) -> list[dict]:
    """Search videos in Elasticsearch by title and description."""
    es = get_es_client()
    response = await es.search(
        index=config.es_video_index,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "description"],
                    "fuzziness": "AUTO",
                }
            },
            "size": limit,
        },
    )
    return [
        {"id": hit["_id"], **hit["_source"]}
        for hit in response["hits"]["hits"]
    ]
