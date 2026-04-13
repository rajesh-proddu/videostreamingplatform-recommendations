"""Tool for searching videos in Elasticsearch."""

import logging

from elasticsearch import AsyncElasticsearch

from src.config import config

logger = logging.getLogger(__name__)


async def search_videos(query: str, limit: int = 20) -> list[dict]:
    """Search videos in Elasticsearch by title and description."""
    es = AsyncElasticsearch(config.elasticsearch_url)
    try:
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
    finally:
        await es.close()
