"""Tool for resolving video IDs to titles.

Reads pgvector.video_embeddings, which only covers videos that have been
embedded. Unresolved IDs are simply absent from the result.
"""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)


async def get_video_titles(video_ids: list[str]) -> dict[str, str]:
    """Return {video_id: title} for the IDs that have a known title."""
    if not video_ids:
        return {}
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT video_id, title FROM video_embeddings WHERE video_id = ANY($1::text[])",
                video_ids,
            )
        return {row["video_id"]: row["title"] for row in rows if row["title"]}
    except Exception:
        logger.warning("Failed to resolve video titles")
        return {}
