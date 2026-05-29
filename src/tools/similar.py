"""Tool for finding videos similar to a user's watch history via pgvector."""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)

USER_VECTOR_HISTORY_LIMIT = 20

# Builds a user vector by averaging embeddings of the most recently watched
# videos, then returns nearest neighbors excluding anything already watched.
# AVG over an empty set returns NULL, which the WHERE clause filters out.
_SIMILAR_VIDEOS_SQL = """
WITH user_vec AS (
    SELECT AVG(embedding) AS v
    FROM video_embeddings
    WHERE video_id = ANY($1::text[])
)
SELECT video_id, title, description
FROM video_embeddings, user_vec
WHERE user_vec.v IS NOT NULL
  AND video_id != ALL($1::text[])
ORDER BY embedding <=> user_vec.v
LIMIT $2
"""


async def get_similar_videos(watch_history: list[str], limit: int = 20) -> list[dict]:
    """Return videos similar to the user's recent watch history."""
    if not watch_history:
        return []
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                _SIMILAR_VIDEOS_SQL,
                watch_history[:USER_VECTOR_HISTORY_LIMIT],
                limit,
            )
            return [
                {
                    "video_id": row["video_id"],
                    "title": row["title"] or "",
                    "description": row["description"] or "",
                }
                for row in rows
            ]
    except Exception:
        logger.warning("Failed to get similar videos")
        return []
