"""Tool for querying user watch history."""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)


async def get_user_history(user_id: str, limit: int = 50) -> list[str]:
    """Get recently watched video IDs for a user from pgvector DB."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT video_id
                FROM watch_history
                WHERE user_id = $1
                ORDER BY watched_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            return [row["video_id"] for row in rows]
    except Exception:
        logger.warning(f"Failed to get watch history for user {user_id}")
        return []
