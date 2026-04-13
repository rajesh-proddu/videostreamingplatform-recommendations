"""Tool for querying user watch history."""

import logging

import asyncpg

from src.config import config

logger = logging.getLogger(__name__)


async def get_user_history(user_id: str, limit: int = 50) -> list[str]:
    """Get recently watched video IDs for a user from pgvector DB."""
    try:
        conn = await asyncpg.connect(config.pgvector_url)
        try:
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
        finally:
            await conn.close()
    except Exception:
        logger.warning(f"Failed to get watch history for user {user_id}")
        return []
