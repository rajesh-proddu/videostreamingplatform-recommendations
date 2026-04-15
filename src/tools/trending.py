"""Tool for getting trending videos."""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)


async def get_trending_videos(hours: int = 24, limit: int = 20) -> list[dict]:
    """Get trending videos based on recent watch counts."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT video_id, COUNT(*) as watch_count
                FROM watch_history
                WHERE watched_at > NOW() - INTERVAL '1 hour' * $1
                GROUP BY video_id
                ORDER BY watch_count DESC
                LIMIT $2
                """,
                hours,
                limit,
            )
            return [
                {"video_id": row["video_id"], "watch_count": row["watch_count"]}
                for row in rows
            ]
    except Exception:
        logger.warning("Failed to get trending videos")
        return []
