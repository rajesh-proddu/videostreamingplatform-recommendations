"""Tool for getting trending videos.

Reads pgvector.trending_videos — a precomputed top-N list rewritten hourly by
the analytics feature-jobs trending CronJob. Returns rows in rank order; no
on-request aggregation.
"""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)


async def get_trending_videos(hours: int = 24, limit: int = 20) -> list[dict]:
    """Return top trending videos from the precomputed table.

    `hours` is kept in the signature for callers, but the window is fixed by
    the batch job (currently 24h). Callers passing a smaller value get a
    capped slice of the same precomputed list, not a tighter aggregation.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT video_id, title, description, watch_count
                FROM trending_videos
                ORDER BY rank
                LIMIT $1
                """,
                limit,
            )
            return [
                {
                    "video_id": row["video_id"],
                    "title": row["title"] or "",
                    "description": row["description"] or "",
                    "watch_count": int(row["watch_count"]),
                }
                for row in rows
            ]
    except Exception:
        logger.warning("Failed to get trending videos")
        return []
