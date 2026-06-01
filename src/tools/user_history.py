"""Tool for querying a user's recent watches.

Reads from pgvector.user_features.recent_watches — a precomputed array of the
last ~20 distinct video IDs the user watched, written daily by the analytics
feature-jobs CronJob. Returns [] for unknown users (cold start).
"""

import logging

from src.db import get_pool

logger = logging.getLogger(__name__)


async def get_user_history(user_id: str, limit: int = 50) -> list[str]:
    """Return recent video IDs for a user from precomputed user_features."""
    if not user_id:
        return []
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT recent_watches FROM user_features WHERE user_id = $1",
                user_id,
            )
            if row is None or row["recent_watches"] is None:
                return []
            return list(row["recent_watches"])[:limit]
    except Exception:
        logger.warning(f"Failed to get watch history for user {user_id}")
        return []
