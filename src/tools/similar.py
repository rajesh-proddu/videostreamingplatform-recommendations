"""Tool for finding videos similar to a user's taste vector.

Reads pgvector.user_features.user_vec (precomputed nightly) and runs an ANN
query against video_embeddings via EmbeddingStore.find_similar (the shared
primitive — see src/embeddings/store.py:find_similar_in_pool).

Returns [] for cold-start users (no user_features row, or user_vec is NULL
because none of their watched videos have embeddings yet).
"""

import logging

from src.db import get_pool
from src.embeddings.store import find_similar_in_pool

logger = logging.getLogger(__name__)


def _parse_pgvector(text: str) -> list[float]:
    """Parse pgvector's text format ('[0.1,0.2,...]') into a list of floats."""
    return [float(x) for x in text.strip("[]").split(",") if x.strip()]


async def get_similar_videos(user_id: str, limit: int = 20) -> list[dict]:
    """Return videos similar to the user's precomputed taste vector."""
    if not user_id:
        return []
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_vec::text AS vec_text FROM user_features "
                "WHERE user_id = $1 AND user_vec IS NOT NULL",
                user_id,
            )
        if row is None or not row["vec_text"]:
            return []
        user_vec = _parse_pgvector(row["vec_text"])
        return await find_similar_in_pool(pool, user_vec, limit=limit)
    except Exception:
        logger.warning(f"Failed to get similar videos for {user_id}")
        return []
