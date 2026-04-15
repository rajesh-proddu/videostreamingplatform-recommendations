"""Shared asyncpg connection pool for pgvector."""

import logging
from typing import Optional

import asyncpg

from src.config import config

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the shared connection pool."""
    global _pool
    if _pool is None or _pool._closed:
        _pool = await asyncpg.create_pool(
            config.pgvector_url,
            min_size=2,
            max_size=20,
        )
        logger.info("Created pgvector connection pool (min=2, max=20)")
    return _pool


async def close_pool() -> None:
    """Close the shared connection pool."""
    global _pool
    if _pool is not None and not _pool._closed:
        await _pool.close()
        _pool = None
        logger.info("Closed pgvector connection pool")
