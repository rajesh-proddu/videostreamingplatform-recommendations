"""pgvector embedding store."""

import logging
from typing import Optional

import asyncpg

from src.config import config

logger = logging.getLogger(__name__)


async def find_similar_in_pool(
    pool: asyncpg.Pool, embedding: list[float], limit: int = 10
) -> list[dict]:
    """Nearest-neighbor search over video_embeddings using a caller-supplied pool.

    Extracted so the recommendations API tools (which share an asyncpg pool via
    db.get_pool) can call the same ANN query that the consumer/batch jobs use
    through EmbeddingStore, without having to instantiate or initialize a store.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT video_id, title, description,
                   1 - (embedding <=> $1::vector) as similarity
            FROM video_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            str(embedding), limit,
        )
        return [
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "description": row["description"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]


class EmbeddingStore:
    """Store and retrieve video embeddings using pgvector."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Create connection pool and ensure schema exists.

        Note: an earlier revision created a `watch_history` table here. That
        table is no longer read by any tool — watch events live in Iceberg
        (analytics warehouse) and the per-user fields (recent_watches, user_vec)
        are precomputed nightly into `user_features`. The old table is left in
        place if present; an explicit DROP from API startup would be unsafe on
        rollback. Drop manually via psql if you want the storage back.
        """
        self.pool = await asyncpg.create_pool(config.pgvector_url)
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS video_embeddings (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    embedding vector({config.embedding_dimension}),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # Precomputed by analytics/feature-jobs/user_features (daily CronJob).
            # user_vec is nullable: a user with no embedded watches has no vector.
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS user_features (
                    user_id TEXT PRIMARY KEY,
                    user_vec vector({config.embedding_dimension}),
                    recent_watches TEXT[] NOT NULL DEFAULT '{{}}',
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            # Precomputed by analytics/feature-jobs/trending (hourly CronJob).
            # Rewritten atomically (TRUNCATE + INSERT in a tx); rank is dense 1..N.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trending_videos (
                    rank INT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    watch_count BIGINT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def store_embedding(self, video_id: str, title: str, description: str, embedding: list[float]):
        """Store a video embedding."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO video_embeddings (video_id, title, description, embedding, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (video_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                video_id, title, description, str(embedding),
            )

    async def find_similar(self, embedding: list[float], limit: int = 10) -> list[dict]:
        """Find similar videos by embedding similarity (uses self.pool)."""
        return await find_similar_in_pool(self.pool, embedding, limit)

    async def delete_embeddings(self, video_ids: list[str]):
        """Delete embedding rows by video_id (idempotent — missing rows are no-ops)."""
        if not video_ids:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM video_embeddings WHERE video_id = ANY($1::text[])",
                video_ids,
            )

    async def close(self):
        if self.pool:
            await self.pool.close()
