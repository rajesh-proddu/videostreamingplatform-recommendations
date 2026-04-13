"""pgvector embedding store."""

import logging
from typing import Optional

import asyncpg

from src.config import config

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store and retrieve video embeddings using pgvector."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Create connection pool and ensure schema exists."""
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
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS watch_history (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    watched_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_watch_history_user
                ON watch_history (user_id, watched_at DESC)
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
        """Find similar videos by embedding similarity."""
        async with self.pool.acquire() as conn:
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

    async def close(self):
        if self.pool:
            await self.pool.close()
