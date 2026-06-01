"""Tests for trending tool — reads from precomputed trending_videos table."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.trending import get_trending_videos


def _acquire_ctx(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    return ctx


@pytest.mark.asyncio
class TestGetTrendingVideos:
    @patch("src.tools.trending.get_pool", new_callable=AsyncMock)
    async def test_returns_rows_in_rank_order(self, mock_pool):
        conn = AsyncMock()
        conn.fetch.return_value = [
            {"video_id": "v1", "title": "T1", "description": "D1", "watch_count": 100},
            {"video_id": "v2", "title": "T2", "description": None, "watch_count": 50},
        ]
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_pool.return_value = pool

        result = await get_trending_videos(limit=10)
        assert len(result) == 2
        assert result[0] == {
            "video_id": "v1", "title": "T1", "description": "D1", "watch_count": 100,
        }
        # NULL description is normalized to empty string for downstream consumers
        assert result[1]["description"] == ""

    @patch("src.tools.trending.get_pool", new_callable=AsyncMock)
    async def test_empty_table_returns_empty(self, mock_pool):
        conn = AsyncMock()
        conn.fetch.return_value = []
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_pool.return_value = pool

        assert await get_trending_videos() == []

    @patch("src.tools.trending.get_pool", new_callable=AsyncMock)
    async def test_swallows_pool_errors(self, mock_pool):
        mock_pool.side_effect = RuntimeError("pool down")
        assert await get_trending_videos() == []
