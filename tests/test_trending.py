"""Tests for trending videos tool."""

from unittest.mock import AsyncMock, patch

import pytest

from src.tools.trending import get_trending_videos


@pytest.mark.asyncio
class TestGetTrendingVideos:
    @patch("src.tools.trending.asyncpg")
    async def test_get_trending_returns_videos(self, mock_asyncpg):
        mock_conn = AsyncMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
        mock_conn.fetch = AsyncMock(return_value=[
            {"video_id": "vid-1", "watch_count": 100},
            {"video_id": "vid-2", "watch_count": 50},
        ])

        result = await get_trending_videos(hours=24, limit=10)
        assert len(result) == 2
        assert result[0] == {"video_id": "vid-1", "watch_count": 100}
        assert result[1] == {"video_id": "vid-2", "watch_count": 50}
        mock_conn.close.assert_called_once()

    @patch("src.tools.trending.asyncpg")
    async def test_get_trending_empty(self, mock_asyncpg):
        mock_conn = AsyncMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
        mock_conn.fetch = AsyncMock(return_value=[])

        result = await get_trending_videos()
        assert result == []
