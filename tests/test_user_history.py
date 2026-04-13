"""Tests for user history tool."""

from unittest.mock import AsyncMock, patch

import pytest

from src.tools.user_history import get_user_history


@pytest.mark.asyncio
class TestGetUserHistory:
    @patch("src.tools.user_history.asyncpg")
    async def test_get_user_history_returns_videos(self, mock_asyncpg):
        mock_conn = AsyncMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
        mock_conn.fetch = AsyncMock(return_value=[
            {"video_id": "vid-1"},
            {"video_id": "vid-2"},
            {"video_id": "vid-3"},
        ])

        result = await get_user_history("user-1", limit=10)
        assert result == ["vid-1", "vid-2", "vid-3"]
        mock_conn.fetch.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("src.tools.user_history.asyncpg")
    async def test_get_user_history_empty(self, mock_asyncpg):
        mock_conn = AsyncMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)
        mock_conn.fetch = AsyncMock(return_value=[])

        result = await get_user_history("user-1")
        assert result == []
