"""Tests for user history tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.user_history import get_user_history


@pytest.mark.asyncio
class TestGetUserHistory:
    @patch("src.tools.user_history.get_pool")
    async def test_get_user_history_returns_videos(self, mock_get_pool):
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[
            {"video_id": "vid-1"},
            {"video_id": "vid-2"},
            {"video_id": "vid-3"},
        ])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        result = await get_user_history("user-1", limit=10)
        assert result == ["vid-1", "vid-2", "vid-3"]
        mock_conn.fetch.assert_called_once()

    @patch("src.tools.user_history.get_pool")
    async def test_get_user_history_empty(self, mock_get_pool):
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_get_pool.return_value = mock_pool

        result = await get_user_history("user-1")
        assert result == []
