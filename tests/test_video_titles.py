"""Tests for the video_titles tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.video_titles import get_video_titles


def _pool_with(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    return pool


@pytest.mark.asyncio
class TestGetVideoTitles:
    @patch("src.tools.video_titles.get_pool", new_callable=AsyncMock)
    async def test_maps_ids_to_titles(self, mock_get_pool):
        conn = AsyncMock()
        conn.fetch.return_value = [
            {"video_id": "v1", "title": "One"},
            {"video_id": "v2", "title": None},
        ]
        mock_get_pool.return_value = _pool_with(conn)

        result = await get_video_titles(["v1", "v2", "v3"])

        assert result == {"v1": "One"}
        assert conn.fetch.call_args[0][1] == ["v1", "v2", "v3"]

    @patch("src.tools.video_titles.get_pool", new_callable=AsyncMock)
    async def test_empty_ids_skip_query(self, mock_get_pool):
        assert await get_video_titles([]) == {}
        mock_get_pool.assert_not_called()

    @patch("src.tools.video_titles.get_pool", new_callable=AsyncMock)
    async def test_db_error_returns_empty(self, mock_get_pool):
        mock_get_pool.side_effect = Exception("db down")
        assert await get_video_titles(["v1"]) == {}
