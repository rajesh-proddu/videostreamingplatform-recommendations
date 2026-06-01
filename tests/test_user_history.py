"""Tests for user_history tool — reads from precomputed user_features."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.user_history import get_user_history


def _acquire_ctx(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    return ctx


@pytest.mark.asyncio
class TestGetUserHistory:
    @patch("src.tools.user_history.get_pool", new_callable=AsyncMock)
    async def test_returns_recent_watches_array(self, mock_get_pool):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"recent_watches": ["v1", "v2", "v3"]}
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_get_pool.return_value = pool

        result = await get_user_history("user-1", limit=10)
        assert result == ["v1", "v2", "v3"]

    @patch("src.tools.user_history.get_pool", new_callable=AsyncMock)
    async def test_truncates_to_limit(self, mock_get_pool):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"recent_watches": [f"v{i}" for i in range(20)]}
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_get_pool.return_value = pool

        result = await get_user_history("user-1", limit=5)
        assert result == ["v0", "v1", "v2", "v3", "v4"]

    @patch("src.tools.user_history.get_pool", new_callable=AsyncMock)
    async def test_missing_user_returns_empty(self, mock_get_pool):
        conn = AsyncMock()
        conn.fetchrow.return_value = None
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_get_pool.return_value = pool

        result = await get_user_history("nobody")
        assert result == []

    async def test_empty_user_id_short_circuits(self):
        # Should not touch the pool at all.
        with patch("src.tools.user_history.get_pool", new_callable=AsyncMock) as p:
            result = await get_user_history("")
        assert result == []
        p.assert_not_called()

    @patch("src.tools.user_history.get_pool", new_callable=AsyncMock)
    async def test_swallows_pool_errors(self, mock_get_pool):
        mock_get_pool.side_effect = RuntimeError("pool down")
        assert await get_user_history("user-1") == []
