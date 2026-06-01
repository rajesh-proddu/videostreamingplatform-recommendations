"""Tests for similar tool — reads user_features.user_vec, calls find_similar_in_pool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.similar import _parse_pgvector, get_similar_videos


def _acquire_ctx(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    return ctx


class TestParsePgvector:
    def test_parses_standard_format(self):
        assert _parse_pgvector("[0.1,0.2,0.3]") == [0.1, 0.2, 0.3]

    def test_parses_spaces_and_negatives(self):
        assert _parse_pgvector("[-0.5, 0.25, 1.0]") == [-0.5, 0.25, 1.0]

    def test_empty_brackets_returns_empty(self):
        assert _parse_pgvector("[]") == []


@pytest.mark.asyncio
class TestGetSimilarVideos:
    async def test_empty_user_id_short_circuits(self):
        with patch("src.tools.similar.get_pool", new_callable=AsyncMock) as p:
            result = await get_similar_videos("")
        assert result == []
        p.assert_not_called()

    @patch("src.tools.similar.find_similar_in_pool", new_callable=AsyncMock)
    @patch("src.tools.similar.get_pool", new_callable=AsyncMock)
    async def test_returns_neighbors_for_user_with_vec(self, mock_pool, mock_find):
        conn = AsyncMock()
        conn.fetchrow.return_value = {"vec_text": "[0.1,0.2,0.3]"}
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_pool.return_value = pool

        mock_find.return_value = [
            {"video_id": "v1", "title": "T1", "description": "D1", "similarity": 0.9},
        ]

        result = await get_similar_videos("user-1", limit=5)

        mock_find.assert_awaited_once()
        called_pool, called_vec, *_ = mock_find.call_args.args
        assert called_pool is pool
        assert called_vec == [0.1, 0.2, 0.3]
        assert mock_find.call_args.kwargs.get("limit") == 5
        assert result[0]["video_id"] == "v1"

    @patch("src.tools.similar.find_similar_in_pool", new_callable=AsyncMock)
    @patch("src.tools.similar.get_pool", new_callable=AsyncMock)
    async def test_user_without_vec_returns_empty(self, mock_pool, mock_find):
        conn = AsyncMock()
        conn.fetchrow.return_value = None  # no row
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
        mock_pool.return_value = pool

        result = await get_similar_videos("cold-user")
        assert result == []
        mock_find.assert_not_called()

    @patch("src.tools.similar.get_pool", new_callable=AsyncMock)
    async def test_swallows_pool_errors(self, mock_pool):
        mock_pool.side_effect = RuntimeError("pool down")
        assert await get_similar_videos("user-1") == []
