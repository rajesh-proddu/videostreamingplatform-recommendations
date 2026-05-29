"""Tests for the similar-videos tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.similar import USER_VECTOR_HISTORY_LIMIT, get_similar_videos


@pytest.mark.asyncio
async def test_get_similar_videos_empty_history_short_circuits():
    # Should not even touch the pool.
    with patch("src.tools.similar.get_pool", new_callable=AsyncMock) as mock_pool:
        result = await get_similar_videos([])
    assert result == []
    mock_pool.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.similar.get_pool", new_callable=AsyncMock)
async def test_get_similar_videos_returns_rows(mock_get_pool):
    rows = [
        {"video_id": "sim-1", "title": "T1", "description": "D1"},
        {"video_id": "sim-2", "title": "T2", "description": None},
    ]
    conn = AsyncMock()
    conn.fetch.return_value = rows

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
    mock_get_pool.return_value = pool

    result = await get_similar_videos(["w-1", "w-2"], limit=5)

    assert result == [
        {"video_id": "sim-1", "title": "T1", "description": "D1"},
        {"video_id": "sim-2", "title": "T2", "description": ""},
    ]
    args = conn.fetch.call_args[0]
    assert args[1] == ["w-1", "w-2"]
    assert args[2] == 5


@pytest.mark.asyncio
@patch("src.tools.similar.get_pool", new_callable=AsyncMock)
async def test_get_similar_videos_trims_history_to_limit(mock_get_pool):
    conn = AsyncMock()
    conn.fetch.return_value = []
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_acquire_ctx(conn))
    mock_get_pool.return_value = pool

    history = [f"v-{i}" for i in range(USER_VECTOR_HISTORY_LIMIT + 10)]
    await get_similar_videos(history)

    sent_history = conn.fetch.call_args[0][1]
    assert sent_history == history[:USER_VECTOR_HISTORY_LIMIT]


@pytest.mark.asyncio
@patch("src.tools.similar.get_pool", new_callable=AsyncMock)
async def test_get_similar_videos_swallows_errors(mock_get_pool):
    mock_get_pool.side_effect = RuntimeError("pool unavailable")
    assert await get_similar_videos(["w-1"]) == []


def _acquire_ctx(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    return ctx
