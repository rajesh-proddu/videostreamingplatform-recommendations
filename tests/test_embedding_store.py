"""Tests for the EmbeddingStore."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.embeddings.store import EmbeddingStore


@pytest.mark.asyncio
@patch("src.embeddings.store.asyncpg.create_pool", new_callable=AsyncMock)
async def test_initialize_creates_tables(mock_create_pool):
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_create_pool.return_value = mock_pool

    store = EmbeddingStore()
    await store.initialize()

    assert store.pool is mock_pool
    # Should execute: CREATE EXTENSION, CREATE TABLE video_embeddings, CREATE TABLE watch_history, CREATE INDEX
    assert mock_conn.execute.call_count == 4


@pytest.mark.asyncio
async def test_store_embedding():
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    store = EmbeddingStore()
    store.pool = mock_pool

    await store.store_embedding("vid-1", "Title", "Desc", [0.1, 0.2, 0.3])

    mock_conn.execute.assert_called_once()
    args = mock_conn.execute.call_args
    assert "vid-1" in args[0]
    assert "Title" in args[0]


@pytest.mark.asyncio
async def test_find_similar():
    mock_row1 = {"video_id": "vid-1", "title": "A", "description": "Desc A", "similarity": 0.95}
    mock_row2 = {"video_id": "vid-2", "title": "B", "description": "Desc B", "similarity": 0.80}

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = [mock_row1, mock_row2]
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    store = EmbeddingStore()
    store.pool = mock_pool

    results = await store.find_similar([0.1, 0.2], limit=5)

    assert len(results) == 2
    assert results[0]["video_id"] == "vid-1"
    assert results[0]["similarity"] == 0.95
    assert results[1]["video_id"] == "vid-2"


@pytest.mark.asyncio
async def test_close():
    mock_pool = AsyncMock()
    store = EmbeddingStore()
    store.pool = mock_pool

    await store.close()
    mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_no_pool():
    store = EmbeddingStore()
    store.pool = None
    await store.close()  # should not raise
