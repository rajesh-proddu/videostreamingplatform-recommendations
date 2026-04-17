"""Tests for the shared connection pool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.db as db_module


@pytest.fixture(autouse=True)
def reset_pool():
    """Reset the global pool singleton between tests."""
    db_module._pool = None
    yield
    db_module._pool = None


@pytest.mark.asyncio
@patch("src.db.asyncpg.create_pool", new_callable=AsyncMock)
async def test_get_pool_creates_pool(mock_create):
    mock_pool = MagicMock()
    mock_pool._closed = False
    mock_create.return_value = mock_pool

    pool = await db_module.get_pool()

    assert pool is mock_pool
    mock_create.assert_called_once()


@pytest.mark.asyncio
@patch("src.db.asyncpg.create_pool", new_callable=AsyncMock)
async def test_get_pool_reuses_existing(mock_create):
    mock_pool = MagicMock()
    mock_pool._closed = False
    mock_create.return_value = mock_pool

    pool1 = await db_module.get_pool()
    pool2 = await db_module.get_pool()

    assert pool1 is pool2
    mock_create.assert_called_once()  # only created once


@pytest.mark.asyncio
@patch("src.db.asyncpg.create_pool", new_callable=AsyncMock)
async def test_get_pool_recreates_if_closed(mock_create):
    mock_pool1 = MagicMock()
    mock_pool1._closed = False
    mock_pool2 = MagicMock()
    mock_pool2._closed = False
    mock_create.side_effect = [mock_pool1, mock_pool2]

    pool1 = await db_module.get_pool()
    mock_pool1._closed = True  # simulate pool closure

    pool2 = await db_module.get_pool()
    assert pool2 is mock_pool2
    assert mock_create.call_count == 2


@pytest.mark.asyncio
async def test_close_pool_when_open():
    mock_pool = AsyncMock()
    mock_pool._closed = False
    db_module._pool = mock_pool

    await db_module.close_pool()

    mock_pool.close.assert_called_once()
    assert db_module._pool is None


@pytest.mark.asyncio
async def test_close_pool_when_none():
    db_module._pool = None
    await db_module.close_pool()  # should not raise


@pytest.mark.asyncio
async def test_close_pool_when_already_closed():
    mock_pool = AsyncMock()
    mock_pool._closed = True
    db_module._pool = mock_pool

    await db_module.close_pool()
    mock_pool.close.assert_not_called()
