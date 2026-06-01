"""Tests for the semantic_search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.semantic_search import semantic_search


@pytest.mark.asyncio
async def test_empty_query_short_circuits():
    with patch("src.tools.semantic_search.get_llm_provider") as p:
        result = await semantic_search("")
    assert result == []
    p.assert_not_called()


@pytest.mark.asyncio
@patch("src.tools.semantic_search.find_similar_in_pool", new_callable=AsyncMock)
@patch("src.tools.semantic_search.get_pool", new_callable=AsyncMock)
@patch("src.tools.semantic_search.get_llm_provider")
async def test_embeds_query_and_calls_find_similar(mock_get_llm, mock_pool, mock_find):
    llm = MagicMock()
    llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    mock_get_llm.return_value = llm
    pool = MagicMock()
    mock_pool.return_value = pool
    mock_find.return_value = [{"video_id": "v1", "title": "T", "description": "D"}]

    result = await semantic_search("cats", limit=5)

    llm.embed.assert_awaited_once_with("cats")
    mock_find.assert_awaited_once_with(pool, [0.1, 0.2, 0.3], limit=5)
    assert result[0]["video_id"] == "v1"


@pytest.mark.asyncio
@patch("src.tools.semantic_search.get_llm_provider")
async def test_provider_without_embed_returns_empty(mock_get_llm):
    llm = MagicMock()
    llm.embed = AsyncMock(side_effect=NotImplementedError("no embed"))
    mock_get_llm.return_value = llm

    assert await semantic_search("cats") == []


@pytest.mark.asyncio
@patch("src.tools.semantic_search.get_llm_provider")
async def test_embed_failure_swallowed(mock_get_llm):
    llm = MagicMock()
    llm.embed = AsyncMock(side_effect=RuntimeError("LLM down"))
    mock_get_llm.return_value = llm

    assert await semantic_search("cats") == []
