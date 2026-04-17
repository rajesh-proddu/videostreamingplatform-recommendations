"""Tests for the recommendation graph."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
@patch("src.agent.graph.recommendation_graph")
async def test_get_recommendations_returns_limited_results(mock_graph):
    from src.agent.graph import get_recommendations

    mock_result = MagicMock()
    mock_result.ranked_results = [
        {"video_id": f"vid-{i}", "score": 0.9 - i * 0.1, "title": f"V{i}", "reason": "good"}
        for i in range(5)
    ]
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)

    results = await get_recommendations("user-1", limit=3)

    assert len(results) == 3
    assert results[0]["video_id"] == "vid-0"


@pytest.mark.asyncio
@patch("src.agent.graph.recommendation_graph")
async def test_get_recommendations_with_query(mock_graph):
    from src.agent.graph import get_recommendations

    mock_result = MagicMock()
    mock_result.ranked_results = [{"video_id": "vid-1", "score": 0.9, "title": "T", "reason": "r"}]
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)

    results = await get_recommendations("user-1", query="python", limit=10)

    assert len(results) == 1
    call_args = mock_graph.ainvoke.call_args[0][0]
    assert call_args.query == "python"
    assert call_args.user_id == "user-1"


@pytest.mark.asyncio
@patch("src.agent.graph.recommendation_graph")
async def test_get_recommendations_empty_results(mock_graph):
    from src.agent.graph import get_recommendations

    mock_result = MagicMock()
    mock_result.ranked_results = []
    mock_graph.ainvoke = AsyncMock(return_value=mock_result)

    results = await get_recommendations("user-1")

    assert results == []
