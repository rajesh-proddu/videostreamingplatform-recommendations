"""End-to-end tests for the LangGraph agent (retrieve → rank → filter).

External dependencies (pgvector, ES, LLM) are mocked. These tests verify the
node wiring — that state flows correctly and final results obey the
documented filter rules.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.graph import get_recommendations
from src.agent.state import AgentState


async def _run_graph(state: AgentState) -> list[dict]:
    return await get_recommendations(
        user_id=state.user_id,
        query=state.query,
        limit=state.limit,
    )


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_filters_watched_when_no_query(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = ["watched-1"]
    mock_trend.return_value = [
        {"video_id": "watched-1", "watch_count": 10},
        {"video_id": "fresh-1", "watch_count": 5},
    ]

    provider = AsyncMock()
    provider.generate.return_value = (
        '[{"video_id":"watched-1","score":0.9,"reason":"x"},'
        ' {"video_id":"fresh-1","score":0.8,"reason":"y"}]'
    )
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-graph", limit=5))
    ids = {r["video_id"] for r in results}
    assert "watched-1" not in ids
    assert "fresh-1" in ids


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_keeps_watched_when_query_present(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = ["watched-1"]
    mock_search.return_value = [
        {"id": "watched-1", "title": "Already Seen", "description": ""},
    ]
    mock_trend.return_value = []

    provider = AsyncMock()
    provider.generate.return_value = '[{"video_id":"watched-1","score":0.9,"reason":"match"}]'
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-graph", query="seen", limit=5))
    assert any(r["video_id"] == "watched-1" for r in results)


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_drops_low_scores(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = []
    mock_trend.return_value = [
        {"video_id": "low-1", "watch_count": 1},
        {"video_id": "high-1", "watch_count": 1},
    ]

    provider = AsyncMock()
    provider.generate.return_value = (
        '[{"video_id":"low-1","score":0.05,"reason":"x"},'
        ' {"video_id":"high-1","score":0.5,"reason":"y"}]'
    )
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-low", limit=10))
    ids = {r["video_id"] for r in results}
    assert "low-1" not in ids
    assert "high-1" in ids


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_truncates_to_limit(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = []
    mock_trend.return_value = [{"video_id": f"v-{i}", "watch_count": 1} for i in range(10)]

    provider = AsyncMock()
    provider.generate.return_value = (
        "["
        + ",".join(f'{{"video_id":"v-{i}","score":0.9,"reason":"r"}}' for i in range(10))
        + "]"
    )
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-trunc", limit=3))
    assert len(results) == 3


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_llm_failure_falls_back_to_default_score(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = []
    mock_trend.return_value = [{"video_id": "fb-1", "watch_count": 1}]

    provider = AsyncMock()
    provider.generate.side_effect = RuntimeError("llm offline")
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-llm-fail", limit=5))
    # Fallback assigns 0.5; filter threshold is 0.1 — item should survive.
    assert any(r["video_id"] == "fb-1" for r in results)


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_empty_retrieve_returns_empty(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = []
    mock_trend.return_value = []

    provider = AsyncMock()
    provider.generate.return_value = "[]"
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-empty", limit=5))
    assert results == []
