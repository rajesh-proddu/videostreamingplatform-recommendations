"""End-to-end tests for the LangGraph agent (retrieve → rank → filter).

External dependencies (pgvector, ES, LLM) are mocked. These tests verify the
node wiring — that state flows correctly and final results obey the
documented filter rules.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.agent.impressions as impressions
from src.agent.graph import get_recommendations
from src.agent.state import AgentState


async def _run_graph(state: AgentState) -> list[dict]:
    return await get_recommendations(
        user_id=state.user_id,
        query=state.query,
        limit=state.limit,
    )


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_filters_watched_when_no_query(
    mock_hist, mock_search, mock_trend, mock_llm, mock_similar,
):
    mock_hist.return_value = ["watched-1"]
    mock_similar.return_value = []
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
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_keeps_watched_when_query_present(
    mock_hist, mock_search, mock_trend, mock_llm, mock_similar,
):
    mock_hist.return_value = ["watched-1"]
    mock_search.return_value = [
        {"id": "watched-1", "title": "Already Seen", "description": ""},
    ]
    mock_similar.return_value = []
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
    # Query path so the LLM ranker runs and can emit a sub-threshold score.
    mock_hist.return_value = []
    mock_search.return_value = [
        {"id": "low-1", "title": "L", "description": ""},
        {"id": "high-1", "title": "H", "description": ""},
    ]
    mock_trend.return_value = []

    provider = AsyncMock()
    provider.generate.return_value = (
        '[{"video_id":"low-1","score":0.05,"reason":"x"},'
        ' {"video_id":"high-1","score":0.5,"reason":"y"}]'
    )
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-low", query="anything", limit=10))
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
    # Query path forces the LLM ranker, which is where the failure fallback lives.
    mock_hist.return_value = []
    mock_search.return_value = [{"id": "fb-1", "title": "F", "description": ""}]
    mock_trend.return_value = []

    provider = AsyncMock()
    provider.generate.side_effect = RuntimeError("llm offline")
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-llm-fail", query="x", limit=5))
    # Fallback assigns 0.5; filter threshold is 0.1 — item should survive.
    assert any(r["video_id"] == "fb-1" for r in results)


@pytest.mark.asyncio
@patch("src.agent.nodes.popular_fallback.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_empty_retrieve_routes_to_popular_fallback(
    mock_hist, mock_search, mock_trend, mock_fallback_trend,
):
    mock_hist.return_value = []
    mock_trend.return_value = []
    mock_fallback_trend.return_value = [{"video_id": "pop-1", "watch_count": 99}]

    results = await _run_graph(AgentState(user_id="u-empty", limit=5))
    assert [r["video_id"] for r in results] == ["pop-1"]


@pytest.mark.asyncio
@patch("src.agent.nodes.popular_fallback.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_empty_everywhere_returns_empty(
    mock_hist, mock_search, mock_trend, mock_fallback_trend,
):
    mock_hist.return_value = []
    mock_trend.return_value = []
    mock_fallback_trend.return_value = []

    results = await _run_graph(AgentState(user_id="u-void", limit=5))
    assert results == []


@pytest.mark.asyncio
@patch("src.agent.nodes.rank.get_llm_provider")
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_graph_no_query_skips_llm_ranker(mock_hist, mock_search, mock_trend, mock_llm):
    mock_hist.return_value = []
    mock_trend.return_value = [{"video_id": "t-1", "watch_count": 5}]

    provider = AsyncMock()
    mock_llm.return_value = provider

    results = await _run_graph(AgentState(user_id="u-feed", limit=5))
    assert any(r["video_id"] == "t-1" for r in results)
    provider.generate.assert_not_called()


# --- Impression log through the real graph ---------------------------------
# The ranking nodes set route/prompt_version/rank_fallback by mutating state;
# these confirm those fields survive into the dict ainvoke returns and land in
# the INSERT, rather than trusting hand-built state dicts.

def _captured_insert(conn) -> dict:
    _sql, *args = conn.execute.call_args[0]
    keys = ["request_id", "user_id", "query", "route", "prompt_version", "model_id",
            "rank_fallback", "latency_ms", "items"]
    row = dict(zip(keys, args))
    row["items"] = json.loads(row["items"])
    return row


async def _run_with_impression(query, llm_response, trending, fallback_trending=()):
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    provider = AsyncMock()
    provider.generate.return_value = llm_response
    request_id = str(uuid.uuid4())

    with patch("src.agent.nodes.retrieve.get_user_history", AsyncMock(return_value=[])), \
         patch("src.agent.nodes.retrieve.get_video_titles", AsyncMock(return_value={})), \
         patch("src.agent.nodes.retrieve.search_videos",
               AsyncMock(return_value=[{"id": "s-1", "title": "S"}])), \
         patch("src.agent.nodes.retrieve.semantic_search", AsyncMock(return_value=[])), \
         patch("src.agent.nodes.retrieve.get_similar_videos", AsyncMock(return_value=[])), \
         patch("src.agent.nodes.retrieve.get_trending_videos", AsyncMock(return_value=list(trending))), \
         patch("src.agent.nodes.popular_fallback.get_trending_videos",
               AsyncMock(return_value=list(fallback_trending))), \
         patch("src.agent.nodes.rank.get_llm_provider", return_value=provider), \
         patch("src.agent.impressions.get_pool", AsyncMock(return_value=pool)):
        results = await get_recommendations(user_id="u-imp", query=query, limit=5, request_id=request_id)
        await impressions.drain()

    row = _captured_insert(conn)
    assert row["request_id"] == request_id
    assert row["user_id"] == "u-imp"
    return results, row


@pytest.mark.asyncio
async def test_impression_llm_route():
    _, row = await _run_with_impression(
        "rust", '[{"video_id":"s-1","score":0.9,"reason":"r"}]', trending=[],
    )
    assert row["route"] == "rank"
    assert row["prompt_version"] == "2"
    assert row["rank_fallback"] is None
    assert row["query"] == "rust"
    assert row["items"] == [{"video_id": "s-1", "rank": 1, "score": 0.9, "source": "search"}]


@pytest.mark.asyncio
async def test_impression_records_rank_fallback():
    _, row = await _run_with_impression("rust", "not json", trending=[])
    assert row["route"] == "rank"
    assert row["rank_fallback"] == "invalid_json"


@pytest.mark.asyncio
async def test_impression_deterministic_route():
    _, row = await _run_with_impression(
        None, "unused", trending=[{"video_id": "t-1", "title": "T"}],
    )
    assert row["route"] == "rank_deterministic"
    assert row["prompt_version"] is None
    assert row["model_id"] is None
    assert row["items"][0]["source"] == "trending"


@pytest.mark.asyncio
async def test_impression_popular_fallback_route():
    results, row = await _run_with_impression(
        None, "unused", trending=[], fallback_trending=[{"video_id": "p-1", "title": "P"}],
    )
    assert [r["video_id"] for r in results] == ["p-1"]
    assert row["route"] == "popular_fallback"
    assert row["items"] == [{"video_id": "p-1", "rank": 1, "score": 0.5, "source": "popular"}]
