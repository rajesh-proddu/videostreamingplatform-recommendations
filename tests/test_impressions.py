"""Tests for the impression log."""

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.agent.impressions as impressions
from src.agent.graph import get_recommendations
from src.agent.state import VideoCandidate


def _pool_with(conn):
    ctx = AsyncMock()
    ctx.__aenter__.return_value = conn
    ctx.__aexit__.return_value = None
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    return pool


def _final_state(**overrides):
    state = {
        "user_id": "u1",
        "query": "rust",
        "route": "rank",
        "prompt_version": "2",
        "rank_fallback": None,
        "candidates": [
            VideoCandidate(video_id="v1", title="A", source="search"),
            VideoCandidate(video_id="v2", title="B", source="similar"),
        ],
    }
    state.update(overrides)
    return state


SERVED = [
    {"video_id": "v2", "title": "B", "score": 0.9, "reason": "r"},
    {"video_id": "v1", "title": "A", "score": 0.4, "reason": "r"},
]


def test_build_row_llm_route():
    with patch.object(impressions.config, "llm_provider", "bedrock"), \
         patch.object(impressions.config, "bedrock_model_id", "model-x"):
        row = impressions._build_row("rid", _final_state(), SERVED, 123)

    request_id, user_id, query, route, prompt_version, model_id, fallback, latency, items = row
    assert (request_id, user_id, query, route, prompt_version, model_id, fallback, latency) == (
        "rid", "u1", "rust", "rank", "2", "model-x", None, 123,
    )
    assert json.loads(items) == [
        {"video_id": "v2", "rank": 1, "score": 0.9, "source": "similar"},
        {"video_id": "v1", "rank": 2, "score": 0.4, "source": "search"},
    ]


def test_build_row_non_llm_routes_have_no_model():
    row = impressions._build_row("rid", _final_state(route="rank_deterministic", prompt_version=None), SERVED, 1)
    assert row[3] == "rank_deterministic"
    assert row[5] is None


def test_build_row_popular_fallback_source():
    row = impressions._build_row(
        "rid", _final_state(route="popular_fallback", candidates=[]), [{"video_id": "p1", "score": 0.5}], 1,
    )
    assert json.loads(row[8]) == [{"video_id": "p1", "rank": 1, "score": 0.5, "source": "popular"}]


@pytest.mark.asyncio
@patch("src.agent.impressions.get_pool", new_callable=AsyncMock)
async def test_record_impression_writes_row(mock_get_pool):
    conn = AsyncMock()
    mock_get_pool.return_value = _pool_with(conn)
    request_id = str(uuid.uuid4())

    impressions.record_impression(request_id, _final_state(), SERVED, 5)
    await impressions.drain()

    sql, *args = conn.execute.call_args[0]
    assert "INSERT INTO recommendation_impressions" in sql
    assert args[0] == request_id
    assert not impressions._pending


@pytest.mark.asyncio
@patch("src.agent.impressions.record_impression_write_failure")
@patch("src.agent.impressions.get_pool", new_callable=AsyncMock)
async def test_failed_write_is_counted_not_raised(mock_get_pool, mock_failure):
    mock_get_pool.side_effect = ConnectionError("pg down")

    impressions.record_impression("rid", _final_state(), SERVED, 5)
    await impressions.drain()

    mock_failure.assert_called_once()
    assert not impressions._pending


@pytest.mark.asyncio
@patch("src.agent.impressions.get_pool", new_callable=AsyncMock)
async def test_ensure_schema_swallows_errors(mock_get_pool):
    mock_get_pool.side_effect = ConnectionError("pg down")
    await impressions.ensure_schema()  # must not raise


@pytest.mark.asyncio
@patch("src.agent.impressions.get_pool", new_callable=AsyncMock)
async def test_ensure_schema_creates_table(mock_get_pool):
    conn = AsyncMock()
    mock_get_pool.return_value = _pool_with(conn)
    await impressions.ensure_schema()
    statements = [c[0][0] for c in conn.execute.call_args_list]
    assert any("CREATE TABLE IF NOT EXISTS recommendation_impressions" in s for s in statements)


@pytest.mark.asyncio
@patch("src.agent.graph.record_impression")
@patch("src.agent.graph.recommendation_graph")
async def test_graph_records_impression_only_with_request_id(mock_graph, mock_record):
    final = {"ranked_results": SERVED, "user_id": "u1"}
    mock_graph.ainvoke = AsyncMock(return_value=final)

    await get_recommendations("u1", limit=1)
    mock_record.assert_not_called()

    results = await get_recommendations("u1", limit=1, request_id="rid")
    request_id, state, served, latency_ms = mock_record.call_args[0]
    assert (request_id, state, served) == ("rid", final, results)
    assert served == SERVED[:1]
    assert latency_ms >= 0


@pytest.mark.asyncio
@patch("src.agent.impressions.get_pool", new_callable=AsyncMock)
@patch("src.agent.graph.recommendation_graph")
async def test_slow_or_failing_write_does_not_block_response(mock_graph, mock_get_pool):
    mock_graph.ainvoke = AsyncMock(return_value={"ranked_results": SERVED, "user_id": "u1"})
    started = asyncio.Event()

    async def slow_fail():
        started.set()
        await asyncio.sleep(0.05)
        raise ConnectionError("pg down")

    mock_get_pool.side_effect = slow_fail

    results = await get_recommendations("u1", request_id="rid")
    assert results == SERVED
    assert impressions._pending  # still in flight after the response is ready
    await impressions.drain()
    assert started.is_set()
    assert not impressions._pending
