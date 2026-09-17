"""Tests for request-path agent metrics, read back through an in-memory reader."""

from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import src.agent.metrics as agent_metrics
from src.agent.graph import _route_after_retrieve
from src.agent.nodes.rank import rank_candidates
from src.agent.nodes.retrieve import retrieve_candidates
from src.agent.state import AgentState, VideoCandidate


@pytest.fixture
def reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    agent_metrics._instruments.cache_clear()
    with patch.object(agent_metrics, "get_meter", provider.get_meter):
        yield reader
    agent_metrics._instruments.cache_clear()


def _points(reader, name) -> dict[tuple, int]:
    out = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    for p in m.data.data_points:
                        out[tuple(sorted(p.attributes.items()))] = p.value
    return out


def _candidates():
    return [VideoCandidate(video_id="v1", title="T", source="search")]


def test_route_counter(reader):
    _route_after_retrieve(AgentState(user_id="u"))
    _route_after_retrieve(AgentState(user_id="u", candidates=_candidates()))
    _route_after_retrieve(AgentState(user_id="u", query="q", candidates=_candidates()))
    _route_after_retrieve(AgentState(user_id="u", query="q", candidates=_candidates()))

    assert _points(reader, "recommendation_route_total") == {
        (("route", "popular_fallback"),): 1,
        (("route", "rank_deterministic"),): 1,
        (("route", "rank"),): 2,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_behaviour, reason",
    [({"return_value": "not json"}, "invalid_json"), ({"side_effect": RuntimeError("down")}, "llm_error")],
)
async def test_rank_fallback_counter(reader, llm_behaviour, reason):
    llm = AsyncMock()
    llm.generate.configure_mock(**llm_behaviour)
    with patch("src.agent.nodes.rank.get_llm_provider", return_value=llm):
        await rank_candidates(AgentState(user_id="u", query="q", candidates=_candidates()))

    assert _points(reader, "recommendation_rank_fallback_total") == {(("reason", reason),): 1}


@pytest.mark.asyncio
async def test_rank_success_records_no_fallback(reader):
    llm = AsyncMock()
    llm.generate.return_value = '[{"video_id": "v1", "score": 0.9, "reason": "r"}]'
    with patch("src.agent.nodes.rank.get_llm_provider", return_value=llm):
        await rank_candidates(AgentState(user_id="u", query="q", candidates=_candidates()))

    assert _points(reader, "recommendation_rank_fallback_total") == {}


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_source_yield_only_for_sources_that_ran(mock_history, mock_trending, mock_similar, reader):
    mock_history.return_value = []
    mock_similar.return_value = [{"video_id": "s1"}, {"video_id": "s2"}]
    mock_trending.return_value = [{"video_id": "s1"}]

    await retrieve_candidates(AgentState(user_id="u"))

    # Counted before dedup; search/semantic did not run without a query.
    assert _points(reader, "recommendation_source_candidates_total") == {
        (("source", "similar"),): 2,
        (("source", "trending"),): 1,
    }
