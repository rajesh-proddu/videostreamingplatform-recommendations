"""Tests for the recommendation agent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.state import AgentState, VideoCandidate
from src.agent.nodes.filter import filter_results
from src.agent.nodes.retrieve import retrieve_candidates


@pytest.mark.asyncio
async def test_filter_removes_watched():
    state = AgentState(
        user_id="user-1",
        watch_history=["vid-1", "vid-2"],
        ranked_results=[
            {"video_id": "vid-1", "title": "Watched", "score": 0.9, "reason": "test"},
            {"video_id": "vid-3", "title": "New", "score": 0.8, "reason": "test"},
        ],
    )
    result = await filter_results(state)
    assert len(result.ranked_results) == 1
    assert result.ranked_results[0]["video_id"] == "vid-3"


@pytest.mark.asyncio
async def test_filter_keeps_watched_with_query():
    state = AgentState(
        user_id="user-1",
        query="specific search",
        watch_history=["vid-1"],
        ranked_results=[
            {"video_id": "vid-1", "title": "Watched", "score": 0.9, "reason": "test"},
        ],
    )
    result = await filter_results(state)
    assert len(result.ranked_results) == 1


@pytest.mark.asyncio
async def test_filter_removes_low_score():
    state = AgentState(
        user_id="user-1",
        ranked_results=[
            {"video_id": "vid-1", "title": "Good", "score": 0.8, "reason": "test"},
            {"video_id": "vid-2", "title": "Bad", "score": 0.05, "reason": "test"},
        ],
    )
    result = await filter_results(state)
    assert len(result.ranked_results) == 1
    assert result.ranked_results[0]["video_id"] == "vid-1"


@pytest.mark.asyncio
async def test_filter_respects_limit():
    state = AgentState(
        user_id="user-1",
        limit=2,
        ranked_results=[
            {"video_id": f"vid-{i}", "title": f"Video {i}", "score": 0.9, "reason": "test"}
            for i in range(5)
        ],
    )
    result = await filter_results(state)
    assert len(result.ranked_results) == 2
