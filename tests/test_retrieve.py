"""Tests for the retrieve candidates node."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.retrieve import retrieve_candidates
from src.agent.state import AgentState


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_with_query(mock_history, mock_search, mock_trending, mock_similar):
    mock_history.return_value = ["vid-old"]
    mock_search.return_value = [
        {"id": "vid-1", "title": "Python Tutorial", "description": "Basics"},
        {"id": "vid-2", "title": "Go Tutorial", "description": "Intro"},
    ]
    mock_similar.return_value = []
    mock_trending.return_value = [
        {"video_id": "vid-3", "watch_count": 42},
    ]

    state = AgentState(user_id="user-1", query="tutorial")
    result = await retrieve_candidates(state)

    assert result.watch_history == ["vid-old"]
    assert len(result.candidates) == 3
    assert result.candidates[0].video_id == "vid-1"
    assert result.candidates[0].source == "search"
    assert result.candidates[2].source == "trending"
    mock_search.assert_called_once_with("tutorial")
    mock_similar.assert_called_once_with(["vid-old"])


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_merges_similar_with_personalization(
    mock_history, mock_search, mock_trending, mock_similar,
):
    mock_history.return_value = ["watched-1"]
    mock_similar.return_value = [
        {"video_id": "sim-1", "title": "Like Watched", "description": "x"},
    ]
    mock_trending.return_value = [
        {"video_id": "trend-1", "watch_count": 7},
    ]

    state = AgentState(user_id="user-1")
    result = await retrieve_candidates(state)

    mock_search.assert_not_called()  # no query
    sources = {c.video_id: c.source for c in result.candidates}
    assert sources == {"sim-1": "similar", "trend-1": "trending"}


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_skips_similar_for_cold_start(
    mock_history, mock_search, mock_trending, mock_similar,
):
    mock_history.return_value = []
    mock_trending.return_value = []

    state = AgentState(user_id="cold-user")
    await retrieve_candidates(state)

    mock_similar.assert_not_called()


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_without_query_skips_search(mock_history, mock_search, mock_trending):
    mock_history.return_value = []
    mock_trending.return_value = [{"video_id": "vid-t1", "watch_count": 10}]

    state = AgentState(user_id="user-2")
    result = await retrieve_candidates(state)

    mock_search.assert_not_called()
    assert len(result.candidates) == 1
    assert result.candidates[0].video_id == "vid-t1"


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_deduplicates_by_video_id(mock_history, mock_search, mock_trending):
    mock_history.return_value = []
    mock_search.return_value = [
        {"id": "vid-1", "title": "Dup", "description": ""},
    ]
    mock_trending.return_value = [
        {"video_id": "vid-1", "watch_count": 5},  # same ID as search result
    ]

    state = AgentState(user_id="user-1", query="test")
    result = await retrieve_candidates(state)

    assert len(result.candidates) == 1
    assert result.candidates[0].source == "search"  # first one wins


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_handles_history_failure(mock_history, mock_search, mock_trending):
    mock_history.side_effect = Exception("DB down")
    mock_trending.return_value = [{"video_id": "vid-1", "watch_count": 1}]

    state = AgentState(user_id="user-1")
    result = await retrieve_candidates(state)

    assert result.watch_history == []
    assert len(result.candidates) == 1


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_handles_search_failure(mock_history, mock_search, mock_trending):
    mock_history.return_value = []
    mock_search.side_effect = Exception("ES down")
    mock_trending.return_value = [{"video_id": "vid-t1", "watch_count": 3}]

    state = AgentState(user_id="user-1", query="test")
    result = await retrieve_candidates(state)

    # Search failed, but trending still works
    assert len(result.candidates) == 1
    assert result.candidates[0].source == "trending"


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_handles_trending_failure(mock_history, mock_search, mock_trending):
    mock_history.return_value = []
    mock_trending.side_effect = Exception("pg down")

    state = AgentState(user_id="user-1")
    result = await retrieve_candidates(state)

    assert result.candidates == []
