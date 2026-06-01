"""Tests for the retrieve candidates node."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.retrieve import retrieve_candidates
from src.agent.state import AgentState


@pytest.mark.asyncio
@patch("src.agent.nodes.retrieve.semantic_search", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_similar_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_trending_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.search_videos", new_callable=AsyncMock)
@patch("src.agent.nodes.retrieve.get_user_history", new_callable=AsyncMock)
async def test_retrieve_with_query(
    mock_history, mock_search, mock_trending, mock_similar, mock_semantic,
):
    mock_history.return_value = ["vid-old"]
    mock_search.return_value = [
        {"id": "vid-1", "title": "Python Tutorial", "description": "Basics"},
        {"id": "vid-2", "title": "Go Tutorial", "description": "Intro"},
    ]
    mock_similar.return_value = []
    mock_semantic.return_value = [
        {"video_id": "vid-sem", "title": "Semantic Hit", "description": "S"},
    ]
    mock_trending.return_value = [
        {"video_id": "vid-3", "watch_count": 42},
    ]

    state = AgentState(user_id="user-1", query="tutorial")
    result = await retrieve_candidates(state)

    assert result.watch_history == ["vid-old"]
    # 2 search + 1 semantic + 1 trending = 4
    assert len(result.candidates) == 4
    sources = {c.video_id: c.source for c in result.candidates}
    assert sources["vid-1"] == "search"
    assert sources["vid-sem"] == "semantic"
    assert sources["vid-3"] == "trending"
    mock_search.assert_called_once_with("tutorial")
    mock_semantic.assert_called_once_with("tutorial")
    # similar is now keyed off user_id (reads precomputed user_features), not history.
    mock_similar.assert_called_once_with("user-1")


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
async def test_retrieve_cold_start_user_gets_no_candidates(
    mock_history, mock_search, mock_trending, mock_similar,
):
    """Cold-start: similar IS called (gated inside the tool by user_features row
    existence), trending may also be empty. Result: no candidates."""
    mock_history.return_value = []
    mock_similar.return_value = []
    mock_trending.return_value = []

    state = AgentState(user_id="cold-user")
    result = await retrieve_candidates(state)

    mock_similar.assert_called_once_with("cold-user")
    assert result.candidates == []


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
