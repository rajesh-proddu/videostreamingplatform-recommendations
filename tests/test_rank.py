"""Tests for LLM-powered rank_candidates node."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.nodes.rank import rank_candidates
from src.agent.state import AgentState, VideoCandidate


def _make_state(**kwargs):
    defaults = dict(user_id="user-1", query="python tutorial")
    defaults.update(kwargs)
    return AgentState(**defaults)


def _make_candidates():
    return [
        VideoCandidate(video_id="vid-1", title="Python Basics", description="Intro to Python", source="search"),
        VideoCandidate(video_id="vid-2", title="Go Tutorial", description="Learn Go", source="trending"),
    ]


@pytest.mark.asyncio
class TestRankCandidates:
    async def test_rank_empty_candidates(self):
        state = _make_state(candidates=[])
        result = await rank_candidates(state)
        assert result.ranked_results == []

    @patch("src.agent.nodes.rank.get_llm_provider")
    async def test_rank_success(self, mock_get_provider):
        mock_llm = AsyncMock()
        mock_get_provider.return_value = mock_llm

        rankings = [
            {"video_id": "vid-1", "score": 0.9, "reason": "Very relevant"},
            {"video_id": "vid-2", "score": 0.6, "reason": "Somewhat relevant"},
        ]
        mock_llm.generate.return_value = json.dumps(rankings)

        state = _make_state(candidates=_make_candidates())
        result = await rank_candidates(state)

        assert len(result.ranked_results) == 2
        assert result.ranked_results[0]["score"] >= result.ranked_results[1]["score"]
        assert result.ranked_results[0]["video_id"] == "vid-1"

    @patch("src.agent.nodes.rank.get_llm_provider")
    async def test_rank_invalid_json_fallback(self, mock_get_provider):
        mock_llm = AsyncMock()
        mock_get_provider.return_value = mock_llm
        mock_llm.generate.return_value = "not valid json at all"

        state = _make_state(candidates=_make_candidates())
        result = await rank_candidates(state)

        assert len(result.ranked_results) == 2
        # search source gets 0.8, trending gets 0.5
        scores = {r["video_id"]: r["score"] for r in result.ranked_results}
        assert scores["vid-1"] == 0.8
        assert scores["vid-2"] == 0.5

    @patch("src.agent.nodes.rank.get_llm_provider")
    async def test_rank_llm_error_fallback(self, mock_get_provider):
        mock_llm = AsyncMock()
        mock_get_provider.return_value = mock_llm
        mock_llm.generate.side_effect = RuntimeError("LLM unavailable")

        state = _make_state(candidates=_make_candidates())
        result = await rank_candidates(state)

        assert len(result.ranked_results) == 2
        for r in result.ranked_results:
            assert r["score"] == 0.5
            assert "Default ranking" in r["reason"]
