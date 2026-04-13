"""Rank candidates using LLM-powered scoring."""

import json
import logging

from src.agent.state import AgentState
from src.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)

RANKING_PROMPT = """\
You are a video recommendation engine. Given a user's watch history \
and candidate videos, score each candidate from 0.0 to 1.0 based on relevance.

User's recent watch history (video IDs): {watch_history}
User's search query: {query}

Candidate videos:
{candidates}

Return a JSON array of objects with "video_id", "score" (0.0-1.0), \
and "reason" (brief explanation).
Only return the JSON array, no other text."""


async def rank_candidates(state: AgentState) -> AgentState:
    """Use LLM to rank candidate videos based on user context."""
    if not state.candidates:
        logger.info("No candidates to rank")
        return state

    candidates_text = "\n".join([
        f"- ID: {c.video_id}, Title: {c.title}, Description: {c.description}, Source: {c.source}"
        for c in state.candidates
    ])

    prompt = RANKING_PROMPT.format(
        watch_history=", ".join(state.watch_history[-20:]) if state.watch_history else "none",
        query=state.query or "none",
        candidates=candidates_text,
    )

    try:
        llm = get_llm_provider()
        response = await llm.generate(prompt)

        rankings = json.loads(response)
        state.ranked_results = sorted(rankings, key=lambda x: x.get("score", 0), reverse=True)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON, falling back to source-based ranking")
        state.ranked_results = [
            {
                "video_id": c.video_id,
                "title": c.title,
                "score": 0.8 if c.source == "search" else 0.5,
                "reason": f"Matched via {c.source}",
            }
            for c in state.candidates
        ]
    except Exception:
        logger.exception("Failed to rank candidates with LLM")
        state.ranked_results = [
            {
                "video_id": c.video_id,
                "title": c.title,
                "score": 0.5,
                "reason": "Default ranking (LLM unavailable)",
            }
            for c in state.candidates
        ]

    logger.info(f"Ranked {len(state.ranked_results)} candidates")
    return state
