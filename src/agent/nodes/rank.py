"""Rank candidates using LLM-powered scoring."""

import json
import logging

from src.agent.metrics import record_rank_fallback
from src.agent.state import AgentState
from src.llm.provider import get_llm_provider
from src.observability import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)

# Bump whenever RANKING_PROMPT changes, so eval scores and impressions can be
# attributed to a specific prompt.
PROMPT_VERSION = "2"

RANKING_PROMPT = """\
You are a video recommendation engine. Given a user's watch history \
and candidate videos, score each candidate from 0.0 to 1.0 based on relevance.

User's recent watch history (titles): {watch_history}
User's search query: {query}

Candidate videos:
{candidates}

Return a JSON array of objects with "video_id", "score" (0.0-1.0), \
and "reason" (brief explanation).
Only return the JSON array, no other text."""


def _strip_code_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    if s.lstrip().lower().startswith("json"):
        s = s.lstrip()[4:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


async def rank_candidates(state: AgentState) -> AgentState:
    """Use LLM to rank candidate videos based on user context."""
    with _tracer.start_as_current_span("agent.rank") as span:
        span.set_attribute("candidates", len(state.candidates))
        return await _rank_inner(state)


async def _rank_inner(state: AgentState) -> AgentState:
    if not state.candidates:
        logger.info("No candidates to rank")
        return state

    candidates_text = "\n".join([
        f"- ID: {c.video_id}, Title: {c.title}, Description: {c.description}, Source: {c.source}"
        for c in state.candidates
    ])

    history = state.watch_history_titles or state.watch_history
    prompt = RANKING_PROMPT.format(
        watch_history=", ".join(history[-20:]) if history else "none",
        query=state.query or "none",
        candidates=candidates_text,
    )

    try:
        llm = get_llm_provider()
        response = await llm.generate(prompt)

        rankings = json.loads(_strip_code_fence(response))
        titles = {c.video_id: c.title for c in state.candidates}
        for r in rankings:
            r.setdefault("title", titles.get(r.get("video_id", ""), ""))
        state.ranked_results = sorted(rankings, key=lambda x: x.get("score", 0), reverse=True)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON, falling back to source-based ranking")
        record_rank_fallback("invalid_json")
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
        record_rank_fallback("llm_error")
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
