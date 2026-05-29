"""Deterministic ranker for the no-query feed path (no LLM)."""

import logging

from src.agent.state import AgentState

logger = logging.getLogger(__name__)

SOURCE_SCORES = {
    "search": 0.7,
    "similar": 0.65,
    "trending": 0.55,
    "history": 0.4,
}


async def rank_deterministic(state: AgentState) -> AgentState:
    """Score candidates by source weight. Used when there's no user query."""
    state.ranked_results = sorted(
        (
            {
                "video_id": c.video_id,
                "title": c.title,
                "score": SOURCE_SCORES.get(c.source, 0.5),
                "reason": f"From {c.source}",
            }
            for c in state.candidates
        ),
        key=lambda x: x["score"],
        reverse=True,
    )
    logger.info(f"Deterministic ranker produced {len(state.ranked_results)} results")
    return state
