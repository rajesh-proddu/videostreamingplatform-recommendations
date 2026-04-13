"""Post-processing filters for recommendations."""

import logging

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


async def filter_results(state: AgentState) -> AgentState:
    """Apply business rules and post-processing filters."""
    if not state.ranked_results:
        return state

    filtered = []
    watched_set = set(state.watch_history)

    for result in state.ranked_results:
        video_id = result.get("video_id", "")

        # Skip already watched videos (unless from explicit search)
        if video_id in watched_set and not state.query:
            continue

        # Enforce minimum score threshold
        if result.get("score", 0) < 0.1:
            continue

        filtered.append(result)

    state.ranked_results = filtered[:state.limit]
    logger.info(f"Filtered to {len(state.ranked_results)} recommendations")
    return state
