"""Retrieve candidate videos from multiple sources."""

import logging

from src.agent.state import AgentState, VideoCandidate
from src.tools.search_videos import search_videos
from src.tools.trending import get_trending_videos
from src.tools.user_history import get_user_history

logger = logging.getLogger(__name__)


async def retrieve_candidates(state: AgentState) -> AgentState:
    """Retrieve candidate videos from ES, watch history, and trending."""
    candidates = []

    # Get user's watch history for context
    try:
        state.watch_history = await get_user_history(state.user_id)
    except Exception:
        logger.warning(f"Failed to get watch history for user {state.user_id}")
        state.watch_history = []

    # Search ES if query provided
    if state.query:
        try:
            search_results = await search_videos(state.query)
            candidates.extend([
                VideoCandidate(
                    video_id=v["id"],
                    title=v.get("title", ""),
                    description=v.get("description", ""),
                    source="search",
                )
                for v in search_results
            ])
        except Exception:
            logger.warning("Failed to search videos")

    # Get trending videos
    try:
        trending = await get_trending_videos()
        candidates.extend([
            VideoCandidate(
                video_id=v["video_id"],
                title=v.get("title", ""),
                source="trending",
            )
            for v in trending
        ])
    except Exception:
        logger.warning("Failed to get trending videos")

    # Deduplicate by video_id
    seen = set()
    unique = []
    for c in candidates:
        if c.video_id not in seen:
            seen.add(c.video_id)
            unique.append(c)
    state.candidates = unique

    logger.info(f"Retrieved {len(state.candidates)} candidates for user {state.user_id}")
    return state
