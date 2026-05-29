"""Last-ditch fallback when retrieval produces no candidates."""

import logging

from src.agent.state import AgentState
from src.tools.trending import get_trending_videos

logger = logging.getLogger(__name__)

FALLBACK_WINDOW_HOURS = 24 * 7


async def popular_fallback(state: AgentState) -> AgentState:
    """Return popular videos over a wider window when retrieve found nothing."""
    try:
        trending = await get_trending_videos(hours=FALLBACK_WINDOW_HOURS, limit=state.limit)
    except Exception:
        logger.exception("Popular fallback failed to fetch trending")
        trending = []

    state.ranked_results = [
        {
            "video_id": v["video_id"],
            "title": v.get("title", ""),
            "score": 0.5,
            "reason": "Popular this week",
        }
        for v in trending
    ]
    logger.info(f"Popular fallback produced {len(state.ranked_results)} results")
    return state
