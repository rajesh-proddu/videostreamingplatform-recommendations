"""Retrieve candidate videos from multiple sources."""

import logging

from src.agent.state import AgentState, VideoCandidate
from src.observability import get_tracer
from src.tools.search_videos import search_videos
from src.tools.semantic_search import semantic_search
from src.tools.similar import get_similar_videos
from src.tools.trending import get_trending_videos
from src.tools.user_history import get_user_history

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


async def retrieve_candidates(state: AgentState) -> AgentState:
    """Retrieve candidate videos from ES, watch history, and trending."""
    with _tracer.start_as_current_span("agent.retrieve") as span:
        return await _retrieve_inner(state, span)


async def _retrieve_inner(state: AgentState, span) -> AgentState:
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

        # Semantic ANN search on the query embedding (complements keyword).
        try:
            semantic = await semantic_search(state.query)
            candidates.extend([
                VideoCandidate(
                    video_id=v["video_id"],
                    title=v.get("title", ""),
                    description=v.get("description", ""),
                    source="semantic",
                )
                for v in semantic
            ])
        except Exception:
            logger.warning("Failed semantic search")

    # Vector similarity to the user's precomputed taste vector (personalization)
    try:
        similar = await get_similar_videos(state.user_id)
        candidates.extend([
            VideoCandidate(
                video_id=v["video_id"],
                title=v.get("title", ""),
                description=v.get("description", ""),
                source="similar",
            )
            for v in similar
        ])
    except Exception:
        logger.warning("Failed to get similar videos")

    # Get trending videos (from precomputed table)
    try:
        trending = await get_trending_videos()
        candidates.extend([
            VideoCandidate(
                video_id=v["video_id"],
                title=v.get("title", ""),
                description=v.get("description", ""),
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

    span.set_attribute("candidates", len(state.candidates))
    span.set_attribute("watch_history", len(state.watch_history))
    logger.info(f"Retrieved {len(state.candidates)} candidates for user {state.user_id}")
    return state
