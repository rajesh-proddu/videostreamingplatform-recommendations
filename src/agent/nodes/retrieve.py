"""Retrieve candidate videos from multiple sources."""

import asyncio
import logging

from src.agent.state import AgentState, VideoCandidate
from src.config import config
from src.observability import get_tracer
from src.tools.search_videos import search_videos
from src.tools.semantic_search import semantic_search
from src.tools.similar import get_similar_videos
from src.tools.trending import get_trending_videos
from src.tools.user_history import get_user_history
from src.tools.video_titles import get_video_titles

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


async def _fetch_history(user_id: str) -> list[str]:
    try:
        return await get_user_history(user_id)
    except Exception:
        logger.warning(f"Failed to get watch history for user {user_id}")
        return []


async def _fetch_history_with_titles(user_id: str) -> tuple[list[str], list[str]]:
    history = await _fetch_history(user_id)
    titles = await get_video_titles(history)
    return history, [titles.get(vid, vid) for vid in history]


async def _fetch_search(query: str) -> list[VideoCandidate]:
    try:
        results = await search_videos(query)
    except Exception:
        logger.warning("Failed to search videos")
        return []
    return [
        VideoCandidate(
            video_id=v["id"],
            title=v.get("title", ""),
            description=v.get("description", ""),
            source="search",
        )
        for v in results
    ]


async def _fetch_semantic(query: str) -> list[VideoCandidate]:
    try:
        results = await semantic_search(query)
    except Exception:
        logger.warning("Failed semantic search")
        return []
    return [
        VideoCandidate(
            video_id=v["video_id"],
            title=v.get("title", ""),
            description=v.get("description", ""),
            source="semantic",
        )
        for v in results
    ]


async def _fetch_similar(user_id: str) -> list[VideoCandidate]:
    try:
        results = await get_similar_videos(user_id)
    except Exception:
        logger.warning("Failed to get similar videos")
        return []
    return [
        VideoCandidate(
            video_id=v["video_id"],
            title=v.get("title", ""),
            description=v.get("description", ""),
            source="similar",
        )
        for v in results
    ]


async def _fetch_trending() -> list[VideoCandidate]:
    try:
        results = await get_trending_videos()
    except Exception:
        logger.warning("Failed to get trending videos")
        return []
    return [
        VideoCandidate(
            video_id=v["video_id"],
            title=v.get("title", ""),
            description=v.get("description", ""),
            source="trending",
        )
        for v in results
    ]


async def retrieve_candidates(state: AgentState) -> AgentState:
    """Retrieve candidate videos from ES, watch history, and trending."""
    with _tracer.start_as_current_span("agent.retrieve") as span:
        return await _retrieve_inner(state, span)


async def _retrieve_inner(state: AgentState, span) -> AgentState:
    # Independent I/O — fan out concurrently instead of awaiting one at a time.
    # Titles are only needed by the LLM ranker, which runs only when there's a query.
    history_task = _fetch_history_with_titles(state.user_id) if state.query else _fetch_history(state.user_id)
    search_task = _fetch_search(state.query) if state.query else None
    semantic_task = _fetch_semantic(state.query) if state.query else None
    similar_task = _fetch_similar(state.user_id)
    trending_task = _fetch_trending()

    tasks = [history_task, similar_task, trending_task]
    if search_task is not None:
        tasks.extend([search_task, semantic_task])

    gathered = await asyncio.gather(*tasks)
    history, similar, trending = gathered[:3]
    if state.query:
        state.watch_history, state.watch_history_titles = history
    else:
        state.watch_history = history
    search_results, semantic = gathered[3:] if search_task is not None else ([], [])

    # Same source priority as before, so dedup below keeps the first match.
    candidates = [*search_results, *semantic, *similar, *trending]

    # Deduplicate by video_id
    seen = set()
    unique = []
    for c in candidates:
        if c.video_id not in seen:
            seen.add(c.video_id)
            unique.append(c)
    state.candidates = unique[: config.max_rank_candidates]

    span.set_attribute("candidates", len(state.candidates))
    span.set_attribute("watch_history", len(state.watch_history))
    logger.info(f"Retrieved {len(state.candidates)} candidates for user {state.user_id}")
    return state
