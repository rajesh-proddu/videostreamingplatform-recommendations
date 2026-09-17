"""Impression log: one pgvector row per served recommendation response.

This is the record of what was shown, which evals, click-through, and
feedback attribution all join against. Writes run as background tasks off
the request path, and a failed write never fails the request: it is logged
and counted (recommendation_impression_write_failures_total).

The API creates the table at startup (ensure_schema). If that fails,
requests keep working and each write fails and is counted.
"""

import asyncio
import json
import logging
from typing import Optional

from src.agent.metrics import record_impression_write_failure
from src.config import config
from src.db import get_pool

logger = logging.getLogger(__name__)

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS recommendation_impressions (
        request_id     UUID PRIMARY KEY,
        user_id        TEXT NOT NULL,
        query          TEXT,
        route          TEXT,
        prompt_version TEXT,
        model_id       TEXT,
        rank_fallback  TEXT,
        latency_ms     INTEGER NOT NULL,
        items          JSONB NOT NULL,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS recommendation_impressions_user_created
        ON recommendation_impressions (user_id, created_at)
    """,
]

_INSERT = """
    INSERT INTO recommendation_impressions
        (request_id, user_id, query, route, prompt_version, model_id, rank_fallback, latency_ms, items)
    VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
"""

# Strong references so in-flight writes aren't garbage-collected.
_pending: set[asyncio.Task] = set()


async def ensure_schema() -> None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            for stmt in _SCHEMA:
                await conn.execute(stmt)
    except Exception:
        logger.exception("Failed to create recommendation_impressions; impression writes will fail")


def _model_id() -> Optional[str]:
    return {
        "ollama": config.ollama_model,
        "bedrock": config.bedrock_model_id,
        "anthropic": config.anthropic_model,
    }.get(config.llm_provider)


def _build_row(request_id: str, final_state: dict, served: list[dict], latency_ms: int) -> tuple:
    sources = {c.video_id: c.source for c in final_state.get("candidates", [])}
    route = final_state.get("route")
    items = [
        {
            "video_id": r.get("video_id"),
            "rank": i,
            "score": r.get("score"),
            # popular_fallback results never came from retrieve.
            "source": sources.get(r.get("video_id"), "popular" if route == "popular_fallback" else None),
        }
        for i, r in enumerate(served, start=1)
    ]
    return (
        request_id,
        final_state["user_id"],
        final_state.get("query"),
        route,
        final_state.get("prompt_version"),
        _model_id() if route == "rank" else None,
        final_state.get("rank_fallback"),
        latency_ms,
        json.dumps(items),
    )


async def _write(request_id: str, final_state: dict, served: list[dict], latency_ms: int) -> None:
    row = _build_row(request_id, final_state, served, latency_ms)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_INSERT, *row)


def _on_done(task: asyncio.Task) -> None:
    _pending.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Failed to write recommendation impression", exc_info=exc)
        record_impression_write_failure()


def record_impression(request_id: str, final_state: dict, served: list[dict], latency_ms: int) -> None:
    """Schedule the impression write and return immediately."""
    task = asyncio.create_task(_write(request_id, final_state, served, latency_ms))
    _pending.add(task)
    task.add_done_callback(_on_done)


async def drain(timeout: float = 5.0) -> None:
    """Wait for in-flight writes (at shutdown, before the pool closes)."""
    if _pending:
        await asyncio.wait(set(_pending), timeout=timeout)
