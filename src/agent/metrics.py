"""Request-path metrics for the recommendation agent.

Instruments are created on first use, not at import: src.api.main imports the
graph before init_observability() binds the real MeterProvider (same reason as
the lazy _init_metrics in the embeddings consumer).
"""

from functools import cache
from types import SimpleNamespace

from src.observability import get_meter


@cache
def _instruments() -> SimpleNamespace:
    meter = get_meter(__name__)
    return SimpleNamespace(
        route=meter.create_counter(
            "recommendation_route_total",
            description="Ranking path chosen after retrieve (rank, rank_deterministic, popular_fallback)",
        ),
        rank_fallback=meter.create_counter(
            "recommendation_rank_fallback_total",
            description="LLM ranking replaced by non-LLM scores, by reason (invalid_json, llm_error)",
        ),
        source_candidates=meter.create_counter(
            "recommendation_source_candidates_total",
            description="Candidates returned per retrieval source, before dedup",
        ),
    )


def record_route(route: str) -> None:
    _instruments().route.add(1, {"route": route})


def record_rank_fallback(reason: str) -> None:
    _instruments().rank_fallback.add(1, {"reason": reason})


def record_source_candidates(source: str, count: int) -> None:
    _instruments().source_candidates.add(count, {"source": source})
