"""LangGraph state graph for video recommendations."""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.filter import filter_results
from src.agent.nodes.popular_fallback import popular_fallback
from src.agent.nodes.rank import rank_candidates
from src.agent.nodes.rank_deterministic import rank_deterministic
from src.agent.nodes.retrieve import retrieve_candidates
from src.agent.state import AgentState
from src.observability import get_tracer

logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


def _route_after_retrieve(state: AgentState) -> str:
    """Pick the ranking path based on retrieve output."""
    if not state.candidates:
        return "popular_fallback"
    if not state.query:
        return "rank_deterministic"
    return "rank"


def build_graph() -> StateGraph:
    """Build the recommendation agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_candidates)
    graph.add_node("rank", rank_candidates)
    graph.add_node("rank_deterministic", rank_deterministic)
    graph.add_node("popular_fallback", popular_fallback)
    graph.add_node("filter", filter_results)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "popular_fallback": "popular_fallback",
            "rank_deterministic": "rank_deterministic",
            "rank": "rank",
        },
    )
    graph.add_edge("rank", "filter")
    graph.add_edge("rank_deterministic", "filter")
    graph.add_edge("popular_fallback", "filter")
    graph.add_edge("filter", END)

    return graph.compile()


# Compiled graph singleton
recommendation_graph = build_graph()


async def get_recommendations(
    user_id: str,
    query: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Run the recommendation graph and return results."""
    initial_state = AgentState(
        user_id=user_id,
        query=query,
        limit=limit,
    )

    with _tracer.start_as_current_span("agent.invoke") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("has_query", query is not None)
        span.set_attribute("limit", limit)
        # LangGraph's ainvoke returns the final state as a dict, not the dataclass.
        result = await recommendation_graph.ainvoke(initial_state)
        results = result["ranked_results"][:limit]
        span.set_attribute("result_count", len(results))
        return results
