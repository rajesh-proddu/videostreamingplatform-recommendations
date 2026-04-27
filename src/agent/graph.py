"""LangGraph state graph for video recommendations."""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.filter import filter_results
from src.agent.nodes.rank import rank_candidates
from src.agent.nodes.retrieve import retrieve_candidates
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build the recommendation agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_candidates)
    graph.add_node("rank", rank_candidates)
    graph.add_node("filter", filter_results)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank", "filter")
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

    # LangGraph's ainvoke returns the final state as a dict, not the dataclass.
    result = await recommendation_graph.ainvoke(initial_state)

    return result["ranked_results"][:limit]
