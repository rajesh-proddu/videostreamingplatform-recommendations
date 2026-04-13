"""Agent state schema for the recommendation graph."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoCandidate:
    video_id: str
    title: str
    description: str = ""
    score: float = 0.0
    source: str = ""  # "search", "history", "trending"


@dataclass
class AgentState:
    """State passed through the LangGraph nodes."""
    user_id: str
    query: Optional[str] = None
    limit: int = 10
    watch_history: list[str] = field(default_factory=list)
    candidates: list[VideoCandidate] = field(default_factory=list)
    ranked_results: list[dict] = field(default_factory=list)
    error: Optional[str] = None
