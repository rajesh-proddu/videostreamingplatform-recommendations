"""Recommendation API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import get_recommendations

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to get recommendations for")
    query: Optional[str] = Field(None, description="Optional search query to refine recommendations")
    limit: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")


class VideoRecommendation(BaseModel):
    video_id: str
    title: str
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[VideoRecommendation]
    query: Optional[str] = None


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get personalized video recommendations for a user."""
    try:
        recommendations = await get_recommendations(
            user_id=request.user_id,
            query=request.query,
            limit=request.limit,
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            query=request.query,
        )
    except Exception as e:
        logger.exception("Failed to get recommendations")
        raise HTTPException(status_code=500, detail=str(e))
