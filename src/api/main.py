"""FastAPI application for the recommendation service."""

from fastapi import FastAPI
from src.api.routes.recommend import router as recommend_router

app = FastAPI(
    title="Video Recommendation Service",
    description="Agentic AI-powered video recommendations using LangGraph",
    version="0.1.0",
)

app.include_router(recommend_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}
