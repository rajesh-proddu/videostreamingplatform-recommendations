"""FastAPI application for the recommendation service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.recommend import router as recommend_router
from src.db import close_pool, get_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage connection pool lifecycle."""
    await get_pool()
    yield
    await close_pool()


app = FastAPI(
    title="Video Recommendation Service",
    description="Agentic AI-powered video recommendations using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(recommend_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}
