"""FastAPI application for the recommendation service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.elasticsearch import ElasticsearchInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.api.routes.recommend import router as recommend_router
from src.db import close_pool, get_pool
from src.observability import init_observability


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

# FastAPIInstrumentor.instrument() (server-side) needs the app object; the rest
# are global patches. Mount /metrics on this app instead of a side-car port so
# Kubernetes can scrape one container port.
init_observability(
    "recommendation-api",
    fastapi_app=app,
    instrumentors=[
        HTTPXClientInstrumentor(),
        AsyncPGInstrumentor(),
        ElasticsearchInstrumentor(),
        BotocoreInstrumentor(),
    ],
)
FastAPIInstrumentor.instrument_app(app)

app.include_router(recommend_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}
