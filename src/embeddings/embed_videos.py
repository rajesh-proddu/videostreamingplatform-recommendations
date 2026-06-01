"""Batch job to compute and store video embeddings."""

import asyncio
import logging

from elasticsearch import AsyncElasticsearch

from src.config import config
from src.embeddings.store import EmbeddingStore
from src.llm.provider import get_llm_provider
from src.observability import get_tracer, init_observability

logger = logging.getLogger(__name__)


async def embed_all_videos():
    """Fetch all videos from ES and compute/store embeddings."""
    tracer = get_tracer(__name__)
    store = EmbeddingStore()
    await store.initialize()

    es = AsyncElasticsearch(config.elasticsearch_url)
    llm = get_llm_provider()

    try:
        with tracer.start_as_current_span("embed_videos.scroll") as span:
            response = await es.search(
                index=config.es_video_index,
                body={"query": {"match_all": {}}, "size": 100},
                scroll="2m",
            )

            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
            total = 0
            failed = 0

            while hits:
                for hit in hits:
                    video_id = hit["_id"]
                    source = hit["_source"]
                    title = source.get("title", "")
                    description = source.get("description", "")

                    text = f"{title}. {description}"
                    try:
                        embedding = await llm.embed(text)
                        await store.store_embedding(video_id, title, description, embedding)
                        total += 1
                    except Exception:
                        failed += 1
                        logger.warning(f"Failed to embed video {video_id}")

                response = await es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]

            span.set_attribute("embedded", total)
            span.set_attribute("failed", failed)
            logger.info(f"Embedded {total} videos")

    finally:
        await es.close()
        await store.close()


def main():
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
    from opentelemetry.instrumentation.elasticsearch import (
        ElasticsearchInstrumentor,
    )
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # Batch job — no Prometheus HTTP server (pod exits too fast to be scraped).
    # Traces still flush via OTLP push on shutdown.
    shutdown = init_observability(
        "embed-videos-job",
        instrumentors=[
            HTTPXClientInstrumentor(),
            AsyncPGInstrumentor(),
            ElasticsearchInstrumentor(),
            BotocoreInstrumentor(),
        ],
    )
    try:
        asyncio.run(embed_all_videos())
    finally:
        # Explicit flush before exit — atexit alone races with boto3's non-daemon
        # threads on short jobs and can drop the last batch of spans.
        shutdown()


if __name__ == "__main__":
    main()
