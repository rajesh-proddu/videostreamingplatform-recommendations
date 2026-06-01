"""
Kafka → pgvector consumer for video embeddings.

Reads the 'video-events' topic, batches messages, calls the configured LLM
provider's embed_batch(), and upserts vectors into pgvector.video_embeddings.
VIDEO_DELETED removes the row.

Delivery semantics: batch-level manual commit. Upserts are idempotent (keyed by
video_id), so re-processing a batch after a crash-before-commit re-applies the
same writes — no data loss, no duplicates.

Failures classified (matches kafka-es-consumer pattern):
  - transient (httpx/pgvector pool/connection): consumer stops WITHOUT committing
    so a restart resumes from the same offset once the downstream recovers.
  - poison (bad JSON, unknown event type, embed validation error): raw message
    is routed to the DLQ topic, offset is committed past it.

The Kafka client is sync (confluent-kafka). The LLM provider and pgvector pool
are async. We drive both with a single asyncio loop owned by this consumer,
calling loop.run_until_complete(flush_batch) from inside the sync poll loop.
"""

import asyncio
import json
import logging
import os
import signal
import time
from typing import Optional

import asyncpg
import httpx
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

from src.config import config
from src.embeddings.store import EmbeddingStore
from src.llm.provider import LLMProvider, get_llm_provider
from src.observability import get_meter, get_tracer, init_observability

logger = logging.getLogger("embeddings-consumer")
_tracer = get_tracer(__name__)

# Metrics are created lazily after init_observability binds the real meter
# provider; calling create_counter on the proxy before init returns a no-op.
_messages_processed = None
_dlq_messages = None
_batch_flush_duration = None


def _init_metrics() -> None:
    global _messages_processed, _dlq_messages, _batch_flush_duration
    meter = get_meter(__name__)
    _messages_processed = meter.create_counter(
        "embeddings_messages_processed_total",
        description="Kafka video-events consumed by the embeddings pipeline",
    )
    _dlq_messages = meter.create_counter(
        "embeddings_messages_dlq_total",
        description="Poison messages routed to the embeddings DLQ",
    )
    _batch_flush_duration = meter.create_histogram(
        "embeddings_batch_flush_seconds",
        description="Wall time of pgvector upsert+delete per batch flush",
        unit="s",
    )


# Connection-level failures that should pause the consumer rather than DLQ the
# message. Anything else (bad payload, embed validation) is treated as poison.
_TRANSIENT_TYPES: tuple[type, ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    asyncpg.PostgresConnectionError,
    asyncpg.exceptions.CannotConnectNowError,
    ConnectionError,
    TimeoutError,
)


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, _TRANSIENT_TYPES)


class EmbeddingsConsumer:
    """Buffers video events, embeds in batches, upserts pgvector."""

    def __init__(
        self,
        consumer: Consumer,
        dlq_producer: Producer,
        store: EmbeddingStore,
        llm: LLMProvider,
        topic: str,
        dlq_topic: str,
        batch_size: int,
        idle_flush_seconds: float,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.consumer = consumer
        self.dlq_producer = dlq_producer
        self.store = store
        self.llm = llm
        self.topic = topic
        self.dlq_topic = dlq_topic
        self.batch_size = batch_size
        self.idle_flush_seconds = idle_flush_seconds
        self.loop = loop or asyncio.new_event_loop()
        self.running = True
        # (video_id, title, description, op) where op ∈ {"upsert", "delete"}
        self.batch: list[tuple[str, Optional[str], Optional[str], str]] = []
        self.last_flush_ts = time.monotonic()

    def install_signal_handlers(self):
        """Register SIGTERM/SIGINT for graceful shutdown. Called by main() only —
        tests skip this so they don't fight pytest's own signal handling."""
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received, flushing batch and stopping...")
        self.running = False

    def start(self):
        """Subscribe and run the poll loop. Initializes pgvector pool on entry."""
        self.loop.run_until_complete(self.store.initialize())
        self.consumer.subscribe([self.topic])
        logger.info("Subscribed to topic: %s", self.topic)

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    if self._should_idle_flush():
                        self._flush_and_commit()
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())

                with _tracer.start_as_current_span("embeddings.process_message") as span:
                    span.set_attribute("kafka.partition", msg.partition())
                    span.set_attribute("kafka.offset", msg.offset())
                    try:
                        parsed = self._parse_message(msg)
                        if parsed is not None:
                            self.batch.append(parsed)
                            if _messages_processed is not None:
                                _messages_processed.add(1, {"outcome": "buffered", "op": parsed[3]})
                        if len(self.batch) >= self.batch_size:
                            self._flush_and_commit()
                    except Exception as exc:
                        # Parse errors are poison; surface unexpected runtime issues
                        # the same way we do flush errors (DLQ).
                        self._to_dlq(msg, exc)
                        self.consumer.commit(asynchronous=False)
        finally:
            # Best-effort drain on shutdown. Wrapped so a failing flush during
            # shutdown (e.g. pgvector already gone) doesn't mask cleanup.
            try:
                if self.batch:
                    self._flush_and_commit()
            except Exception:
                logger.exception("Failed to flush final batch on shutdown")
            try:
                self.loop.run_until_complete(self.store.close())
            except Exception:
                logger.exception("Failed to close pgvector pool")
            self.consumer.close()
            self.loop.close()
            logger.info("Consumer closed")

    def _parse_message(
        self, msg
    ) -> Optional[tuple[str, Optional[str], Optional[str], str]]:
        """Parse a Kafka message. Returns None for messages we deliberately skip
        (no video_id, unknown event type). Raises for poison (bad JSON / structure)
        so the caller routes to DLQ."""
        value = json.loads(msg.value().decode("utf-8"))
        event_type = value.get("type", "")
        payload = value.get("payload") or {}
        video_id = payload.get("id") or ""

        if not video_id:
            logger.warning("Skipping event with no video ID: %s", value)
            return None

        # Both snake_case and UPPER variants supported — matches kafka-es-consumer
        # which already handles producer-side variation.
        if event_type in ("video.deleted", "VIDEO_DELETED"):
            return (video_id, None, None, "delete")
        if event_type in (
            "video.created",
            "VIDEO_CREATED",
            "video.updated",
            "VIDEO_UPDATED",
        ):
            title = payload.get("title") or ""
            description = payload.get("description") or ""
            return (video_id, title, description, "upsert")

        logger.warning("Unknown event type %r, skipping", event_type)
        return None

    def _flush_and_commit(self):
        """Run async flush, classify any error, then commit on success."""
        try:
            self.loop.run_until_complete(self._flush_batch())
        except Exception as exc:
            if _is_transient(exc):
                logger.exception(
                    "Transient flush failure; stopping consumer without commit"
                )
                raise
            # Non-transient flush failure: we don't have a single offset to DLQ
            # (the failure spans the batch). Re-raise — the operator will see
            # the trace and decide. This is the same fail-loud stance as the
            # other consumers for ambiguous middle-of-batch failures.
            logger.exception(
                "Non-transient flush failure; stopping consumer for operator review"
            )
            raise
        self.consumer.commit(asynchronous=False)

    async def _flush_batch(self):
        """Dedupe the buffered batch and apply upserts + deletes to pgvector."""
        if not self.batch:
            return

        flush_started = time.monotonic()

        # Last-write-wins within the batch. If create+update for the same id
        # arrive together we only embed the latest; if create+delete arrive
        # we only delete.
        deduped: dict[str, tuple[Optional[str], Optional[str], str]] = {}
        for video_id, title, description, op in self.batch:
            deduped[video_id] = (title, description, op)

        upserts = [
            (vid, t or "", d or "")
            for vid, (t, d, op) in deduped.items()
            if op == "upsert"
        ]
        deletes = [vid for vid, (_, _, op) in deduped.items() if op == "delete"]

        if upserts:
            texts = [f"{title}. {description}" for _, title, description in upserts]
            vectors = await self.llm.embed_batch(texts)
            for (vid, title, description), vec in zip(upserts, vectors):
                await self.store.store_embedding(vid, title, description, vec)
            logger.info(
                "Upserted %d embeddings (%d pre-dedup)",
                len(upserts),
                len(self.batch),
            )

        if deletes:
            await self.store.delete_embeddings(deletes)
            logger.info("Deleted %d embeddings", len(deletes))

        self.batch.clear()
        self.last_flush_ts = time.monotonic()
        if _batch_flush_duration is not None:
            _batch_flush_duration.record(time.monotonic() - flush_started)

    def _should_idle_flush(self) -> bool:
        return (
            bool(self.batch)
            and (time.monotonic() - self.last_flush_ts) >= self.idle_flush_seconds
        )

    def _to_dlq(self, msg, exc: Exception):
        """Publish a poison message to the DLQ with diagnostic headers.
        DLQ produce failure is fatal — we don't commit past a message we
        failed to preserve."""
        headers = [
            ("error_type", type(exc).__name__.encode("utf-8")),
            ("error_message", str(exc)[:512].encode("utf-8")),
            ("original_topic", (msg.topic() or "").encode("utf-8")),
            ("original_partition", str(msg.partition()).encode("utf-8")),
            ("original_offset", str(msg.offset()).encode("utf-8")),
        ]
        self.dlq_producer.produce(
            self.dlq_topic,
            value=msg.value(),
            key=msg.key(),
            headers=headers,
        )
        remaining = self.dlq_producer.flush(timeout=10)
        if remaining:
            raise RuntimeError(
                f"DLQ produce did not complete for offset {msg.offset()} "
                f"({remaining} message(s) undelivered)"
            )
        if _dlq_messages is not None:
            _dlq_messages.add(1, {"error_type": type(exc).__name__})
        logger.error(
            "Routed poison message at offset %s to DLQ %r: %s: %s",
            msg.offset(),
            self.dlq_topic,
            type(exc).__name__,
            exc,
        )


def build_consumer() -> EmbeddingsConsumer:
    """Construct the production consumer from environment-backed config."""
    kafka_consumer = Consumer({
        "bootstrap.servers": config.kafka_brokers,
        "group.id": config.kafka_embeddings_group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    dlq_producer = Producer({"bootstrap.servers": config.kafka_brokers})

    return EmbeddingsConsumer(
        consumer=kafka_consumer,
        dlq_producer=dlq_producer,
        store=EmbeddingStore(),
        llm=get_llm_provider(),
        topic=config.kafka_video_topic,
        dlq_topic=config.kafka_embeddings_dlq_topic,
        batch_size=config.embeddings_batch_size,
        idle_flush_seconds=config.embeddings_idle_flush_seconds,
    )


def main():
    # Instrumentors must run before Consumer/Producer/asyncpg/httpx classes
    # are first used, so initialize observability before build_consumer().
    from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
    from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
    from opentelemetry.instrumentation.confluent_kafka import (
        ConfluentKafkaInstrumentor,
    )
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    init_observability(
        "embeddings-consumer",
        prometheus_port=int(os.getenv("METRICS_PORT", "9100")),
        instrumentors=[
            ConfluentKafkaInstrumentor(),
            HTTPXClientInstrumentor(),
            AsyncPGInstrumentor(),
            BotocoreInstrumentor(),
        ],
    )
    _init_metrics()
    consumer = build_consumer()
    consumer.install_signal_handlers()
    consumer.start()


if __name__ == "__main__":
    main()
