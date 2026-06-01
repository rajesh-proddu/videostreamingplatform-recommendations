"""Tests for the Kafka → pgvector embeddings consumer."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.consumers.embeddings_consumer import EmbeddingsConsumer


def _msg(value: dict | bytes, offset: int = 0, partition: int = 0, topic: str = "video-events"):
    """Build a Kafka message stub with the methods the consumer touches."""
    m = MagicMock()
    m.error.return_value = None
    m.offset.return_value = offset
    m.partition.return_value = partition
    m.topic.return_value = topic
    m.key.return_value = None
    m.value.return_value = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
    return m


def _make_consumer(batch_size: int = 32, idle_flush_seconds: float = 5.0) -> EmbeddingsConsumer:
    """Construct an EmbeddingsConsumer with all dependencies mocked.

    We avoid install_signal_handlers() in tests so SIGINT/SIGTERM stay owned by pytest.
    """
    return EmbeddingsConsumer(
        consumer=MagicMock(),
        dlq_producer=MagicMock(),
        store=AsyncMock(),
        llm=AsyncMock(),
        topic="video-events",
        dlq_topic="video-events-embeddings-dlq",
        batch_size=batch_size,
        idle_flush_seconds=idle_flush_seconds,
        loop=asyncio.new_event_loop(),
    )


class TestParseMessage:
    def test_video_created_returns_upsert(self):
        c = _make_consumer()
        msg = _msg({
            "type": "VIDEO_CREATED",
            "payload": {"id": "v1", "title": "T", "description": "D"},
        })
        assert c._parse_message(msg) == ("v1", "T", "D", "upsert")

    def test_video_updated_returns_upsert(self):
        c = _make_consumer()
        msg = _msg({
            "type": "video.updated",
            "payload": {"id": "v2", "title": "T2", "description": "D2"},
        })
        assert c._parse_message(msg) == ("v2", "T2", "D2", "upsert")

    def test_video_deleted_returns_delete(self):
        c = _make_consumer()
        msg = _msg({"type": "VIDEO_DELETED", "payload": {"id": "v3"}})
        assert c._parse_message(msg) == ("v3", None, None, "delete")

    def test_missing_video_id_skipped(self):
        c = _make_consumer()
        msg = _msg({"type": "VIDEO_CREATED", "payload": {"title": "no id"}})
        assert c._parse_message(msg) is None

    def test_unknown_event_type_skipped(self):
        c = _make_consumer()
        msg = _msg({"type": "VIDEO_BOOPED", "payload": {"id": "v4"}})
        assert c._parse_message(msg) is None

    def test_null_title_and_description_default_to_empty(self):
        c = _make_consumer()
        # Avro schema permits null for title/description; we must not crash.
        msg = _msg({
            "type": "VIDEO_CREATED",
            "payload": {"id": "v5", "title": None, "description": None},
        })
        assert c._parse_message(msg) == ("v5", "", "", "upsert")


class TestFlushBatch:
    def test_upserts_call_embed_batch_and_store(self):
        c = _make_consumer()
        c.llm.embed_batch.return_value = [[0.1], [0.2]]
        c.batch = [
            ("v1", "T1", "D1", "upsert"),
            ("v2", "T2", "D2", "upsert"),
        ]

        c.loop.run_until_complete(c._flush_batch())

        c.llm.embed_batch.assert_awaited_once_with(["T1. D1", "T2. D2"])
        assert c.store.store_embedding.await_count == 2
        c.store.store_embedding.assert_any_await("v1", "T1", "D1", [0.1])
        c.store.store_embedding.assert_any_await("v2", "T2", "D2", [0.2])
        assert c.batch == []

    def test_deletes_call_store_delete(self):
        c = _make_consumer()
        c.batch = [("v1", None, None, "delete"), ("v2", None, None, "delete")]

        c.loop.run_until_complete(c._flush_batch())

        c.llm.embed_batch.assert_not_called()
        c.store.delete_embeddings.assert_awaited_once_with(["v1", "v2"])

    def test_mixed_batch_handles_both(self):
        c = _make_consumer()
        c.llm.embed_batch.return_value = [[0.1]]
        c.batch = [
            ("v1", "T", "D", "upsert"),
            ("v2", None, None, "delete"),
        ]

        c.loop.run_until_complete(c._flush_batch())

        c.llm.embed_batch.assert_awaited_once_with(["T. D"])
        c.store.store_embedding.assert_awaited_once_with("v1", "T", "D", [0.1])
        c.store.delete_embeddings.assert_awaited_once_with(["v2"])

    def test_dedup_last_write_wins_within_batch(self):
        # Same video appearing as created then updated should only embed once
        # with the later payload.
        c = _make_consumer()
        c.llm.embed_batch.return_value = [[0.9]]
        c.batch = [
            ("v1", "old", "old", "upsert"),
            ("v1", "new", "new", "upsert"),
        ]

        c.loop.run_until_complete(c._flush_batch())

        c.llm.embed_batch.assert_awaited_once_with(["new. new"])
        c.store.store_embedding.assert_awaited_once_with("v1", "new", "new", [0.9])

    def test_dedup_delete_after_upsert_wins(self):
        c = _make_consumer()
        c.batch = [
            ("v1", "T", "D", "upsert"),
            ("v1", None, None, "delete"),
        ]

        c.loop.run_until_complete(c._flush_batch())

        c.llm.embed_batch.assert_not_called()
        c.store.store_embedding.assert_not_called()
        c.store.delete_embeddings.assert_awaited_once_with(["v1"])

    def test_empty_batch_is_noop(self):
        c = _make_consumer()
        c.loop.run_until_complete(c._flush_batch())
        c.llm.embed_batch.assert_not_called()
        c.store.store_embedding.assert_not_called()
        c.store.delete_embeddings.assert_not_called()


class TestIdleFlush:
    def test_should_idle_flush_when_old_enough(self):
        c = _make_consumer(idle_flush_seconds=0.0)
        c.batch = [("v1", "T", "D", "upsert")]
        c.last_flush_ts = time.monotonic() - 10
        assert c._should_idle_flush() is True

    def test_should_not_idle_flush_empty_batch(self):
        c = _make_consumer(idle_flush_seconds=0.0)
        c.last_flush_ts = time.monotonic() - 10
        assert c._should_idle_flush() is False

    def test_should_not_idle_flush_too_recent(self):
        c = _make_consumer(idle_flush_seconds=60.0)
        c.batch = [("v1", "T", "D", "upsert")]
        c.last_flush_ts = time.monotonic()
        assert c._should_idle_flush() is False


class TestDlqRouting:
    def test_to_dlq_publishes_with_diagnostic_headers(self):
        c = _make_consumer()
        c.dlq_producer.flush.return_value = 0  # all delivered
        msg = _msg({"type": "VIDEO_CREATED"}, offset=42, partition=3)

        c._to_dlq(msg, ValueError("bad payload"))

        c.dlq_producer.produce.assert_called_once()
        kwargs = c.dlq_producer.produce.call_args.kwargs
        assert kwargs["value"] == msg.value()
        headers = dict(kwargs["headers"])
        assert headers["error_type"] == b"ValueError"
        assert headers["error_message"] == b"bad payload"
        assert headers["original_offset"] == b"42"
        assert headers["original_partition"] == b"3"

    def test_to_dlq_raises_if_delivery_incomplete(self):
        c = _make_consumer()
        c.dlq_producer.flush.return_value = 1  # 1 message undelivered
        msg = _msg({"type": "VIDEO_CREATED"}, offset=7)

        with pytest.raises(RuntimeError, match="DLQ produce did not complete"):
            c._to_dlq(msg, ValueError("x"))
