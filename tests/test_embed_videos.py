"""Tests for the embed_all_videos batch job."""

from unittest.mock import AsyncMock, patch

import pytest

from src.embeddings.embed_videos import embed_all_videos


@pytest.mark.asyncio
@patch("src.embeddings.embed_videos.get_llm_provider")
@patch("src.embeddings.embed_videos.AsyncElasticsearch")
@patch("src.embeddings.embed_videos.EmbeddingStore")
async def test_embed_all_videos_processes_hits(mock_store_class, mock_es_class, mock_get_llm):
    # Setup mock LLM
    mock_llm = AsyncMock()
    mock_llm.embed.return_value = [0.1, 0.2, 0.3]
    mock_get_llm.return_value = mock_llm

    # Setup mock ES with one page of results then empty
    mock_es = AsyncMock()
    mock_es_class.return_value = mock_es
    mock_es.search.return_value = {
        "_scroll_id": "scroll-1",
        "hits": {
            "hits": [
                {"_id": "vid-1", "_source": {"title": "Python", "description": "Learn Python"}},
                {"_id": "vid-2", "_source": {"title": "Go", "description": "Learn Go"}},
            ]
        },
    }
    mock_es.scroll.return_value = {
        "_scroll_id": "scroll-2",
        "hits": {"hits": []},
    }

    # Setup mock store
    mock_store = AsyncMock()
    mock_store_class.return_value = mock_store

    await embed_all_videos()

    assert mock_llm.embed.call_count == 2
    assert mock_store.store_embedding.call_count == 2
    mock_store.initialize.assert_called_once()
    mock_store.close.assert_called_once()
    mock_es.close.assert_called_once()


@pytest.mark.asyncio
@patch("src.embeddings.embed_videos.get_llm_provider")
@patch("src.embeddings.embed_videos.AsyncElasticsearch")
@patch("src.embeddings.embed_videos.EmbeddingStore")
async def test_embed_all_videos_handles_embed_failure(mock_store_class, mock_es_class, mock_get_llm):
    mock_llm = AsyncMock()
    mock_llm.embed.side_effect = [Exception("LLM down"), [0.1]]
    mock_get_llm.return_value = mock_llm

    mock_es = AsyncMock()
    mock_es_class.return_value = mock_es
    mock_es.search.return_value = {
        "_scroll_id": "scroll-1",
        "hits": {
            "hits": [
                {"_id": "vid-1", "_source": {"title": "Fail", "description": ""}},
                {"_id": "vid-2", "_source": {"title": "OK", "description": ""}},
            ]
        },
    }
    mock_es.scroll.return_value = {"_scroll_id": "scroll-2", "hits": {"hits": []}}

    mock_store = AsyncMock()
    mock_store_class.return_value = mock_store

    await embed_all_videos()

    # First embed fails, second succeeds
    assert mock_store.store_embedding.call_count == 1
    mock_store.close.assert_called_once()


@pytest.mark.asyncio
@patch("src.embeddings.embed_videos.get_llm_provider")
@patch("src.embeddings.embed_videos.AsyncElasticsearch")
@patch("src.embeddings.embed_videos.EmbeddingStore")
async def test_embed_all_videos_empty_index(mock_store_class, mock_es_class, mock_get_llm):
    mock_llm = AsyncMock()
    mock_get_llm.return_value = mock_llm

    mock_es = AsyncMock()
    mock_es_class.return_value = mock_es
    mock_es.search.return_value = {
        "_scroll_id": "scroll-1",
        "hits": {"hits": []},
    }

    mock_store = AsyncMock()
    mock_store_class.return_value = mock_store

    await embed_all_videos()

    mock_llm.embed.assert_not_called()
    mock_store.store_embedding.assert_not_called()
