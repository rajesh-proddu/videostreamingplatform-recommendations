"""Tests for recommendation tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.search_videos import search_videos


@pytest.mark.asyncio
@patch("src.tools.search_videos.AsyncElasticsearch")
async def test_search_videos(mock_es_class):
    mock_es = AsyncMock()
    mock_es_class.return_value = mock_es
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "vid-1", "_source": {"title": "Python Tutorial", "description": "Learn Python"}},
                {"_id": "vid-2", "_source": {"title": "Go Tutorial", "description": "Learn Go"}},
            ]
        }
    }

    results = await search_videos("tutorial")
    assert len(results) == 2
    assert results[0]["id"] == "vid-1"
    assert results[0]["title"] == "Python Tutorial"
