"""Tests for the recommendation API."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200


@patch("src.api.routes.recommend.get_recommendations")
def test_recommend(mock_get_reco):
    mock_get_reco.return_value = [
        {"video_id": "vid-1", "title": "Test Video", "score": 0.9, "reason": "Relevant"},
    ]

    response = client.post("/api/v1/recommend", json={
        "user_id": "user-1",
        "limit": 5,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "user-1"
    assert len(data["recommendations"]) == 1


@patch("src.api.routes.recommend.get_recommendations")
def test_recommend_with_query(mock_get_reco):
    mock_get_reco.return_value = []

    response = client.post("/api/v1/recommend", json={
        "user_id": "user-1",
        "query": "python tutorial",
        "limit": 10,
    })
    assert response.status_code == 200
