"""API contract tests for the recommendation FastAPI service.

Covers: limit bounds (0, 51, 1, 50), missing user_id, error path 500.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_limit_zero_rejected():
    response = client.post("/api/v1/recommend", json={"user_id": "u", "limit": 0})
    assert response.status_code == 422


def test_limit_too_high_rejected():
    response = client.post("/api/v1/recommend", json={"user_id": "u", "limit": 51})
    assert response.status_code == 422


def test_limit_one_accepted():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.return_value = []
        response = client.post("/api/v1/recommend", json={"user_id": "u", "limit": 1})
        assert response.status_code == 200


def test_limit_max_accepted():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.return_value = []
        response = client.post("/api/v1/recommend", json={"user_id": "u", "limit": 50})
        assert response.status_code == 200


def test_missing_user_id_rejected():
    response = client.post("/api/v1/recommend", json={"limit": 10})
    assert response.status_code == 422


def test_empty_body_rejected():
    response = client.post("/api/v1/recommend", json={})
    assert response.status_code == 422


def test_default_limit_is_ten():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.return_value = []
        client.post("/api/v1/recommend", json={"user_id": "u"})
        assert mock.call_args.kwargs["limit"] == 10


def test_query_passthrough():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.return_value = []
        client.post("/api/v1/recommend", json={"user_id": "u", "query": "needle"})
        assert mock.call_args.kwargs["query"] == "needle"


def test_internal_error_returns_500():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.side_effect = RuntimeError("graph blew up")
        response = client.post("/api/v1/recommend", json={"user_id": "u", "limit": 5})
        assert response.status_code == 500
        assert "graph blew up" in response.json()["detail"]


def test_response_shape_preserves_user_and_query():
    with patch("src.api.routes.recommend.get_recommendations") as mock:
        mock.return_value = [
            {"video_id": "v1", "title": "T", "score": 0.5, "reason": "because"},
        ]
        response = client.post("/api/v1/recommend", json={
            "user_id": "u-shape",
            "query": "q-shape",
            "limit": 5,
        })
        body = response.json()
        assert body["user_id"] == "u-shape"
        assert body["query"] == "q-shape"
        assert body["recommendations"][0]["video_id"] == "v1"
