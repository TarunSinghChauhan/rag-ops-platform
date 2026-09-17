from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import app.api.routes as routes_module


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_root_returns_welcome_message(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["docs"] == "/docs"
    assert "RAG Ops Platform" in body["message"]


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_document_embeds_and_upserts(client):
    fake_vector = np.zeros(routes_module.embedder.dim, dtype=np.float32)
    with patch.object(routes_module.embedder, "embed", return_value=fake_vector) as mock_embed, \
         patch.object(routes_module.vector_store, "upsert") as mock_upsert:
        resp = client.post("/index", json={"doc_id": 1, "text": "hello world", "metadata": {"source": "test"}})
    assert resp.status_code == 200
    assert resp.json() == {"status": "indexed", "doc_id": 1}
    mock_embed.assert_called_once_with("hello world")
    mock_upsert.assert_called_once()
    args, kwargs = mock_upsert.call_args
    assert args[0] == 1
    assert args[2] == {"text": "hello world", "source": "test"}


def test_reindex_baseline_resets_detector(client):
    with patch.object(routes_module._drift_detector, "reindex_baseline") as mock_reset:
        resp = client.post("/reindex-baseline")
    assert resp.status_code == 200
    assert resp.json() == {"status": "baseline reset"}
    mock_reset.assert_called_once()


def test_query_returns_cached_result_on_cache_hit(client):
    cached_payload = {"results": [], "drift": {"score": 0.0}}
    with patch.object(routes_module.cache, "get", return_value=cached_payload):
        resp = client.post("/query", json={"query": "what is rag"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is True


def test_query_computes_fresh_result_on_cache_miss(client):
    fake_vector = np.zeros(routes_module.embedder.dim, dtype=np.float32)
    fake_hit = MagicMock(id=1, score=0.9, payload={"text": "chunk"})
    with patch.object(routes_module.cache, "get", return_value=None), \
         patch.object(routes_module.embedder, "embed", return_value=fake_vector), \
         patch.object(routes_module._drift_detector, "add_embedding", return_value={"score": 0.05}), \
         patch.object(routes_module.vector_store, "search", return_value=[fake_hit]), \
         patch.object(routes_module.cache, "set") as mock_set:
        resp = client.post("/query", json={"query": "what is rag", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cache_hit"] is False
    assert body["results"][0]["doc_id"] == 1
    assert body["drift"] == {"score": 0.05}
    mock_set.assert_called_once()


def test_query_handles_hit_with_no_payload(client):
    fake_vector = np.zeros(routes_module.embedder.dim, dtype=np.float32)
    fake_hit = MagicMock(id=2, score=0.5, payload=None)
    with patch.object(routes_module.cache, "get", return_value=None), \
         patch.object(routes_module.embedder, "embed", return_value=fake_vector), \
         patch.object(routes_module._drift_detector, "add_embedding", return_value={"score": 0.0}), \
         patch.object(routes_module.vector_store, "search", return_value=[fake_hit]), \
         patch.object(routes_module.cache, "set"):
        resp = client.post("/query", json={"query": "another query"})
    body = resp.json()
    assert body["results"][0]["payload"] == {}


def test_drift_events_returns_events_and_count(client):
    fake_events = [{"index": 1}, {"index": 2}]
    with patch.object(routes_module._drift_detector, "drift_events", fake_events):
        resp = client.get("/drift/events")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["events"] == fake_events
