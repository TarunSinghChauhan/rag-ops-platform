"""New documents must invalidate cached answers, and /index must reject input the store cannot take."""
from unittest.mock import MagicMock, patch

import pytest
import redis
from fastapi.testclient import TestClient

import app.api.routes as routes_module
import app.main as main_module
from app.core.cache import QueryCache
from app.models.schemas import IndexRequest


class FakeRedis:
    """Just enough of Redis for QueryCache: get, setex, scan_iter, delete."""

    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value

    def scan_iter(self, match=None, count=None):
        prefix = match.rstrip("*")
        return iter([k for k in list(self.data) if k.startswith(prefix)])

    def delete(self, *keys):
        for k in keys:
            self.data.pop(k, None)
        return len(keys)


def make_cache(client=None):
    with patch("app.core.cache.redis.Redis"):
        c = QueryCache(ttl_seconds=60)
    c.client = client if client is not None else FakeRedis()
    return c


# ---- QueryCache.invalidate ----------------------------------------------------

def test_invalidate_removes_cached_results_and_nothing_else():
    c = make_cache()
    c.set("a", {"results": [1]})
    c.set("b", {"results": [2]}, top_k=10)
    c.client.data["someone:else"] = "keep me"
    assert c.invalidate() == 2
    assert c.get("a") is None and c.get("b", 10) is None
    assert c.client.data == {"someone:else": "keep me"}


def test_invalidate_on_an_empty_cache_deletes_nothing():
    client = MagicMock()
    client.scan_iter.return_value = iter([])
    assert make_cache(client).invalidate() == 0
    client.delete.assert_not_called()


def test_invalidate_deletes_in_batches():
    client = MagicMock()
    client.scan_iter.return_value = iter([f"ragops:query:{i}" for i in range(450)])
    assert make_cache(client).invalidate() == 450
    assert [len(call.args) for call in client.delete.call_args_list] == [200, 200, 50]


def test_invalidate_uses_only_this_caches_prefix():
    client = MagicMock()
    client.scan_iter.return_value = iter([])
    make_cache(client).invalidate()
    assert client.scan_iter.call_args.kwargs["match"] == "ragops:query:*"


def test_a_redis_outage_during_invalidate_is_not_an_error():
    client = MagicMock()
    client.scan_iter.side_effect = redis.ConnectionError("down")
    assert make_cache(client).invalidate() == 0


# ---- /index ---------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_index_invalidates_the_cache_after_the_upsert(client):
    order = MagicMock()
    with patch.object(routes_module.vector_store, "count", return_value=0), \
         patch.object(routes_module.vector_store, "upsert", order.upsert), \
         patch.object(routes_module.cache, "invalidate", order.invalidate):
        resp = client.post("/index", json={"doc_id": 1, "text": "hello"})
    assert resp.status_code == 200
    assert [c[0] for c in order.mock_calls] == ["upsert", "invalidate"]


def test_a_rejected_index_leaves_the_cache_alone(client):
    with patch.object(routes_module.vector_store, "count", return_value=10**6), \
         patch.object(routes_module.cache, "invalidate") as inv:
        resp = client.post("/index", json={"doc_id": 1, "text": "hello"})
    assert resp.status_code == 429
    inv.assert_not_called()


def test_metadata_cannot_replace_the_indexed_text(client):
    with patch.object(routes_module.vector_store, "count", return_value=0), \
         patch.object(routes_module.vector_store, "upsert") as up, \
         patch.object(routes_module.cache, "invalidate"):
        client.post("/index", json={"doc_id": 1, "text": "the real text",
                                    "metadata": {"text": "spoofed", "source": "x"}})
    assert up.call_args.args[2] == {"source": "x", "text": "the real text"}


def test_a_new_document_shows_up_in_a_query_that_was_already_cached(client):
    """The whole bug: ask, get cached, index a document, ask again."""
    hit_old = MagicMock(id=1, score=0.9, payload={"text": "old"})
    hit_new = MagicMock(id=2, score=0.95, payload={"text": "new"})
    corpus = {"hits": [hit_old]}
    fake = make_cache()
    with patch.object(routes_module, "cache", fake), \
         patch.object(routes_module.vector_store, "count", return_value=0), \
         patch.object(routes_module.vector_store, "search", side_effect=lambda *a, **k: corpus["hits"]), \
         patch.object(routes_module.vector_store, "upsert", side_effect=lambda *a, **k: corpus.update(hits=[hit_new, hit_old])):
        first = client.post("/query", json={"query": "what is rag"}).json()
        again = client.post("/query", json={"query": "what is rag"}).json()
        client.post("/index", json={"doc_id": 2, "text": "new document about rag"})
        after = client.post("/query", json={"query": "what is rag"}).json()
    assert first["cache_hit"] is False and again["cache_hit"] is True
    assert after["cache_hit"] is False
    assert [r["doc_id"] for r in after["results"]] == [2, 1]


# ---- input validation -----------------------------------------------------------

@pytest.mark.parametrize("doc_id", [-1, -10**9])
def test_negative_ids_are_rejected(client, doc_id):
    assert client.post("/index", json={"doc_id": doc_id, "text": "x"}).status_code == 422


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_text_is_rejected(client, text):
    assert client.post("/index", json={"doc_id": 1, "text": text}).status_code == 422


def test_oversized_text_is_rejected(client):
    assert client.post("/index", json={"doc_id": 1, "text": "x" * 20001}).status_code == 422


def test_boundary_values_are_accepted():
    assert IndexRequest(doc_id=0, text="x").doc_id == 0
    assert IndexRequest(doc_id=1, text="x" * 20000).text == "x" * 20000
    assert IndexRequest(doc_id=1, text="  padded  ").text == "padded"
