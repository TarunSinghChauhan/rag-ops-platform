import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.cache import QueryCache


@pytest.fixture
def cache():
    with patch("app.core.cache.redis.Redis") as mock_redis_cls:
        mock_client = MagicMock()
        mock_redis_cls.return_value = mock_client
        c = QueryCache(ttl_seconds=120)
        c.client = mock_client  # keep a direct handle for assertions
        yield c


def test_key_is_deterministic_sha256_hash():
    key1 = QueryCache._key("what is rag")
    key2 = QueryCache._key("what is rag")
    key3 = QueryCache._key("different query")
    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("ragops:query:")


def test_get_returns_none_when_cache_miss(cache):
    cache.client.get.return_value = None
    result = cache.get("some query")
    assert result is None


def test_get_returns_parsed_json_on_cache_hit(cache):
    cache.client.get.return_value = json.dumps({"answer": "42"})
    result = cache.get("some query")
    assert result == {"answer": "42"}


def test_set_stores_json_with_configured_ttl(cache):
    cache.set("some query", {"answer": "42"})
    args, kwargs = cache.client.setex.call_args
    key, ttl, payload = args
    assert key == QueryCache._key("some query")
    assert ttl == 120
    assert json.loads(payload) == {"answer": "42"}


def test_init_reads_redis_host_and_port_from_env(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "myhost")
    monkeypatch.setenv("REDIS_PORT", "1234")
    with patch("app.core.cache.redis.Redis") as mock_redis_cls:
        QueryCache()
        mock_redis_cls.assert_called_once_with(host="myhost", port=1234, decode_responses=True)


def test_init_defaults_to_localhost_and_6379(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    with patch("app.core.cache.redis.Redis") as mock_redis_cls:
        QueryCache()
        mock_redis_cls.assert_called_once_with(host="localhost", port=6379, decode_responses=True)
