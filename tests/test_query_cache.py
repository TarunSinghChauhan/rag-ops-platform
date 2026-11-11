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
        mock_redis_cls.assert_called_once_with(
            host="myhost", port=1234, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5,
        )


def test_init_defaults_to_localhost_and_6379(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    with patch("app.core.cache.redis.Redis") as mock_redis_cls:
        QueryCache()
        mock_redis_cls.assert_called_once_with(
            host="localhost", port=6379, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5,
        )


def test_redis_url_takes_precedence_over_host_and_port(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "rediss://user:pw@cache.example:6380/0")
    monkeypatch.setenv("REDIS_HOST", "ignored")
    with patch("app.core.cache.redis.from_url") as from_url, \
         patch("app.core.cache.redis.Redis") as redis_cls:
        QueryCache()
    from_url.assert_called_once_with(
        "rediss://user:pw@cache.example:6380/0",
        decode_responses=True, socket_timeout=5, socket_connect_timeout=5,
    )
    redis_cls.assert_not_called()


def test_key_depends_on_top_k():
    assert QueryCache._key("what is rag", 1) != QueryCache._key("what is rag", 10)
    assert QueryCache._key("what is rag", 5) == QueryCache._key("what is rag", 5)
    assert QueryCache._key("what is rag") == QueryCache._key("what is rag", 5)


def test_key_cannot_be_forged_by_moving_digits_between_top_k_and_query():
    assert QueryCache._key("0 apples", 1) != QueryCache._key("apples", 10)
    assert QueryCache._key("1\n2", 3) != QueryCache._key("2", 13)


def test_get_and_set_use_the_same_top_k_scoped_key(cache):
    cache.set("q", {"results": []}, top_k=3)
    stored_key = cache.client.setex.call_args.args[0]
    cache.client.get.return_value = None
    cache.get("q", top_k=3)
    cache.client.get.assert_called_with(stored_key)
    cache.get("q", top_k=4)
    assert cache.client.get.call_args.args[0] != stored_key
