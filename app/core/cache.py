"""
Redis cache wrapper for repeated-query caching. Uses REDIS_URL (Upstash) when
set, otherwise a local Redis. Tracks hit/miss counts for the ops dashboard.
"""
import os
import json
import hashlib
import redis


class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        url = os.getenv("REDIS_URL")
        if url:
            self.client = redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        else:
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str, top_k: int = 5) -> str:
        # top_k is part of the key: a cached 1-result answer must not be served to a
        # request that asked for 10 results for the same text.
        material = f"{top_k}\n{query}"
        return "ragops:query:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int = 5):
        try:
            raw = self.client.get(self._key(query, top_k))
        except redis.RedisError:
            return None  # cache outage should never break a query
        if raw:
            self.hits += 1
            return json.loads(raw)
        self.misses += 1
        return None

    def set(self, query: str, result: dict, top_k: int = 5):
        try:
            self.client.setex(self._key(query, top_k), self.ttl, json.dumps(result))
        except redis.RedisError:
            pass

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
