"""
Local Redis cache wrapper for repeated-query caching. No API key required —
connects to the Redis container started by docker-compose.
"""
import os
import json
import hashlib
import redis


class QueryCache:
    def __init__(self, ttl_seconds: int = 3600):
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl_seconds

    @staticmethod
    def _key(query: str) -> str:
        return "ragops:query:" + hashlib.sha256(query.encode("utf-8")).hexdigest()

    def get(self, query: str):
        raw = self.client.get(self._key(query))
        return json.loads(raw) if raw else None

    def set(self, query: str, result: dict):
        self.client.setex(self._key(query), self.ttl, json.dumps(result))
