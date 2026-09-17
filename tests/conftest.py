"""
Mocks Qdrant/Redis clients BEFORE any test module imports app.api.routes,
since that module constructs real VectorStore()/QueryCache() instances at
import time. This must run at conftest load time, not inside a fixture,
because fixtures execute after test modules are already imported.
"""
from unittest.mock import MagicMock, patch

_qdrant_patcher = patch("app.core.qdrant_client.QdrantClient")
_redis_patcher = patch("app.core.cache.redis.Redis")

_mock_qdrant_cls = _qdrant_patcher.start()
_mock_redis_cls = _redis_patcher.start()

_mock_qdrant = MagicMock()
_collections_result = MagicMock()
_collections_result.collections = []
_mock_qdrant.get_collections.return_value = _collections_result
_mock_qdrant_cls.return_value = _mock_qdrant

_mock_redis_cls.return_value = MagicMock()
