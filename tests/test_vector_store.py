from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.qdrant_client import VectorStore


def make_store(existing_collections=None):
    """Builds a VectorStore with QdrantClient mocked, returns (store, mock_client)."""
    with patch("app.core.qdrant_client.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        collections_result = MagicMock()
        collections_result.collections = [
            MagicMock(name=name) for name in (existing_collections or [])
        ]
        # MagicMock(name=...) sets the mock's repr name, not the .name attribute — set explicitly
        for m, name in zip(collections_result.collections, existing_collections or []):
            m.name = name
        mock_client.get_collections.return_value = collections_result
        mock_cls.return_value = mock_client
        store = VectorStore()
        return store, mock_client


def test_init_creates_collection_when_missing():
    store, mock_client = make_store(existing_collections=[])
    mock_client.create_collection.assert_called_once()
    _, kwargs = mock_client.create_collection.call_args
    assert kwargs["collection_name"] == store.collection


def test_init_skips_creation_when_collection_exists():
    store, mock_client = make_store(existing_collections=["rag_ops_docs"])
    mock_client.create_collection.assert_not_called()


def test_init_reads_env_vars(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "myhost")
    monkeypatch.setenv("QDRANT_PORT", "9999")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom_docs")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
    with patch("app.core.qdrant_client.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        collections_result = MagicMock()
        collections_result.collections = []
        mock_client.get_collections.return_value = collections_result
        mock_cls.return_value = mock_client
        store = VectorStore()
    mock_cls.assert_called_once_with(host="myhost", port=9999)
    assert store.collection == "custom_docs"
    assert store.dim == 768


def test_upsert_sends_point_with_vector_as_list():
    store, mock_client = make_store(existing_collections=["rag_ops_docs"])
    vector = np.array([0.1, 0.2, 0.3])
    store.upsert(point_id=5, vector=vector, payload={"text": "chunk"})
    _, kwargs = mock_client.upsert.call_args
    assert kwargs["collection_name"] == store.collection
    point = kwargs["points"][0]
    assert point.id == 5
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {"text": "chunk"}


def test_search_passes_vector_as_list_and_top_k():
    store, mock_client = make_store(existing_collections=["rag_ops_docs"])
    mock_client.search.return_value = ["result1", "result2"]
    vector = np.array([0.5, 0.6])
    results = store.search(vector, top_k=3)
    _, kwargs = mock_client.search.call_args
    assert kwargs["collection_name"] == store.collection
    assert kwargs["query_vector"] == [0.5, 0.6]
    assert kwargs["limit"] == 3
    assert results == ["result1", "result2"]
