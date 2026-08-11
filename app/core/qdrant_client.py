"""
Thin wrapper around a local Qdrant instance. No cloud account, no API key —
this connects to the Qdrant container started by docker-compose.
"""
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection = os.getenv("QDRANT_COLLECTION", "rag_ops_docs")
        self.dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, point_id: int, vector: np.ndarray, payload: dict):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
        )

    def search(self, vector: np.ndarray, top_k: int = 5):
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
        )
