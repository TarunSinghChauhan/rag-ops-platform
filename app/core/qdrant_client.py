"""
Thin wrapper around Qdrant. Connects to Qdrant Cloud when QDRANT_URL is set,
otherwise falls back to a local instance (docker-compose).
"""
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    def __init__(self):
        self.collection = os.getenv("QDRANT_COLLECTION", "rag_ops_docs")
        self.dim = int(os.getenv("EMBEDDING_DIM", "384"))

        url = os.getenv("QDRANT_URL")
        if url:
            self.client = QdrantClient(
                url=url,
                api_key=os.getenv("QDRANT_API_KEY"),
                timeout=30,
            )
        else:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            self.client = QdrantClient(host=host, port=port)

        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count

    def upsert(self, point_id: int, vector: np.ndarray, payload: dict):
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
        )

    def search(self, vector: np.ndarray, top_k: int = 5):
        # Newer qdrant-client versions use query_points; older ones use search.
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=vector.tolist(),
                limit=top_k,
            )
            return res.points
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
        )
