import os
import numpy as np
from fastapi import APIRouter

from app.core.embedder import NgramEmbedder
from app.core.qdrant_client import VectorStore
from app.core.cache import QueryCache
from app.core.drift import DriftDetector
from app.models.schemas import IndexRequest, QueryRequest, QueryResponse, QueryResult

router = APIRouter()

embedder = NgramEmbedder(
    dim=int(os.getenv("EMBEDDING_DIM", "384")),
    ngram_size=int(os.getenv("NGRAM_SIZE", "3")),
)
vector_store = VectorStore()
cache = QueryCache()

# Baseline centroid starts as a zero vector until the first documents are indexed;
# call /reindex-baseline after bulk-loading a corpus to set a real baseline.
_drift_detector = DriftDetector(
    baseline_centroid=np.zeros(embedder.dim, dtype=np.float32),
    window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "50")),
    threshold=float(os.getenv("DRIFT_SIMILARITY_THRESHOLD", "0.85")),
)


@router.post("/index")
def index_document(req: IndexRequest):
    vector = embedder.embed(req.text)
    vector_store.upsert(req.doc_id, vector, {"text": req.text, **req.metadata})
    return {"status": "indexed", "doc_id": req.doc_id}


@router.post("/reindex-baseline")
def reindex_baseline():
    _drift_detector.reindex_baseline()
    return {"status": "baseline reset"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    cached = cache.get(req.query)
    if cached:
        return QueryResponse(**cached, cache_hit=True)

    vector = embedder.embed(req.query)
    drift_result = _drift_detector.add_embedding(vector)

    hits = vector_store.search(vector, top_k=req.top_k)
    results = [
        QueryResult(doc_id=h.id, score=h.score, payload=h.payload or {})
        for h in hits
    ]

    response = QueryResponse(results=results, cache_hit=False, drift=drift_result)
    cache.set(req.query, {"results": [r.model_dump() for r in results], "drift": drift_result})
    return response


@router.get("/drift/events")
def drift_events():
    return {"events": _drift_detector.drift_events, "count": len(_drift_detector.drift_events)}


@router.get("/health")
def health():
    return {"status": "ok"}
