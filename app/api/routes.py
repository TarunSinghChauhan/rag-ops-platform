import os
import numpy as np
from fastapi import APIRouter, HTTPException

from app.core.embedder import NgramEmbedder
from app.core.qdrant_client import VectorStore
from app.core.cache import QueryCache
from app.core.drift import DriftDetector
from app.core.metrics import drift_history
from app.models.schemas import IndexRequest, QueryRequest, QueryResponse, QueryResult

router = APIRouter()

embedder = NgramEmbedder(
    dim=int(os.getenv("EMBEDDING_DIM", "384")),
    ngram_size=int(os.getenv("NGRAM_SIZE", "3")),
)
vector_store = VectorStore()
cache = QueryCache()

# Baseline centroid starts as a zero vector; the startup seeding sets a real one.
_drift_detector = DriftDetector(
    baseline_centroid=np.zeros(embedder.dim, dtype=np.float32),
    window_size=int(os.getenv("DRIFT_WINDOW_SIZE", "50")),
    threshold=float(os.getenv("DRIFT_SIMILARITY_THRESHOLD", "0.85")),
    # Unset = wait for a full window before judging drift (see DriftDetector).
    min_samples=int(os.environ["DRIFT_MIN_SAMPLES"]) if os.getenv("DRIFT_MIN_SAMPLES") else None,
)


@router.post("/index")
def index_document(req: IndexRequest):
    # Keep the public demo from filling the free-tier database.
    if vector_store.count() >= int(os.getenv("MAX_DOCS", "200")):
        raise HTTPException(status_code=429, detail="Demo document limit reached")
    vector = embedder.embed(req.text)
    # The real text goes last: metadata {"text": ...} used to silently replace it, so search
    # results showed text that was never embedded.
    vector_store.upsert(req.doc_id, vector, {**req.metadata, "text": req.text})
    # Cached answers were computed against the old corpus; without this a new or changed
    # document is invisible to every previously asked query until the TTL expires.
    cache.invalidate()
    return {"status": "indexed", "doc_id": req.doc_id}


@router.post("/reindex-baseline")
def reindex_baseline():
    # The baseline is the centroid of the recent queries, so with none there is nothing to
    # reset to. This used to answer "baseline reset" without having changed anything.
    if not _drift_detector.reindex_baseline():
        raise HTTPException(status_code=409, detail="No recent queries in the window; baseline unchanged")
    return {"status": "baseline reset"}


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # Drift is tracked on every incoming query, including cached ones.
    vector = embedder.embed(req.query)
    drift_result = _drift_detector.add_embedding(vector)
    if drift_result["similarity"] is not None:
        drift_history.append(drift_result["similarity"])

    cached = cache.get(req.query, req.top_k)
    if cached:
        return QueryResponse(results=cached["results"], cache_hit=True, drift=drift_result)

    hits = vector_store.search(vector, top_k=req.top_k)
    results = [
        QueryResult(doc_id=h.id, score=h.score, payload=h.payload or {})
        for h in hits
    ]

    response = QueryResponse(results=results, cache_hit=False, drift=drift_result)
    cache.set(req.query, {"results": [r.model_dump() for r in results]}, req.top_k)
    return response


@router.get("/drift/events")
def drift_events():
    return {"events": _drift_detector.drift_events, "count": len(_drift_detector.drift_events)}


@router.get("/health")
def health():
    return {"status": "ok"}
