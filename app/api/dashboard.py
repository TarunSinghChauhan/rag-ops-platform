import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.api.routes import embedder, vector_store, cache, _drift_detector
from app.core.drift import cosine_similarity
from app.core.metrics import latency, drift_history
from app.core.seed_data import SEED_DOCS, EVAL_QUERIES, OFF_TOPIC_QUERIES

dash_router = APIRouter()

_DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "static" / "dashboard.html"
LATENCY_SLO_MS = float(os.getenv("LATENCY_SLO_MS", "500"))

_eval_result = {"k": 5, "hit_at_5": None, "hits": 0, "total": len(EVAL_QUERIES), "details": []}


def run_eval(k: int = 5) -> dict:
    """Hit@k: share of labeled test queries whose expected doc is in the top k."""
    global _eval_result
    details, hits = [], 0
    for query, expected in EVAL_QUERIES:
        found = vector_store.search(embedder.embed(query), top_k=k)
        ids = [h.id for h in found]
        ok = expected in ids
        hits += int(ok)
        details.append({"query": query, "expected": expected, "retrieved": ids, "hit": ok})
    _eval_result = {
        "k": k,
        "hit_at_5": round(hits / len(EVAL_QUERIES), 3),
        "hits": hits,
        "total": len(EVAL_QUERIES),
        "details": details,
    }
    return _eval_result


def seed_if_empty():
    """Index the sample corpus once, set the drift baseline, and run the Hit@5 eval."""
    try:
        if vector_store.count() == 0:
            for doc_id, text in SEED_DOCS:
                vector_store.upsert(doc_id, embedder.embed(text), {"text": text, "source": "seed"})
        vectors = np.stack([embedder.embed(text) for _, text in SEED_DOCS])
        _drift_detector.baseline_centroid = vectors.mean(axis=0)
        run_eval()
    except Exception as exc:
        print(f"[startup] seeding skipped: {exc}")


@dash_router.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(_DASHBOARD_PATH.read_text(encoding="utf-8"))


@dash_router.get("/metrics")
def metrics():
    window_n = len(_drift_detector.window)
    similarity = None
    if window_n:
        current = np.mean(np.stack(_drift_detector.window), axis=0)
        similarity = round(cosine_similarity(_drift_detector.baseline_centroid, current), 4)
    try:
        docs = vector_store.count()
    except Exception:
        docs = None
    return {
        "latency": {**latency.summary(), "slo_ms": LATENCY_SLO_MS},
        "cache": cache.stats(),
        "drift": {
            "current_similarity": similarity,
            "threshold": _drift_detector.threshold,
            "window_size": window_n,
            "events": len(_drift_detector.drift_events),
            "history": [round(float(x), 4) for x in drift_history],
        },
        "retrieval": {k: _eval_result[k] for k in ("k", "hit_at_5", "hits", "total")},
        "documents": docs,
    }


@dash_router.get("/eval")
def eval_details():
    return run_eval()


@dash_router.post("/simulate-drift")
def simulate_drift():
    """Fill the window with off-topic queries so the similarity visibly drops."""
    result = None
    for i in range(_drift_detector.window.maxlen):
        text = OFF_TOPIC_QUERIES[i % len(OFF_TOPIC_QUERIES)]
        result = _drift_detector.add_embedding(embedder.embed(text))
        drift_history.append(result["similarity"])
    return result


@dash_router.post("/reset-drift")
def reset_drift():
    _drift_detector.window.clear()
    _drift_detector.drift_events.clear()
    drift_history.clear()
    return {"status": "reset"}
