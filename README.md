# Real-Time RAG Ops Platform with Drift Detection

A production-style RAG observability platform: deterministic (zero-API-cost) embeddings,
Qdrant vector search, Redis caching, and cosine-similarity-based embedding drift detection
with automated re-indexing triggers.

## Why "zero API cost"

This project intentionally avoids any external LLM/embedding API (OpenAI, Cohere, etc.).
Instead it uses a **deterministic n-gram hashing embedder** — pure Python, no network calls,
no API keys, no bill. This keeps the whole stack runnable offline with just Docker.

| Component     | Tool                          | Requires API key? |
|----------------|-------------------------------|--------------------|
| Embeddings     | Deterministic n-gram hashing  | No                 |
| Vector DB      | Qdrant (self-hosted, Docker)  | No                 |
| Cache          | Redis (self-hosted, Docker)   | No                 |
| Backend        | FastAPI                       | No                 |
| Dashboard      | Vanilla JS + Chart.js         | No                 |

## Quickstart

```bash
docker compose up -d        # starts Qdrant + Redis locally
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API, or `http://localhost:8000/dashboard`
for the ops dashboard (once added).

## How drift detection works

1. Each incoming document/query is embedded via the n-gram hasher into a fixed 384-dim vector.
2. A rolling window of recent embedding centroids is compared against a baseline centroid
   (computed at index time) using cosine similarity.
3. If similarity drops below a configurable threshold, a drift event is logged and an
   automated re-indexing job is triggered.

## Project layout

```
app/
  core/
    embedder.py       # deterministic n-gram embedding (no API)
    drift.py          # cosine similarity drift detection
    qdrant_client.py  # local Qdrant wrapper
    cache.py          # Redis wrapper
  api/
    routes.py         # FastAPI endpoints
  models/
    schemas.py        # pydantic request/response models
  main.py             # FastAPI app entrypoint
tests/
docker-compose.yml     # Qdrant + Redis, both local, no keys
.env.example            # non-secret config only (host/port)
.gitignore
```

## Config

All config lives in `.env` (copy from `.env.example`). Nothing in there is a secret —
just local hostnames/ports for Qdrant and Redis.
