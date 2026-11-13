# Real-Time RAG Ops Platform with Drift Detection

A production-style retrieval observability platform: deterministic (zero-API-cost) embeddings, Qdrant vector search, Redis caching, cosine-similarity embedding drift detection with a manual baseline reset, and a live ops dashboard.

**Live demo:** [rag-ops-platform-7dzz.onrender.com/dashboard](https://rag-ops-platform-7dzz.onrender.com/dashboard)
*(Hosted on Render's free tier. If the instance is asleep, the first load can take up to ~50 seconds.)*

## What it does

- **Retrieves** documents with Qdrant vector search, behind a Redis query cache.
- **Measures itself:** latency percentiles (p50 / p95 / p99) against an SLO, cache hit rate, retrieval quality (Hit@5) and embedding drift, all shown live on a dashboard.
- **Detects drift:** compares recent embeddings to a baseline and flags when they diverge.

## Dashboard

The dashboard refreshes every 4 seconds and has three controls: **Run sample traffic**, **Simulate drift** and **Reset drift**.

Example readings from one run of the live instance after sample traffic:

| Metric | Value | Notes |
| --- | --- | --- |
| Latency p50 | 7.8 ms | 16 requests measured |
| Latency p95 | 32.6 ms | SLO of 500 ms: met |
| Latency p99 | 82.7 ms | Includes the first, cold-start request |
| Cache hit rate | 50% | 8 hits / 8 misses |
| Retrieval Hit@5 | 100% | 18 of 18 labeled test queries |
| Drift similarity | 0.653 | Threshold 0.6 |
| Documents indexed | 20 | |

These are single-run demo numbers on a small corpus, not production benchmarks. See [Limitations](#limitations).

## Why "zero API cost"

This project deliberately uses no external LLM or embedding API (OpenAI, Cohere and so on). It uses a **deterministic n-gram hashing embedder**: pure Python, no network calls, no API keys, no bill. The whole stack runs offline with just Docker.

| Component | Tool | Requires API key? |
| --- | --- | --- |
| Embeddings | Deterministic n-gram hashing | No |
| Vector DB | Qdrant (self-hosted, Docker) | No |
| Cache | Redis (self-hosted, Docker) | No |
| Backend | FastAPI | No |
| Dashboard | Vanilla JS + Chart.js | No |

**Trade-off:** n-gram hashing matches on text overlap, not meaning, so it will not find paraphrases the way a learned semantic embedding model would. The embedder sits in one module (`app/core/embedder.py`) and can be swapped for a semantic model without changing the rest of the pipeline.

## Quickstart

```bash
docker compose up -d        # starts Qdrant + Redis locally
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Interactive API docs: <http://localhost:8000/docs>
- Ops dashboard: <http://localhost:8000/dashboard>

Click **Run sample traffic** on the dashboard to generate queries and populate the metrics.

## How the metrics are computed

- **Latency percentiles:** recorded per `/query` request and computed over the recorded window. p95/p99 are only meaningful once enough requests have been measured.
- **Cache hit rate:** Redis hits divided by total lookups. It depends on the query mix. Repeated queries raise it, unique queries lower it.
- **Hit@5:** for each query in a small labeled test set, whether the expected document appears in the top 5 results, averaged over the set.
- **Drift similarity:** cosine similarity between the centroid of recent embeddings and a baseline centroid computed at index time.

## How drift detection works

1. Each incoming document or query is embedded by the n-gram hasher into a fixed 384-dimensional vector.
2. A rolling window of recent embedding centroids is compared against the baseline centroid using cosine similarity.
3. Once the window is full, if similarity drops below the threshold (`DRIFT_SIMILARITY_THRESHOLD`, 0.85 by default; the example readings above were taken with 0.6), a drift event is recorded. Nothing is re-indexed automatically: `POST /reindex-baseline` resets the baseline to the current window, and the vector index itself is not rebuilt.

Until the window holds `DRIFT_WINDOW_SIZE` embeddings (set `DRIFT_MIN_SAMPLES` to judge earlier) no drift verdict is made, so the first few queries after a start or reset cannot raise a false alarm. Only the most recent 1000 events are kept; the dashboard counter keeps the running total.

**Simulate drift** feeds the system off-distribution text so you can watch similarity fall and an event fire. **Reset drift** restores the baseline.

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
docker-compose.yml    # Qdrant + Redis, both local, no keys
.env.example          # non-secret config only (host/port)
```

## Config

All config lives in `.env` (copy from `.env.example`). Nothing in it is a secret, only local hostnames and ports for Qdrant and Redis.

## Limitations

Documenting these on purpose:

- **Small scale.** The demo indexes 20 documents and the latency figures come from a few dozen requests. Percentiles from so few samples are noisy, and the first request (cold start) can dominate p99.
- **Hit@5 is easy on a tiny corpus.** With 20 documents, the top 5 covers a quarter of the corpus, and the test set has 18 queries. Read 100% as "the pipeline works end to end", not as evidence of retrieval quality at scale.
- **Lexical, not semantic, embeddings.** See the trade-off above.
- **Drift is simulated in the demo.** The drift events you see come from the **Simulate drift** control, which fills the whole window with off-topic text, so it records a single event when the window fills. They are not from real production traffic shifting.
- **Retrieval only.** There is no LLM generation step. This is the retrieval and observability layer of a RAG system.

## Roadmap

- [ ] Swap in a semantic embedding model behind the same embedder interface
- [ ] Larger labeled evaluation set, with MRR alongside Hit@5
- [x] Skip drift alerts until the rolling window is full
- [ ] Persist metrics beyond process restarts
- [ ] Expand test coverage

## License

MIT
