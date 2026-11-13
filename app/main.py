from dotenv import load_dotenv
load_dotenv()

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.routes import router
from app.api.dashboard import dash_router, seed_if_empty
from app.core.metrics import latency


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_if_empty()
    yield


app = FastAPI(
    title="Real-Time RAG Ops Platform",
    description="RAG pipeline with zero-API-cost embeddings, Qdrant search, "
                 "Redis caching, and cosine-similarity drift detection.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_latency(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    # Rejected requests (422 validation errors and the like) return in microseconds and
    # would drag the percentiles down; only answered queries are a latency sample.
    if request.url.path == "/query" and response.status_code < 400:
        latency.record((time.perf_counter() - start) * 1000)
    return response


app.include_router(router)
app.include_router(dash_router)


@app.get("/")
def root():
    return {"message": "RAG Ops Platform is running", "docs": "/docs", "dashboard": "/dashboard"}
