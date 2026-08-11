from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Real-Time RAG Ops Platform",
    description="RAG pipeline with zero-API-cost embeddings, Qdrant search, "
                 "Redis caching, and cosine-similarity drift detection.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG Ops Platform is running", "docs": "/docs"}
