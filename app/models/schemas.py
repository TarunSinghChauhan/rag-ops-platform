from pydantic import BaseModel


class IndexRequest(BaseModel):
    doc_id: int
    text: str
    metadata: dict = {}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    doc_id: int
    score: float
    payload: dict


class QueryResponse(BaseModel):
    results: list[QueryResult]
    cache_hit: bool
    drift: dict
