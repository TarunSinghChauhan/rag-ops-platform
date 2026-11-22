from pydantic import BaseModel, ConfigDict, Field


class IndexRequest(BaseModel):
    # Qdrant point ids are unsigned integers: a negative id was accepted here and then
    # failed inside the upsert as a 500. An empty text embeds to a zero vector, which
    # cosine search cannot rank; the length cap keeps one request from hashing megabytes.
    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: int = Field(ge=0, le=2**63 - 1)
    text: str = Field(min_length=1, max_length=20000)
    metadata: dict = {}


class QueryRequest(BaseModel):
    # Whitespace is stripped first, so a blank query is rejected (it would embed to a
    # zero vector and drag the drift centroid) instead of being searched.
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResult(BaseModel):
    doc_id: int
    score: float
    payload: dict


class QueryResponse(BaseModel):
    results: list[QueryResult]
    cache_hit: bool
    drift: dict
