from app.models.schemas import IndexRequest, QueryRequest, QueryResult, QueryResponse


def test_index_request_defaults_empty_metadata():
    req = IndexRequest(doc_id=1, text="hello world")
    assert req.metadata == {}


def test_index_request_accepts_metadata():
    req = IndexRequest(doc_id=1, text="hello", metadata={"source": "wiki"})
    assert req.metadata["source"] == "wiki"


def test_query_request_defaults_top_k_to_5():
    req = QueryRequest(query="what is rag")
    assert req.top_k == 5


def test_query_request_accepts_custom_top_k():
    req = QueryRequest(query="what is rag", top_k=10)
    assert req.top_k == 10


def test_query_result_holds_expected_fields():
    result = QueryResult(doc_id=1, score=0.87, payload={"text": "chunk"})
    assert result.doc_id == 1
    assert result.score == 0.87
    assert result.payload == {"text": "chunk"}


def test_query_response_wraps_results_list():
    result = QueryResult(doc_id=1, score=0.9, payload={})
    resp = QueryResponse(results=[result], cache_hit=True, drift={"score": 0.1})
    assert len(resp.results) == 1
    assert resp.cache_hit is True
    assert resp.drift == {"score": 0.1}
