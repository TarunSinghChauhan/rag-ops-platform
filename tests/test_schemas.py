import pytest
from pydantic import ValidationError

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


@pytest.mark.parametrize("top_k", [0, -3, 51, 1_000_000])
def test_query_request_rejects_out_of_range_top_k(top_k):
    with pytest.raises(ValidationError):
        QueryRequest(query="what is rag", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 50])
def test_query_request_accepts_the_top_k_bounds(top_k):
    assert QueryRequest(query="what is rag", top_k=top_k).top_k == top_k


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_query_request_rejects_blank_query(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


def test_query_request_strips_surrounding_whitespace():
    assert QueryRequest(query="  what is rag \n").query == "what is rag"


def test_query_request_rejects_oversized_query():
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)
