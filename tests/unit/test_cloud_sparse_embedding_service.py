import pytest

from app.embeddings.base import SparseText
from app.embeddings.cloud_sparse_embedding_service import CloudSparseEmbeddingService


def test_cloud_sparse_query_returns_server_inference_request() -> None:
    service = CloudSparseEmbeddingService(model_name="qdrant/bm25")
    result = service.embed_query("  hybrid search  ")
    assert result == SparseText(text="hybrid search", model="qdrant/bm25")


def test_cloud_sparse_documents_preserve_one_result_per_non_empty_text() -> None:
    service = CloudSparseEmbeddingService(model_name="qdrant/bm25")
    results = service.embed_documents([" first ", "", " second "])
    assert results == [
        SparseText(text="first", model="qdrant/bm25"),
        SparseText(text="second", model="qdrant/bm25"),
    ]


def test_cloud_sparse_rejects_empty_query() -> None:
    service = CloudSparseEmbeddingService(model_name="qdrant/bm25")
    with pytest.raises(ValueError, match="Query must not be empty"):
        service.embed_query("   ")
