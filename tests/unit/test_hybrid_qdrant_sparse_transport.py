from unittest.mock import MagicMock

from qdrant_client import models

from app.config.settings import HybridSearchConfig, QdrantConfig, QdrantDeployment
from app.embeddings.base import SparseText, SparseVector
from app.vectorstore.hybrid_qdrant_store import HybridQdrantStore


def _store() -> HybridQdrantStore:
    store = object.__new__(HybridQdrantStore)
    store.config = HybridSearchConfig()
    store.collection_name = "test"
    store.vector_size = 512
    store.qdrant_config = QdrantConfig(deployment=QdrantDeployment.LOCAL)
    store.client = MagicMock()
    return store


def test_local_sparse_vector_converts_at_qdrant_boundary() -> None:
    converted = HybridQdrantStore._to_qdrant_sparse(
        SparseVector(indices=[1, 7], values=[0.2, 0.8])
    )
    assert isinstance(converted, models.SparseVector)
    assert converted.indices == [1, 7]
    assert converted.values == [0.2, 0.8]


def test_cloud_sparse_text_converts_to_qdrant_document() -> None:
    converted = HybridQdrantStore._to_qdrant_sparse(
        SparseText(text="hybrid search", model="qdrant/bm25")
    )
    assert isinstance(converted, models.Document)
    assert converted.text == "hybrid search"
    assert converted.model == "qdrant/bm25"


def test_sparse_search_uses_same_public_method_for_cloud_inference() -> None:
    store = _store()
    store.client.query_points.return_value.points = []
    store.sparse_search(
        query_embedding=SparseText(text="metadata filters", model="qdrant/bm25"),
        limit=4,
    )
    call = store.client.query_points.call_args.kwargs
    assert isinstance(call["query"], models.Document)
    assert call["using"] == store.config.sparse_vector_name
    assert call["limit"] == 4
