from __future__ import annotations

from typing import List

from qdrant_client.models import ScoredPoint

from app.embeddings.base import DenseEmbeddingService, SparseEmbeddingService
from app.vectorstore.hybrid_qdrant_store import HybridQdrantStore


class HybridChildRetriever:
    """Provider-neutral dense, sparse and hybrid child retrieval."""

    def __init__(
        self,
        dense_embedding_service: DenseEmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: HybridQdrantStore,
    ) -> None:
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store

    def retrieve_dense(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        self._validate_query(query)
        dense_vector = self.dense_embedding_service.embed_query(query)
        return self.vector_store.dense_search(
            query_vector=dense_vector,
            limit=limit,
        )

    def retrieve_sparse(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        self._validate_query(query)
        sparse_embedding = self.sparse_embedding_service.embed_query(query)
        return self.vector_store.sparse_search(
            query_embedding=sparse_embedding,
            limit=limit,
        )

    def retrieve_hybrid(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        self._validate_query(query)
        dense_vector = self.dense_embedding_service.embed_query(query)
        sparse_embedding = self.sparse_embedding_service.embed_query(query)
        return self.vector_store.hybrid_search(
            dense_query_vector=dense_vector,
            sparse_query_embedding=sparse_embedding,
            limit=limit,
        )

    def retrieve_hybrid_with_dense_text(
        self,
        *,
        sparse_query: str,
        dense_text: str,
        limit: int = 10,
    ) -> List[ScoredPoint]:
        """Hybrid retrieval with different dense and sparse textual views.

        HyDE uses this path: the hypothetical document drives semantic dense
        retrieval while the original user query preserves exact lexical signal.
        """

        self._validate_query(sparse_query)
        self._validate_query(dense_text)
        dense_vector = self.dense_embedding_service.embed_query(dense_text)
        sparse_embedding = self.sparse_embedding_service.embed_query(sparse_query)
        return self.vector_store.hybrid_search(
            dense_query_vector=dense_vector,
            sparse_query_embedding=sparse_embedding,
            limit=limit,
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
