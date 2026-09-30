from typing import List

from qdrant_client.models import ScoredPoint

from app.embeddings.base import (
    DenseEmbeddingService,
    SparseEmbeddingService,
)

from app.vectorstore.hybrid_qdrant_store import (
    HybridQdrantStore,
)


class HybridChildRetriever:

    def __init__(
        self,
        dense_embedding_service: DenseEmbeddingService,
        vector_store: HybridQdrantStore,
        sparse_embedding_service: SparseEmbeddingService | None = None,
    ):
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store

    # =========================================================
    # DENSE
    # =========================================================

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

    # =========================================================
    # SPARSE
    # =========================================================

    def retrieve_sparse(self, query: str, limit: int = 10) -> List[ScoredPoint]:

        self._validate_query(query)
        if self.vector_store.uses_cloud_sparse_inference:
            return self.vector_store.sparse_search_text(
                query_text=query,
                limit=limit,
            )
        if self.sparse_embedding_service is None:
            raise RuntimeError(
                "Local sparse retrieval requires " "SparseEmbeddingService."
            )

        sparse_vector = self.sparse_embedding_service.embed_query(query)

        return self.vector_store.sparse_search(
            query_vector=sparse_vector,
            limit=limit,
        )

    # =========================================================
    # HYBRID
    # =========================================================

    def retrieve_hybrid(self, query: str, limit: int = 10) -> List[ScoredPoint]:

        self._validate_query(query)

        dense_vector = self.dense_embedding_service.embed_query(query)

        if self.vector_store.uses_cloud_sparse_inference:
            return self.vector_store.hybrid_search_text(
                dense_query_vector=(dense_vector),
                query_text=query,
                limit=limit,
            )

        if self.sparse_embedding_service is None:
            raise RuntimeError(
                "Local hybrid retrieval requires " "SparseEmbeddingService."
            )

        sparse_vector = self.sparse_embedding_service.embed_query(query)

        return self.vector_store.hybrid_search(
            dense_query_vector=dense_vector,
            sparse_query_vector=sparse_vector,
            limit=limit,
        )

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_query(query: str) -> None:

        if not query or not query.strip():
            raise ValueError("Query must not be empty.")
