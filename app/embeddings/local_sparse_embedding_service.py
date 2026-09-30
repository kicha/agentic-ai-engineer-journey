from __future__ import annotations

import logging
from typing import Sequence

from fastembed import SparseTextEmbedding

from app.embeddings.base import (
    SparseEmbeddingService,
    SparseVector,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOCAL SPARSE EMBEDDING SERVICE
# ============================================================


class LocalSparseEmbeddingService(SparseEmbeddingService):
    """
    Local sparse lexical embedding provider backed by FastEmbed.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name

        logger.info(
            "Loading local sparse embedding model: %s",
            self.model_name,
        )

        self.model = SparseTextEmbedding(
            model_name=self.model_name,
        )

    @staticmethod
    def _to_sparse_vector(
        embedding,
    ) -> SparseVector:

        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseVector]:

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        sparse_embeddings = list(self.model.embed(cleaned_texts))

        return [self._to_sparse_vector(embedding) for embedding in sparse_embeddings]

    def embed_query(
        self,
        query: str,
    ) -> SparseVector:

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        embeddings = list(self.model.embed([query]))

        if not embeddings:
            raise RuntimeError("Sparse embedding model returned no vector.")

        return self._to_sparse_vector(embeddings[0])
