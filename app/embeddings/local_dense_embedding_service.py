from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.embeddings.base import (
    DenseEmbeddingService,
    DenseVector,
)

logger = logging.getLogger(__name__)


# ============================================================
# LOCAL DENSE EMBEDDING SERVICE
# ============================================================


class LocalDenseEmbeddingService(DenseEmbeddingService):
    """
    Local dense embedding provider backed by SentenceTransformers.

    This preserves our current behaviour while allowing the application
    to depend on DenseEmbeddingService rather than SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name

        logger.info(f"Loading local dense embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    @property
    def embedding_dimension(self) -> int:

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Unable to determine embedding dimension "
                f"for model '{self.model_name}'."
            )

        return int(dimension)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        embeddings = self.model.encode(
            cleaned_texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> DenseVector:

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()
