from __future__ import annotations

from typing import Sequence

from app.embeddings.base import (
    InferenceDocument,
    SparseEmbedding,
    SparseEmbeddingService,
)


class CloudSparseEmbeddingService(SparseEmbeddingService):
    """
    Sparse embedding provider using Qdrant Cloud Inference.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:

        self.model_name = model_name

    def embed_query(
        self,
        query: str,
    ) -> SparseEmbedding:

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        return InferenceDocument(
            text=query,
            model=self.model_name,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[SparseEmbedding]:

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        return [
            InferenceDocument(
                text=text,
                model=self.model_name,
            )
            for text in cleaned_texts
        ]
