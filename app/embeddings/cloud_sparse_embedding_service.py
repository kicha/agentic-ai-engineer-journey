from __future__ import annotations

from typing import Sequence

from app.embeddings.base import (
    SparseEmbedding,
    SparseEmbeddingService,
    SparseText,
)


class CloudSparseEmbeddingService(SparseEmbeddingService):
    """Describe sparse embeddings that Qdrant Cloud should compute server-side."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_query(self, query: str) -> SparseEmbedding:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        return SparseText(
            text=normalized_query,
            model=self.model_name,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[SparseEmbedding]:
        cleaned_texts = [text.strip() for text in texts if text and text.strip()]
        return [
            SparseText(text=text, model=self.model_name)
            for text in cleaned_texts
        ]
