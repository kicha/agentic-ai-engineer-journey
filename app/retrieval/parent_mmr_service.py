from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

from app.embeddings.base import DenseEmbeddingService, DenseVector

if TYPE_CHECKING:
    from app.reranking.cross_encoder_reranker import RerankedParentResult
from app.retrieval.parent_mmr_selector import (
    ParentMMRCandidate,
    ParentMMRResult,
    ParentMMRSelector,
)


class ParentMMRService:
    """Build and select parent-level MMR candidates with embedding reuse."""

    def __init__(
        self,
        embedder: DenseEmbeddingService,
        lambda_mult: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.selector = ParentMMRSelector(lambda_mult=lambda_mult)
        self._parent_embedding_cache: dict[str, DenseVector] = {}

    @staticmethod
    def _cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0.0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)

    @staticmethod
    def _cache_key(item: RerankedParentResult) -> str:
        parent = item.result.parent
        content_hash = getattr(parent, "content_hash", "")
        return f"{parent.parent_id}:{content_hash}"

    def _get_parent_embeddings(
        self,
        reranked_results: Sequence["RerankedParentResult"],
    ) -> list[DenseVector]:
        missing_items: list["RerankedParentResult"] = []
        missing_keys: list[str] = []

        for item in reranked_results:
            key = self._cache_key(item)
            if key not in self._parent_embedding_cache:
                missing_items.append(item)
                missing_keys.append(key)

        if missing_items:
            generated = self.embedder.embed_documents(
                [item.result.parent.content for item in missing_items]
            )
            if len(generated) != len(missing_items):
                raise RuntimeError(
                    "Dense embedding provider returned an unexpected number "
                    f"of parent embeddings. expected={len(missing_items)} "
                    f"actual={len(generated)}"
                )
            for key, embedding in zip(missing_keys, generated):
                self._parent_embedding_cache[key] = embedding

        return [
            self._parent_embedding_cache[self._cache_key(item)]
            for item in reranked_results
        ]

    def select(
        self,
        *,
        query: str,
        reranked_results: Sequence["RerankedParentResult"],
        top_k: int,
    ) -> list[ParentMMRResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not reranked_results:
            return []

        query_vector = np.asarray(
            self.embedder.embed_query(query),
            dtype=np.float32,
        )
        parent_embeddings = self._get_parent_embeddings(reranked_results)

        candidates: list[ParentMMRCandidate] = []
        for reranked_item, parent_embedding in zip(
            reranked_results,
            parent_embeddings,
        ):
            parent_vector = np.asarray(parent_embedding, dtype=np.float32)
            candidates.append(
                ParentMMRCandidate(
                    parent=reranked_item.result.parent,
                    query_similarity=self._cosine_similarity(
                        query_vector,
                        parent_vector,
                    ),
                    reranker_score=reranked_item.reranker_score,
                    reranker_rank=reranked_item.reranked_rank,
                    embedding=parent_vector.tolist(),
                )
            )

        return self.selector.select(candidates=candidates, top_k=top_k)
