from __future__ import annotations

from app.embeddings.base import DenseEmbeddingService
from app.reranking.base import Reranker
from app.retrieval.hybrid_parent_child_retriever import HybridParentChildRetriever
from app.retrieval.parent_mmr_selector import ParentMMRResult
from app.retrieval.parent_mmr_service import ParentMMRService


class HybridRerankMMRRetriever:
    """Hybrid -> parent expansion/dedup -> reranker -> dense Parent MMR."""

    def __init__(
        self,
        hybrid_parent_child_retriever: HybridParentChildRetriever,
        reranker: Reranker,
        embedder: DenseEmbeddingService | None = None,
        lambda_mult: float = 0.7,
        parent_mmr_service: ParentMMRService | None = None,
    ) -> None:
        self.hybrid_parent_child_retriever = hybrid_parent_child_retriever
        self.reranker = reranker
        if parent_mmr_service is None:
            if embedder is None:
                raise ValueError(
                    "embedder is required when parent_mmr_service is not supplied."
                )
            parent_mmr_service = ParentMMRService(
                embedder=embedder,
                lambda_mult=lambda_mult,
            )
        self.parent_mmr_service = parent_mmr_service

    def retrieve(
        self,
        query: str,
        child_top_k: int = 15,
        reranker_top_k: int = 5,
        final_top_k: int = 3,
    ) -> list[ParentMMRResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query cannot be empty.")
        if child_top_k <= 0 or reranker_top_k <= 0 or final_top_k <= 0:
            raise ValueError("Retrieval top-k values must be greater than zero.")
        if final_top_k > reranker_top_k:
            raise ValueError("final_top_k cannot exceed reranker_top_k.")

        parent_candidates = self.hybrid_parent_child_retriever.retrieve(
            query=normalized_query,
            child_top_k=child_top_k,
            parent_top_k=None,
        )
        if not parent_candidates:
            return []

        reranked = self.reranker.rerank(
            query=normalized_query,
            candidates=parent_candidates,
            top_k=reranker_top_k,
        )
        return self.parent_mmr_service.select(
            query=normalized_query,
            reranked_results=reranked.results,
            top_k=final_top_k,
        )
