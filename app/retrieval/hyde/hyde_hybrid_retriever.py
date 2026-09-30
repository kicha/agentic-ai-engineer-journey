from __future__ import annotations

from pydantic import BaseModel

from app.reranking.base import Reranker
from app.retrieval.hybrid_child_retriever import HybridChildRetriever
from app.retrieval.hybrid_parent_child_retriever import HybridParentChildRetriever
from app.retrieval.hyde.hyde_generator import HyDEGenerator, HyDEResult
from app.retrieval.parent_mmr_selector import ParentMMRResult
from app.retrieval.parent_mmr_service import ParentMMRService


class HyDEHybridRetrievalResult(BaseModel):
    generation: HyDEResult
    parent_candidates_before_rerank: int
    final_results: list[ParentMMRResult]


class HyDEHybridRetriever:
    """HyDE dense view + original-query sparse view -> RRF -> parents -> rerank -> MMR."""

    def __init__(
        self,
        hyde_generator: HyDEGenerator,
        hybrid_child_retriever: HybridChildRetriever,
        parent_child_retriever: HybridParentChildRetriever,
        reranker: Reranker,
        parent_mmr_service: ParentMMRService,
    ) -> None:
        self.hyde_generator = hyde_generator
        self.hybrid_child_retriever = hybrid_child_retriever
        self.parent_child_retriever = parent_child_retriever
        self.reranker = reranker
        self.parent_mmr_service = parent_mmr_service

    def retrieve(
        self,
        query: str,
        child_top_k: int = 15,
        reranker_top_k: int = 5,
        final_top_k: int = 3,
    ) -> HyDEHybridRetrievalResult:
        original_query = query.strip()
        if not original_query:
            raise ValueError("Query cannot be empty.")
        if child_top_k <= 0 or reranker_top_k <= 0 or final_top_k <= 0:
            raise ValueError("Retrieval top-k values must be greater than zero.")
        if final_top_k > reranker_top_k:
            raise ValueError("final_top_k cannot exceed reranker_top_k.")

        generation = self.hyde_generator.generate(original_query)
        child_hits = self.hybrid_child_retriever.retrieve_hybrid_with_dense_text(
            sparse_query=original_query,
            dense_text=generation.hypothetical_document,
            limit=child_top_k,
        )
        parent_candidates = self.parent_child_retriever.expand_hits(
            hits=child_hits,
            parent_top_k=None,
        )
        if not parent_candidates:
            return HyDEHybridRetrievalResult(
                generation=generation,
                parent_candidates_before_rerank=0,
                final_results=[],
            )

        reranked = self.reranker.rerank(
            query=original_query,
            candidates=parent_candidates,
            top_k=reranker_top_k,
        )
        final_results = self.parent_mmr_service.select(
            query=original_query,
            reranked_results=reranked.results,
            top_k=final_top_k,
        )
        return HyDEHybridRetrievalResult(
            generation=generation,
            parent_candidates_before_rerank=len(parent_candidates),
            final_results=final_results,
        )
