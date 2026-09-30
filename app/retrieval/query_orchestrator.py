from __future__ import annotations

from pydantic import BaseModel, Field

from app.config.settings import QueryTransformationMode, QueryingConfig
from app.querying.multi_query_generator import MultiQueryGenerator
from app.querying.query_rewriter import QueryRewriter
from app.retrieval.hybrid_rerank_mmr_retriever import HybridRerankMMRRetriever
from app.retrieval.hyde.hyde_hybrid_retriever import HyDEHybridRetriever
from app.retrieval.multi_query_hybrid_retriever import MultiQueryHybridRetriever
from app.retrieval.parent_mmr_selector import ParentMMRResult


class QueryTrace(BaseModel):
    mode: QueryTransformationMode
    original_query: str
    retrieval_queries: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None
    hypothetical_document: str | None = None
    used_fallback: bool = False
    error: str | None = None


class AdvancedRetrievalResult(BaseModel):
    trace: QueryTrace
    results: list[ParentMMRResult]


class QueryOrchestrator:
    """Single entry point for original, rewrite, multi-query and HyDE retrieval."""

    def __init__(
        self,
        *,
        config: QueryingConfig,
        query_rewriter: QueryRewriter,
        multi_query_generator: MultiQueryGenerator,
        base_retriever: HybridRerankMMRRetriever,
        multi_query_retriever: MultiQueryHybridRetriever,
        hyde_retriever: HyDEHybridRetriever,
    ) -> None:
        self.config = config
        self.query_rewriter = query_rewriter
        self.multi_query_generator = multi_query_generator
        self.base_retriever = base_retriever
        self.multi_query_retriever = multi_query_retriever
        self.hyde_retriever = hyde_retriever

    def retrieve(
        self,
        query: str,
        *,
        mode: QueryTransformationMode | None = None,
        context: str | None = None,
    ) -> AdvancedRetrievalResult:
        original_query = query.strip()
        if not original_query:
            raise ValueError("Query cannot be empty.")

        selected_mode = mode or self.config.mode

        if selected_mode == QueryTransformationMode.MULTI_QUERY:
            result = self.multi_query_retriever.retrieve(
                query=original_query,
                child_top_k_per_query=self.config.child_top_k_per_query,
                fused_child_top_k=self.config.fused_child_top_k,
                reranker_top_k=self.config.reranker_candidate_k,
                final_top_k=self.config.final_parent_k,
            )
            return AdvancedRetrievalResult(
                trace=QueryTrace(
                    mode=selected_mode,
                    original_query=original_query,
                    retrieval_queries=result.generation.all_queries,
                    used_fallback=result.generation.used_fallback,
                    error=result.generation.error,
                ),
                results=result.final_results,
            )

        if selected_mode == QueryTransformationMode.REWRITE:
            rewrite = self.query_rewriter.rewrite(
                query=original_query,
                context=context,
            )
            results = self.base_retriever.retrieve(
                query=rewrite.rewritten_query,
                child_top_k=self.config.fused_child_top_k,
                reranker_top_k=self.config.reranker_candidate_k,
                final_top_k=self.config.final_parent_k,
            )
            return AdvancedRetrievalResult(
                trace=QueryTrace(
                    mode=selected_mode,
                    original_query=original_query,
                    retrieval_queries=[rewrite.rewritten_query],
                    rewritten_query=rewrite.rewritten_query,
                    used_fallback=rewrite.used_fallback,
                    error=rewrite.error,
                ),
                results=results,
            )

        if selected_mode == QueryTransformationMode.HYDE:
            result = self.hyde_retriever.retrieve(
                query=original_query,
                child_top_k=self.config.fused_child_top_k,
                reranker_top_k=self.config.reranker_candidate_k,
                final_top_k=self.config.final_parent_k,
            )
            return AdvancedRetrievalResult(
                trace=QueryTrace(
                    mode=selected_mode,
                    original_query=original_query,
                    retrieval_queries=[original_query],
                    hypothetical_document=result.generation.hypothetical_document,
                ),
                results=result.final_results,
            )

        if selected_mode == QueryTransformationMode.ORIGINAL:
            results = self.base_retriever.retrieve(
                query=original_query,
                child_top_k=self.config.fused_child_top_k,
                reranker_top_k=self.config.reranker_candidate_k,
                final_top_k=self.config.final_parent_k,
            )
            return AdvancedRetrievalResult(
                trace=QueryTrace(
                    mode=selected_mode,
                    original_query=original_query,
                    retrieval_queries=[original_query],
                ),
                results=results,
            )

        raise ValueError(f"Unsupported query transformation mode: {selected_mode}")
