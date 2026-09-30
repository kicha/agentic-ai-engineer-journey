from __future__ import annotations

import logging
from time import perf_counter

from app.repositories.parent_repository import (
    ParentRepository,
)
from app.reranking.base import Reranker
from app.retrieval.parent_child_retriever import (
    ParentRetrievalResult,
)
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)

logger = logging.getLogger(__name__)
RERANKER_CANDIDATE_K = 5


class ParentRerankingRunner:
    """
    Rerank an existing parent-level benchmark result.

    Important
    ---------
    This runner:

    - does NOT rerun child retrieval,
    - does NOT repeat parent expansion,
    - receives the exact parent candidate set produced by
      BaselineParentExpansionRunner,
    - reloads authoritative ParentChunks from MongoDB,
    - applies the configured parent-level reranker,
    - emits a standard BenchmarkRunResult.
    """

    def __init__(
        self,
        *,
        parent_repository: ParentRepository,
        reranker: Reranker,
        final_k: int = RERANKER_CANDIDATE_K,
    ) -> None:

        if final_k <= 0:
            raise ValueError("final_k must be greater than zero.")

        self.parent_repository = parent_repository
        self.reranker = reranker
        self.final_k = final_k

    # ========================================================
    # PUBLIC RUNNER
    # ========================================================

    def run(
        self,
        *,
        parent_run_result: BenchmarkRunResult,
    ) -> BenchmarkRunResult:

        if parent_run_result.retrieval_level != RetrievalLevel.PARENT:
            raise ValueError(
                "ParentRerankingRunner requires " "a parent-level BenchmarkRunResult."
            )

        reranked_results: list[QueryRankingResult] = []

        for query_result in parent_run_result.results:

            reranked_results.append(
                self._rerank_query(
                    query_result=query_result,
                )
            )

        return BenchmarkRunResult(
            benchmark_version=(parent_run_result.benchmark_version),
            strategy=RetrievalStrategy.HYBRID_RERANK,
            retrieval_level=RetrievalLevel.PARENT,
            results=reranked_results,
        )

    # ========================================================
    # ONE QUERY
    # ========================================================

    def _rerank_query(
        self,
        *,
        query_result: QueryRankingResult,
    ) -> QueryRankingResult:

        started = perf_counter()

        candidates: list[ParentRetrievalResult] = []

        original_items: list[RankedItem] = []

        # ----------------------------------------------------
        # REHYDRATE AUTHORITATIVE PARENTS
        # ----------------------------------------------------
        #
        # Benchmark ranking stores parent_key as the stable
        # evaluation identity.
        #
        # The actual ParentChunk content still comes from
        # MongoDB, which remains authoritative.
        # ----------------------------------------------------

        for ranked_item in query_result.items:

            parent_id = ranked_item.metadata.get("parent_id")

            if not parent_id:
                raise RuntimeError(
                    "Parent benchmark item is missing " "parent_id metadata."
                )

            parent_id_string = str(parent_id)

            parent = self.parent_repository.get_by_id(parent_id_string)

            if parent is None:
                raise RuntimeError(
                    "Dangling parent reference "
                    "detected during reranking: "
                    f"parent_id={parent_id_string}"
                )

            vector_score = ranked_item.score if ranked_item.score is not None else 0.0

            candidates.append(
                ParentRetrievalResult(
                    parent=parent,
                    best_score=vector_score,
                    matched_children=[],
                )
            )

            original_items.append(ranked_item)

        # ----------------------------------------------------
        # PARENT-LEVEL RERANKING
        # ----------------------------------------------------

        reranking_response = self.reranker.rerank(
            query=query_result.query,
            candidates=candidates,
            top_k=self.final_k,
        )

        reranked_items: list[RankedItem] = []

        for reranked_result in reranking_response.results:

            original_index = reranked_result.original_rank - 1

            if original_index < 0 or original_index >= len(original_items):
                raise RuntimeError("Reranker returned invalid " "original rank.")

            original_item = original_items[original_index]

            reranked_items.append(
                RankedItem(
                    item_id=(original_item.item_id),
                    rank=(reranked_result.reranked_rank),
                    score=(reranked_result.reranker_score),
                    metadata={
                        **original_item.metadata,
                        "pre_rerank_rank": (reranked_result.original_rank),
                        "pre_rerank_score": (reranked_result.vector_score),
                        "reranker_score": (reranked_result.reranker_score),
                    },
                )
            )

        reranking_latency_ms = (perf_counter() - started) * 1000.0

        # ----------------------------------------------------
        # CUMULATIVE PIPELINE LATENCY
        # ----------------------------------------------------
        #
        # query_result.latency_ms already contains:
        #
        # child retrieval
        # + parent expansion
        #
        # Therefore the reranked result should add reranking
        # latency rather than replace the existing latency.
        # ----------------------------------------------------

        previous_latency_ms = query_result.latency_ms or 0.0

        total_latency_ms = previous_latency_ms + reranking_latency_ms

        logger.info(
            f"query_id={query_result.query_id} "
            f"candidate_parents="
            f"{len(candidates)} "
            f"reranked_parents="
            f"{len(reranked_items)} "
            f"reranking_ms="
            f"{reranking_latency_ms:.2f} "
            f"total_pipeline_ms="
            f"{total_latency_ms:.2f}"
        )

        return QueryRankingResult(
            query_id=query_result.query_id,
            query=query_result.query,
            retrieval_level=(RetrievalLevel.PARENT),
            items=reranked_items,
            latency_ms=total_latency_ms,
        )
