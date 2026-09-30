from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from app.embeddings.base import DenseEmbeddingService

# from app.repositories.parent_repository import ParentRepository
from typing import Protocol

from app.models.parent_chunk import ParentChunk
from app.retrieval.parent_mmr_selector import (
    ParentMMRCandidate,
    ParentMMRSelector,
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

FINAL_PARENT_K = 3


class ParentRepositoryProtocol(Protocol):
    """
    Minimal repository contract required by ParentMMRRunner.

    The runner does not need the complete ParentRepository API.
    It only requires lookup by parent_id.
    """

    def get_by_id(
        self,
        parent_id: str,
    ) -> ParentChunk | None: ...


class ParentMMRRunner:
    """
    Apply Parent-level MMR to an existing reranked parent result.

    Canonical benchmark flow:

        child retrieval
            ↓
        parent expansion / deduplication
            ↓
        parent reranking
            ↓
        ParentMMRRunner
            ↓
        final parent context

    Important:
        - Does NOT rerun retrieval.
        - Does NOT rerun parent expansion.
        - Does NOT rerun CrossEncoder / hosted reranking.
        - Reuses the existing production ParentMMRSelector.
        - Uses dense query-to-parent similarity as MMR relevance.
        - Uses parent-to-parent similarity as redundancy.
        - Reranker score is retained only for traceability.
    """

    def __init__(
        self,
        *,
        parent_repository: ParentRepositoryProtocol,
        embedder: DenseEmbeddingService,
        lambda_mult: float = 0.7,
        final_k: int = FINAL_PARENT_K,
    ) -> None:

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1.")

        if final_k <= 0:
            raise ValueError("final_k must be greater than zero.")

        self.parent_repository = parent_repository
        self.embedder = embedder
        self.final_k = final_k

        self.mmr_selector = ParentMMRSelector(
            lambda_mult=lambda_mult,
        )

    # ========================================================
    # PUBLIC RUNNER
    # ========================================================

    def run(
        self,
        *,
        reranked_run_result: BenchmarkRunResult,
    ) -> BenchmarkRunResult:
        """
        Apply Parent MMR to every query in an existing
        reranked parent benchmark run.
        """

        if reranked_run_result.retrieval_level != RetrievalLevel.PARENT:
            raise ValueError(
                "ParentMMRRunner requires " "a parent-level BenchmarkRunResult."
            )

        mmr_results: list[QueryRankingResult] = []

        for query_result in reranked_run_result.results:
            mmr_results.append(
                self._select_query(
                    query_result=query_result,
                )
            )

        return BenchmarkRunResult(
            benchmark_version=(reranked_run_result.benchmark_version),
            strategy=(RetrievalStrategy.HYBRID_RERANK_MMR),
            retrieval_level=(RetrievalLevel.PARENT),
            results=mmr_results,
        )

    # ========================================================
    # ONE QUERY
    # ========================================================

    def _select_query(
        self,
        *,
        query_result: QueryRankingResult,
    ) -> QueryRankingResult:
        """
        Convert an existing reranked parent result into MMR
        candidates and select the final diverse context.
        """

        started = perf_counter()

        # ----------------------------------------------------
        # FINAL K MUST NOT EXCEED AVAILABLE CANDIDATES
        # ----------------------------------------------------

        if self.final_k > len(query_result.items):
            raise ValueError(
                "final_k cannot exceed the number " "of available parent candidates."
            )

        if not query_result.items:
            return QueryRankingResult(
                query_id=query_result.query_id,
                query=query_result.query,
                retrieval_level=(RetrievalLevel.PARENT),
                items=[],
                latency_ms=(query_result.latency_ms or 0.0),
            )

        # ----------------------------------------------------
        # 1. REHYDRATE AUTHORITATIVE PARENTS
        # ----------------------------------------------------

        parents = []

        for ranked_item in query_result.items:

            parent_id = ranked_item.metadata.get("parent_id")

            if not parent_id:
                raise RuntimeError(
                    "Parent benchmark item is missing " "parent_id metadata."
                )

            parent = self.parent_repository.get_by_id(str(parent_id))

            if parent is None:
                raise RuntimeError(
                    "Dangling parent reference "
                    "detected during Parent MMR: "
                    f"parent_id={parent_id}"
                )

            parents.append(parent)

        # ----------------------------------------------------
        # 2. EMBED QUERY
        # ----------------------------------------------------

        query_embedding = np.asarray(
            self.embedder.embed_query(query_result.query),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # 3. EMBED ALL PARENT CANDIDATES IN ONE BATCH
        # ----------------------------------------------------

        parent_texts = [parent.content for parent in parents]

        parent_embeddings = self.embedder.embed_documents(parent_texts)

        if len(parent_embeddings) != len(parents):
            raise RuntimeError(
                "Embedding service returned an "
                "unexpected number of parent embeddings."
            )

        # ----------------------------------------------------
        # 4. BUILD PRODUCTION MMR CANDIDATES
        #
        # IMPORTANT:
        #
        # MMR relevance is NOT the reranker score.
        #
        # Relevance:
        #     dense(query, parent)
        #
        # Redundancy:
        #     dense(parent, selected_parent)
        #
        # Reranker score/rank are preserved only for
        # diagnostics and traceability.
        # ----------------------------------------------------

        mmr_candidates: list[ParentMMRCandidate] = []

        for (
            ranked_item,
            parent,
            parent_embedding,
        ) in zip(
            query_result.items,
            parents,
            parent_embeddings,
        ):

            parent_vector = np.asarray(
                parent_embedding,
                dtype=np.float32,
            )

            query_similarity = self._cosine_similarity(
                query_embedding,
                parent_vector,
            )

            reranker_score = ranked_item.score if ranked_item.score is not None else 0.0

            mmr_candidates.append(
                ParentMMRCandidate(
                    parent=parent,
                    query_similarity=(query_similarity),
                    reranker_score=(reranker_score),
                    reranker_rank=(ranked_item.rank),
                    embedding=(parent_vector.tolist()),
                )
            )

        # ----------------------------------------------------
        # 5. PRODUCTION PARENT MMR
        # ----------------------------------------------------

        mmr_results = self.mmr_selector.select(
            candidates=mmr_candidates,
            top_k=self.final_k,
        )

        # ----------------------------------------------------
        # 6. MAP MMR OUTPUT BACK TO STABLE BENCHMARK IDs
        #
        # Evaluation must continue using parent_key,
        # never runtime Mongo UUIDs.
        # ----------------------------------------------------

        original_items_by_parent_id = {
            str(item.metadata["parent_id"]): item for item in query_result.items
        }

        final_items: list[RankedItem] = []

        for mmr_result in mmr_results:

            parent_id = str(mmr_result.parent.parent_id)

            original_item = original_items_by_parent_id.get(parent_id)

            if original_item is None:
                raise RuntimeError(
                    "MMR returned a parent that was "
                    "not present in the input "
                    "candidate set: "
                    f"parent_id={parent_id}"
                )

            final_items.append(
                RankedItem(
                    item_id=(original_item.item_id),
                    rank=mmr_result.rank,
                    score=(mmr_result.mmr_score),
                    metadata={
                        **original_item.metadata,
                        # ----------------------------
                        # RERANK TRACE
                        # ----------------------------
                        "pre_mmr_rank": (mmr_result.reranker_rank),
                        "reranker_score": (mmr_result.reranker_score),
                        # ----------------------------
                        # MMR TRACE
                        # ----------------------------
                        "query_similarity": (mmr_result.query_similarity),
                        "max_redundancy": (mmr_result.max_redundancy),
                        "relevance_contribution": (mmr_result.relevance_contribution),
                        "redundancy_penalty": (mmr_result.redundancy_penalty),
                        "mmr_score": (mmr_result.mmr_score),
                    },
                )
            )

        # ----------------------------------------------------
        # 7. LATENCY
        # ----------------------------------------------------

        mmr_latency_ms = (perf_counter() - started) * 1000.0

        previous_latency_ms = query_result.latency_ms or 0.0

        total_latency_ms = previous_latency_ms + mmr_latency_ms

        logger.info(
            f"query_id={query_result.query_id} "
            f"mmr_candidates="
            f"{len(mmr_candidates)} "
            f"selected_parents="
            f"{len(final_items)} "
            f"mmr_ms="
            f"{mmr_latency_ms:.2f} "
            f"total_pipeline_ms="
            f"{total_latency_ms:.2f}"
        )

        return QueryRankingResult(
            query_id=query_result.query_id,
            query=query_result.query,
            retrieval_level=(RetrievalLevel.PARENT),
            items=final_items,
            latency_ms=total_latency_ms,
        )

    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    @staticmethod
    def _cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        """
        Dense cosine similarity used for query-to-parent
        relevance.

        ParentMMRSelector performs the parent-to-parent
        redundancy calculation internally.
        """

        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

        if denominator == 0.0:
            return 0.0

        return float(
            np.dot(
                vector_a,
                vector_b,
            )
            / denominator
        )
