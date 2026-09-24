from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)

# ============================================================
# RETRIEVAL STRATEGY
# ============================================================


class RetrievalStrategy(StrEnum):
    """
    Canonical names for benchmarked retrieval strategies.

    Keeping these names centralized prevents runners from using
    slightly different labels for the same retrieval method.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"
    HYBRID_RERANK_MMR = "hybrid_rerank_mmr"
    QUERY_REWRITE = "query_rewrite"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"


# ============================================================
# ONE RANKED ITEM
# ============================================================


class RankedItem(BaseModel):
    """
    One ranked retrieval result returned by a benchmark runner.

    item_id
        Stable evaluation identity:
        - document_id
        - parent_key
        - child_key

    metadata
        Optional transport metadata required by downstream benchmark
        stages.

        Example for child retrieval:
            parent_id
            parent_key
            document_id

        This metadata is NOT used directly by ranking metrics.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    item_id: str = Field(
        min_length=1,
    )

    rank: int = Field(
        ge=1,
    )

    score: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================
# ONE QUERY'S RETRIEVAL OUTPUT
# ============================================================


class QueryRankingResult(BaseModel):
    """
    Standardized retrieval output for exactly one benchmark query.

    This model contains retrieval output only.

    It deliberately does NOT contain:
    - gold relevance
    - Hit@K
    - Recall@K
    - MRR
    - nDCG

    Metrics belong exclusively to the centralized evaluation layer.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    query_id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
    )

    retrieval_level: RetrievalLevel

    items: list[RankedItem]

    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
    )

    @model_validator(
        mode="after",
    )
    def validate_ranking(
        self,
    ) -> "QueryRankingResult":
        """
        Ensure a ranking contains:
        - unique stable identifiers
        - contiguous ranks beginning at 1

        Duplicate IDs usually indicate a bug in upstream
        deduplication and must not silently reach metric code.
        """

        item_ids = [item.item_id for item in self.items]

        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Ranking contains duplicate item IDs.")

        actual_ranks = [item.rank for item in self.items]

        expected_ranks = list(
            range(
                1,
                len(self.items) + 1,
            )
        )

        if actual_ranks != expected_ranks:
            raise ValueError("Ranking ranks must be contiguous and start at 1.")

        return self

    def ranked_ids(
        self,
    ) -> list[str]:
        """
        Return only stable ranked identifiers.

        This is the exact representation expected by
        RankingEvaluator / BenchmarkEvaluator.
        """

        return [item.item_id for item in self.items]


# ============================================================
# COMPLETE RUNNER OUTPUT
# ============================================================


class BenchmarkRunResult(BaseModel):
    """
    Standard output contract for one complete retrieval benchmark run.

    All retrieval strategies must eventually produce this shape before
    passing results to BenchmarkEvaluator.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    benchmark_version: str = Field(
        min_length=1,
    )

    strategy: RetrievalStrategy

    retrieval_level: RetrievalLevel

    results: list[QueryRankingResult]

    @model_validator(
        mode="after",
    )
    def validate_query_results(
        self,
    ) -> "BenchmarkRunResult":
        """
        Verify that:
        - every query_id appears only once
        - every query result uses the run's retrieval level

        Mixing child-level and parent-level results inside one run
        would make the aggregate metrics meaningless.
        """

        query_ids = [result.query_id for result in self.results]

        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Benchmark run contains duplicate query IDs.")

        invalid_levels = [
            result.query_id
            for result in self.results
            if result.retrieval_level != self.retrieval_level
        ]

        if invalid_levels:
            raise ValueError(
                "All query results must use the benchmark run "
                f"retrieval level. Invalid query IDs: "
                f"{invalid_levels}"
            )

        return self

    def ranked_ids_by_query_id(
        self,
    ) -> dict[str, list[str]]:
        """
        Convert the runner output into the mapping expected by
        BenchmarkEvaluator.

        Example:

            {
                "Q001": ["parent-a", "parent-b"],
                "Q002": ["parent-x", "parent-y"],
            }
        """

        return {result.query_id: result.ranked_ids() for result in self.results}
