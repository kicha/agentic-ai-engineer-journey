from __future__ import annotations

from enum import StrEnum
from typing import Sequence

from pydantic import BaseModel, ConfigDict

from benchmarks.evaluation.gold_models import (
    GoldQueryV2,
)
from benchmarks.evaluation.ranking_metrics import (
    RankingMetrics,
    RankingMetricsResult,
)

# ============================================================
# RETRIEVAL LEVEL
# ============================================================


class RetrievalLevel(StrEnum):
    DOCUMENT = "document"
    PARENT = "parent"
    CHILD = "child"


# ============================================================
# EVALUATION RESULT
# ============================================================


class RankingEvaluationResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
    )

    query_id: str
    query: str
    retrieval_level: RetrievalLevel
    k: int
    ranked_ids: list[str]
    relevance: dict[str, int]
    metrics: RankingMetricsResult


# ============================================================
# RANKING EVALUATOR
# ============================================================


class RankingEvaluator:
    """
    Evaluate one ranked retrieval result against one GoldQueryV2.

    Responsibilities:
    - obtain the correct relevance map from GoldQueryV2
    - enforce a consistent top-k evaluation boundary
    - reject duplicate ranked identifiers
    - delegate metric calculations to RankingMetrics
    - return a typed evaluation result

    This class does not perform retrieval and does not access
    Qdrant, MongoDB, embeddings, rerankers, or LLMs.
    """

    def evaluate(
        self,
        gold_query: GoldQueryV2,
        ranked_ids: Sequence[str],
        retrieval_level: RetrievalLevel,
        k: int,
    ) -> RankingEvaluationResult:

        self._validate_k(k)

        ranked_ids_list = list(ranked_ids)

        self._validate_unique_ranked_ids(
            ranked_ids=ranked_ids_list,
        )

        # ----------------------------------------------------
        # Establish one common top-k boundary.
        #
        # This is particularly important for reciprocal rank.
        # RankingMetrics.reciprocal_rank() itself has no k
        # parameter, so passing only top_k means the evaluator
        # produces MRR semantics bounded by this evaluation k.
        # ----------------------------------------------------

        top_k_ranked_ids = ranked_ids_list[:k]

        relevance = gold_query.relevance_map(retrieval_level.value)

        metrics = RankingMetrics.evaluate(
            ranked_ids=top_k_ranked_ids,
            relevance=relevance,
            k=k,
        )

        return RankingEvaluationResult(
            query_id=gold_query.query_id,
            query=gold_query.query,
            retrieval_level=retrieval_level,
            k=k,
            ranked_ids=top_k_ranked_ids,
            relevance=relevance,
            metrics=metrics,
        )

    @staticmethod
    def _validate_unique_ranked_ids(
        ranked_ids: Sequence[str],
    ) -> None:

        if len(ranked_ids) != len(set(ranked_ids)):
            raise ValueError("ranked_ids must not contain duplicate identifiers.")

    @staticmethod
    def _validate_k(
        k: int,
    ) -> None:

        if k <= 0:
            raise ValueError("k must be greater than 0.")
