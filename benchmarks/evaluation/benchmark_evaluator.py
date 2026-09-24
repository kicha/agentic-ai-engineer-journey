from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from benchmarks.evaluation.gold_models import (
    Answerability,
    GoldDatasetV2,
)
from benchmarks.evaluation.ranking_evaluator import (
    RankingEvaluationResult,
    RankingEvaluator,
    RetrievalLevel,
)

# ============================================================
# AGGREGATED BENCHMARK METRICS
# ============================================================


class BenchmarkMetricsResult(BaseModel):
    """
    Dataset-level mean ranking metrics.

    These metrics are calculated only across answerable queries.

    Unanswerable queries require a separate abstention-quality
    evaluation and therefore must not artificially lower normal
    retrieval relevance metrics.
    """

    model_config = ConfigDict(
        frozen=True,
    )

    mean_hit_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_average_precision_at_k: float
    mean_ndcg_at_k: float


# ============================================================
# COMPLETE BENCHMARK RESULT
# ============================================================


class BenchmarkEvaluationResult(BaseModel):
    """
    Complete evaluation result for one benchmark run.

    Contains:
    - benchmark configuration
    - query counts
    - all per-query ranking results
    - aggregate metrics over answerable queries
    """

    model_config = ConfigDict(
        frozen=True,
    )

    retrieval_level: RetrievalLevel
    k: int

    total_query_count: int
    answerable_query_count: int
    unanswerable_query_count: int

    query_results: list[RankingEvaluationResult]

    metrics: BenchmarkMetricsResult


# ============================================================
# BENCHMARK EVALUATOR
# ============================================================


class BenchmarkEvaluator:
    """
    Evaluate ranked retrieval results for an entire GoldDatasetV2.

    This class deliberately does NOT perform retrieval.

    The caller supplies ranked identifiers for each query.

    Example:

        {
            "Q001": [
                "parent-key-1",
                "parent-key-2",
            ],
            "Q002": [
                "parent-key-7",
                "parent-key-3",
            ],
        }

    Responsibilities:
    - validate benchmark input completeness
    - evaluate every query using RankingEvaluator
    - retain individual query diagnostics
    - aggregate answerable-query ranking metrics
    """

    def __init__(
        self,
        ranking_evaluator: RankingEvaluator | None = None,
    ) -> None:

        # Dependency injection keeps BenchmarkEvaluator decoupled
        # from the concrete RankingEvaluator implementation.
        #
        # In normal production use we construct the real evaluator.
        # Unit tests can inject a mock if we specifically want to
        # isolate BenchmarkEvaluator orchestration behavior.
        self.ranking_evaluator = ranking_evaluator or RankingEvaluator()

    def evaluate(
        self,
        gold_dataset: GoldDatasetV2,
        ranked_ids_by_query_id: Mapping[
            str,
            Sequence[str],
        ],
        retrieval_level: RetrievalLevel,
        k: int,
    ) -> BenchmarkEvaluationResult:

        self._validate_k(k)

        self._validate_query_ids(
            gold_dataset=gold_dataset,
            ranked_ids_by_query_id=(ranked_ids_by_query_id),
        )

        query_results: list[RankingEvaluationResult] = []

        answerable_results: list[RankingEvaluationResult] = []

        answerable_query_count = 0
        unanswerable_query_count = 0

        # Evaluate every query so that diagnostics remain complete.
        #
        # However, only answerable queries are accumulated into
        # answerable_results and therefore contribute to the
        # dataset-level ranking metrics.
        for gold_query in gold_dataset.queries:

            ranked_ids = ranked_ids_by_query_id[gold_query.query_id]

            query_result = self.ranking_evaluator.evaluate(
                gold_query=gold_query,
                ranked_ids=ranked_ids,
                retrieval_level=retrieval_level,
                k=k,
            )

            query_results.append(query_result)

            if gold_query.answerability == Answerability.ANSWERABLE:
                answerable_query_count += 1

                answerable_results.append(query_result)

            else:
                unanswerable_query_count += 1

        metrics = self._aggregate_metrics(
            query_results=answerable_results,
        )

        return BenchmarkEvaluationResult(
            retrieval_level=retrieval_level,
            k=k,
            total_query_count=len(gold_dataset.queries),
            answerable_query_count=(answerable_query_count),
            unanswerable_query_count=(unanswerable_query_count),
            query_results=query_results,
            metrics=metrics,
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    @staticmethod
    def _aggregate_metrics(
        query_results: Sequence[RankingEvaluationResult],
    ) -> BenchmarkMetricsResult:
        """
        Calculate arithmetic means across answerable queries.

        MAP@K is therefore the mean of each query's AP@K.
        MRR is the mean of each query's reciprocal rank.
        """

        if not query_results:
            return BenchmarkMetricsResult(
                mean_hit_at_k=0.0,
                mean_precision_at_k=0.0,
                mean_recall_at_k=0.0,
                mean_reciprocal_rank=0.0,
                mean_average_precision_at_k=0.0,
                mean_ndcg_at_k=0.0,
            )

        return BenchmarkMetricsResult(
            mean_hit_at_k=fmean(result.metrics.hit_at_k for result in query_results),
            mean_precision_at_k=fmean(
                result.metrics.precision_at_k for result in query_results
            ),
            mean_recall_at_k=fmean(
                result.metrics.recall_at_k for result in query_results
            ),
            mean_reciprocal_rank=fmean(
                result.metrics.reciprocal_rank for result in query_results
            ),
            mean_average_precision_at_k=fmean(
                result.metrics.average_precision_at_k for result in query_results
            ),
            mean_ndcg_at_k=fmean(result.metrics.ndcg_at_k for result in query_results),
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_query_ids(
        gold_dataset: GoldDatasetV2,
        ranked_ids_by_query_id: Mapping[
            str,
            Sequence[str],
        ],
    ) -> None:
        """
        Ensure the retriever supplied exactly one ranking for every
        gold query and did not supply unknown query identifiers.

        Silently skipping a query would make benchmark comparisons
        misleading, so incomplete benchmark runs are rejected.
        """

        expected_query_ids = {query.query_id for query in gold_dataset.queries}

        actual_query_ids = set(ranked_ids_by_query_id)

        missing_query_ids = expected_query_ids - actual_query_ids

        unknown_query_ids = actual_query_ids - expected_query_ids

        if missing_query_ids:
            raise ValueError(
                "Missing ranked results for query IDs: " f"{sorted(missing_query_ids)}"
            )

        if unknown_query_ids:
            raise ValueError(
                "Ranked results contain unknown query IDs: "
                f"{sorted(unknown_query_ids)}"
            )

    @staticmethod
    def _validate_k(
        k: int,
    ) -> None:

        if k <= 0:
            raise ValueError("k must be greater than 0.")
