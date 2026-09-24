from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from qdrant_client.models import ScoredPoint

from app.retrieval.hybrid_child_retriever import (
    HybridChildRetriever,
)
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.gold_models import (
    GoldDatasetV2,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)

logger = logging.getLogger(__name__)


class BaselineRetrievalRunner:
    """
    Run Dense, Sparse, or Hybrid CHILD retrieval against Benchmark V2.

    Responsibilities:
    - execute the requested retrieval strategy
    - measure per-query retrieval latency
    - convert Qdrant ScoredPoint results into stable child_key rankings
    - return the standardized BenchmarkRunResult contract

    This class deliberately does NOT calculate retrieval metrics.
    Metrics belong exclusively to BenchmarkEvaluator / RankingMetrics.
    """

    def __init__(
        self,
        retriever: HybridChildRetriever,
        child_candidate_limit: int = 10,
    ) -> None:

        if child_candidate_limit <= 0:
            raise ValueError("child_candidate_limit must be greater than 0.")

        self.retriever = retriever
        self.child_candidate_limit = child_candidate_limit

    # ========================================================
    # PUBLIC RUNNER
    # ========================================================

    def run(
        self,
        gold_dataset: GoldDatasetV2,
        strategy: RetrievalStrategy,
    ) -> BenchmarkRunResult:
        """
        Execute one retrieval strategy across the full gold dataset.

        Every gold query receives a QueryRankingResult, including
        unanswerable queries. The evaluation layer later decides how
        those queries contribute to aggregate metrics.
        """

        retrieve = self._strategy_callable(
            strategy=strategy,
        )

        results: list[QueryRankingResult] = []

        for gold_query in gold_dataset.queries:

            started = perf_counter()

            hits = retrieve(
                query=gold_query.query,
                limit=self.child_candidate_limit,
            )

            latency_ms = (perf_counter() - started) * 1000.0

            ranked_items = self._to_ranked_items(
                hits=hits,
            )

            results.append(
                QueryRankingResult(
                    query_id=gold_query.query_id,
                    query=gold_query.query,
                    retrieval_level=RetrievalLevel.CHILD,
                    items=ranked_items,
                    latency_ms=latency_ms,
                )
            )

            logger.info(
                f"query_id={gold_query.query_id} "
                f"strategy={strategy.value} "
                f"retrieved_children={len(ranked_items)} "
                f"latency_ms={latency_ms:.2f}"
            )

        return BenchmarkRunResult(
            benchmark_version=(gold_dataset.benchmark_version),
            strategy=strategy,
            retrieval_level=RetrievalLevel.CHILD,
            results=results,
        )

    # ========================================================
    # STRATEGY DISPATCH
    # ========================================================

    def _strategy_callable(
        self,
        strategy: RetrievalStrategy,
    ) -> Callable[..., list[ScoredPoint]]:
        """
        Resolve the benchmark strategy to the actual retriever method.

        Keeping this dispatch in one place prevents strategy-specific
        branching from leaking into the main benchmark loop.
        """

        if strategy == RetrievalStrategy.DENSE:
            return self.retriever.retrieve_dense

        if strategy == RetrievalStrategy.SPARSE:
            return self.retriever.retrieve_sparse

        if strategy == RetrievalStrategy.HYBRID:
            return self.retriever.retrieve_hybrid

        raise ValueError(
            f"Unsupported baseline retrieval strategy: " f"{strategy.value}"
        )

    # ========================================================
    # QDRANT RESULT CONVERSION
    # ========================================================

    @staticmethod
    def _to_ranked_items(
        hits: list[ScoredPoint],
    ) -> list[RankedItem]:
        """
        Convert Qdrant child hits into the common benchmark contract.

        child_key—not child_id—is used because Benchmark V2 gold
        relevance is tied to deterministic stable retrieval identities.
        """

        ranked_items: list[RankedItem] = []

        seen_child_keys: set[str] = set()

        for hit in hits:

            payload = hit.payload or {}

            child_key = payload.get("child_key")

            if not child_key:
                raise RuntimeError(
                    "Retrieved benchmark child does not contain " "child_key."
                )

            child_key_string = str(child_key)

            # Qdrant should not return duplicate points, but protecting
            # the benchmark boundary here prevents duplicate retrieval
            # units from corrupting ranking metrics.
            if child_key_string in seen_child_keys:
                raise RuntimeError(
                    "Duplicate child_key returned by retrieval: " f"{child_key_string}"
                )

            seen_child_keys.add(child_key_string)

            ranked_items.append(
                RankedItem(
                    item_id=child_key_string,
                    rank=len(ranked_items) + 1,
                    score=float(hit.score),
                    metadata={
                        "parent_id": payload.get("parent_id"),
                        "parent_key": payload.get("parent_key"),
                        "document_id": payload.get("document_id"),
                    },
                )
            )

        return ranked_items
