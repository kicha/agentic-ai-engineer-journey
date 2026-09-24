from __future__ import annotations

import logging
from time import perf_counter

from app.repositories.parent_repository import (
    ParentRepository,
)
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)

logger = logging.getLogger(__name__)


class BaselineParentExpansionRunner:
    """
    Expand child-level baseline retrieval results into parent rankings.

    Flow
    ----
    Gold query
        -> Dense / Sparse / Hybrid child retrieval
        -> group children by parent_id
        -> fetch authoritative ParentChunk from MongoDB
        -> rank parents by the best contributing child rank
        -> emit stable parent_key ranking

    Important
    ---------
    This runner performs NO metric calculations.

    Its responsibility is only to convert child retrieval output
    into the exact parent-level ranking that will later be scored
    by BenchmarkEvaluator.
    """

    def __init__(
        self,
        parent_repository: ParentRepository,
        parent_candidate_limit: int = 5,
    ) -> None:

        if parent_candidate_limit <= 0:
            raise ValueError("parent_candidate_limit must be greater than 0.")

        self.parent_repository = parent_repository
        self.parent_candidate_limit = parent_candidate_limit

    # ========================================================
    # RUN
    # ========================================================

    def expand(
        self,
        child_run_result: BenchmarkRunResult,
    ) -> BenchmarkRunResult:
        """
        Convert one completed child-level benchmark run into a parent-level
        benchmark run.

        The child retrieval is NOT executed again.

        This guarantees that:
        - child metrics
        - parent metrics

        are both derived from the exact same retrieval results.
        """

        if child_run_result.retrieval_level != RetrievalLevel.CHILD:
            raise ValueError("child_run_result must use CHILD retrieval level.")

        parent_results: list[QueryRankingResult] = []

        for child_query_result in child_run_result.results:

            started = perf_counter()

            parent_items = self._expand_to_parents(
                child_query_result=child_query_result,
            )

            expansion_latency_ms = (perf_counter() - started) * 1000.0

            child_latency_ms = child_query_result.latency_ms or 0.0

            total_latency_ms = child_latency_ms + expansion_latency_ms

            parent_results.append(
                QueryRankingResult(
                    query_id=(child_query_result.query_id),
                    query=child_query_result.query,
                    retrieval_level=(RetrievalLevel.PARENT),
                    items=parent_items,
                    latency_ms=total_latency_ms,
                )
            )

            logger.info(
                f"query_id="
                f"{child_query_result.query_id} "
                f"strategy="
                f"{child_run_result.strategy.value} "
                f"child_candidates="
                f"{len(child_query_result.items)} "
                f"unique_parents="
                f"{len(parent_items)} "
                f"parent_expansion_ms="
                f"{expansion_latency_ms:.2f}"
            )

        return BenchmarkRunResult(
            benchmark_version=(child_run_result.benchmark_version),
            strategy=(child_run_result.strategy),
            retrieval_level=(RetrievalLevel.PARENT),
            results=parent_results,
        )

    # ========================================================
    # CHILD -> PARENT EXPANSION
    # ========================================================

    def _expand_to_parents(
        self,
        child_query_result: QueryRankingResult,
    ) -> list[RankedItem]:
        """
        Expand ranked child results into unique ParentChunks.

        Parent ordering is based on the earliest-ranked child that
        points to each parent.

        Example:

            child rank 1 -> Parent A
            child rank 2 -> Parent A
            child rank 3 -> Parent B
            child rank 4 -> Parent C

        becomes:

            parent rank 1 -> Parent A
            parent rank 2 -> Parent B
            parent rank 3 -> Parent C

        Duplicate children from the same parent therefore do not
        consume multiple parent ranking positions.
        """
        ranked_items: list[RankedItem] = []
        seen_parent_ids: set[str] = set()

        for child_item in child_query_result.items:

            parent_id = child_item.metadata.get("parent_id")

            if not parent_id:
                raise RuntimeError(
                    "Child benchmark result does not " "contain parent_id metadata."
                )

            parent_id_string = str(parent_id)

            # ----------------------------------------------------
            # Parent deduplication.
            #
            # The first occurrence is the strongest child because
            # child_query_result.items is already ranked.
            # ----------------------------------------------------

            if parent_id_string in seen_parent_ids:
                continue

            parent = self.parent_repository.get_by_id(parent_id_string)

            if parent is None:
                raise RuntimeError(
                    "Dangling ChildChunk reference "
                    "detected during benchmark parent "
                    f"expansion: parent_id="
                    f"{parent_id_string}"
                )

            seen_parent_ids.add(parent_id_string)

            ranked_items.append(
                RankedItem(
                    item_id=parent.parent_key,
                    rank=len(ranked_items) + 1,
                    score=child_item.score,
                    metadata={
                        "parent_id": parent.parent_id,
                        "document_id": (parent.document_id),
                    },
                )
            )

            if len(ranked_items) >= self.parent_candidate_limit:
                break

        return ranked_items
