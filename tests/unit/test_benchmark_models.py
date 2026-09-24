import pytest
from pydantic import ValidationError

from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)

# ============================================================
# RANKED ITEM
# ============================================================


def test_ranked_item_accepts_valid_values() -> None:
    """
    Purpose:
    Verify that one valid retrieval result can be represented by the
    canonical RankedItem model.

    Why this matters:
    Every runner will eventually emit RankedItem objects, so this model
    is the smallest common contract between retrieval and evaluation.
    """

    item = RankedItem(
        item_id="PARENT-001",
        rank=1,
        score=0.91,
    )

    assert item.item_id == "PARENT-001"
    assert item.rank == 1
    assert item.score == pytest.approx(0.91)


def test_ranked_item_rejects_zero_rank() -> None:
    """
    Purpose:
    Ensure ranking positions are one-based.

    Regression protected:
    Accidentally mixing Python's zero-based list indexing with
    information-retrieval rank numbering.
    """

    with pytest.raises(
        ValidationError,
    ):
        RankedItem(
            item_id="PARENT-001",
            rank=0,
        )


# ============================================================
# QUERY RANKING RESULT
# ============================================================


def test_query_ranking_returns_ranked_ids_in_order() -> None:
    """
    Purpose:
    Verify that the model exposes stable IDs in exact ranking order.

    Why this matters:
    RankingEvaluator consumes this ordering directly, so changing the
    sequence would change MRR, MAP and nDCG.
    """

    result = QueryRankingResult(
        query_id="Q001",
        query="What does ef_construct control?",
        retrieval_level=RetrievalLevel.PARENT,
        items=[
            RankedItem(
                item_id="PARENT-003",
                rank=1,
            ),
            RankedItem(
                item_id="PARENT-001",
                rank=2,
            ),
        ],
    )

    assert result.ranked_ids() == [
        "PARENT-003",
        "PARENT-001",
    ]


def test_query_ranking_rejects_duplicate_item_ids() -> None:
    """
    Purpose:
    Ensure the same retrieval unit cannot appear twice in one ranking.

    Why this matters:
    Duplicate parents could artificially increase hit counts or distort
    Precision, Recall and Average Precision.
    """

    with pytest.raises(
        ValidationError,
        match="duplicate item IDs",
    ):
        QueryRankingResult(
            query_id="Q001",
            query="Test query",
            retrieval_level=RetrievalLevel.PARENT,
            items=[
                RankedItem(
                    item_id="PARENT-001",
                    rank=1,
                ),
                RankedItem(
                    item_id="PARENT-001",
                    rank=2,
                ),
            ],
        )


def test_query_ranking_rejects_non_contiguous_ranks() -> None:
    """
    Purpose:
    Verify ranks always form 1, 2, 3, ... without gaps.

    Why this matters:
    Metric calculations rely on the item's actual position. A ranking
    such as 1, 3, 5 would make position-based metrics ambiguous.
    """

    with pytest.raises(
        ValidationError,
        match="contiguous",
    ):
        QueryRankingResult(
            query_id="Q001",
            query="Test query",
            retrieval_level=RetrievalLevel.PARENT,
            items=[
                RankedItem(
                    item_id="PARENT-001",
                    rank=1,
                ),
                RankedItem(
                    item_id="PARENT-002",
                    rank=3,
                ),
            ],
        )


def test_query_ranking_allows_empty_results() -> None:
    """
    Purpose:
    Verify a retriever is allowed to return zero candidates.

    Why this matters:
    Empty retrieval is a legitimate system outcome and should later
    produce zero ranking metrics rather than causing schema failure.
    """

    result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.PARENT,
        items=[],
    )

    assert result.ranked_ids() == []


# ============================================================
# COMPLETE BENCHMARK RUN
# ============================================================


def test_benchmark_run_converts_results_to_evaluator_mapping() -> None:
    """
    Purpose:
    Verify BenchmarkRunResult can produce exactly the dictionary
    expected by BenchmarkEvaluator.

    This is the central bridge:

        runner output
            ->
        BenchmarkRunResult
            ->
        ranked_ids_by_query_id()
            ->
        BenchmarkEvaluator
    """

    run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.HYBRID,
        retrieval_level=RetrievalLevel.PARENT,
        results=[
            QueryRankingResult(
                query_id="Q001",
                query="Query one",
                retrieval_level=RetrievalLevel.PARENT,
                items=[
                    RankedItem(
                        item_id="PARENT-001",
                        rank=1,
                    ),
                    RankedItem(
                        item_id="PARENT-002",
                        rank=2,
                    ),
                ],
            ),
            QueryRankingResult(
                query_id="Q002",
                query="Query two",
                retrieval_level=RetrievalLevel.PARENT,
                items=[
                    RankedItem(
                        item_id="PARENT-010",
                        rank=1,
                    ),
                ],
            ),
        ],
    )

    assert run.ranked_ids_by_query_id() == {
        "Q001": [
            "PARENT-001",
            "PARENT-002",
        ],
        "Q002": [
            "PARENT-010",
        ],
    }


def test_benchmark_run_rejects_duplicate_query_ids() -> None:
    """
    Purpose:
    Ensure each gold query contributes exactly one ranked result.

    Why this matters:
    Duplicate query results could cause one query to be evaluated twice
    while another query is missing, invalidating aggregate metrics.
    """

    with pytest.raises(
        ValidationError,
        match="duplicate query IDs",
    ):
        BenchmarkRunResult(
            benchmark_version="2.0",
            strategy=RetrievalStrategy.DENSE,
            retrieval_level=RetrievalLevel.PARENT,
            results=[
                QueryRankingResult(
                    query_id="Q001",
                    query="Query one",
                    retrieval_level=RetrievalLevel.PARENT,
                    items=[],
                ),
                QueryRankingResult(
                    query_id="Q001",
                    query="Query one duplicate",
                    retrieval_level=RetrievalLevel.PARENT,
                    items=[],
                ),
            ],
        )


def test_benchmark_run_rejects_mixed_retrieval_levels() -> None:
    """
    Purpose:
    Prevent parent-level and child-level outputs from being combined in
    one benchmark run.

    Why this matters:
    Parent relevance and child relevance are different gold contracts.
    Aggregating them together would make metrics semantically invalid.
    """

    with pytest.raises(
        ValidationError,
        match="retrieval level",
    ):
        BenchmarkRunResult(
            benchmark_version="2.0",
            strategy=RetrievalStrategy.HYBRID,
            retrieval_level=RetrievalLevel.PARENT,
            results=[
                QueryRankingResult(
                    query_id="Q001",
                    query="Query one",
                    retrieval_level=RetrievalLevel.PARENT,
                    items=[],
                ),
                QueryRankingResult(
                    query_id="Q002",
                    query="Query two",
                    retrieval_level=RetrievalLevel.CHILD,
                    items=[],
                ),
            ],
        )


def test_benchmark_run_accepts_optional_latency() -> None:
    """
    Purpose:
    Verify per-query latency can be preserved alongside ranked results.

    Why this matters:
    Retrieval quality alone is not enough for production comparison.
    Later runner reports will compare quality and latency together.
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.PARENT,
        items=[],
        latency_ms=14.7,
    )

    assert query_result.latency_ms == pytest.approx(14.7)
