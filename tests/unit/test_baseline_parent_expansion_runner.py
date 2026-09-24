from unittest.mock import Mock

import pytest

from app.models.parent_chunk import ParentChunk
from app.repositories.parent_repository import (
    ParentRepository,
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
from benchmarks.runners.baseline_parent_expansion_runner import (
    BaselineParentExpansionRunner,
)
from benchmarks.runners.baseline_retrieval_runner import (
    BaselineRetrievalRunner,
)

# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def child_runner() -> Mock:
    """
    Purpose:
    Mock the child retrieval boundary.

    Why this matters:
    This unit test is specifically testing child-to-parent expansion.
    Dense embeddings, BM25, and Qdrant retrieval must not run here.
    """

    return Mock(spec=BaselineRetrievalRunner)


@pytest.fixture
def parent_repository() -> Mock:
    """
    Purpose:
    Mock MongoDB parent persistence.

    Why this matters:
    Parent expansion needs authoritative ParentChunks, but a unit test
    must not depend on a live MongoDB server.
    """

    return Mock(spec=ParentRepository)


# ============================================================
# PARENT DEDUPLICATION
# ============================================================


def test_parent_expansion_deduplicates_children_from_same_parent(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify multiple highly-ranked children belonging to the same parent
    consume only one parent ranking position.

    Scenario:
        child rank 1 -> Parent A
        child rank 2 -> Parent A
        child rank 3 -> Parent B

    Expected parent ranking:
        rank 1 -> Parent A
        rank 2 -> Parent B

    Why this matters:
    This is the central reason parent-level Recall may be substantially
    higher than child-level Recall.
    """

    parent_a = Mock(spec=ParentChunk)
    parent_a.parent_id = "PARENT-ID-A"
    parent_a.parent_key = "PARENT-KEY-A"
    parent_a.document_id = "DOC-001"

    parent_b = Mock(spec=ParentChunk)
    parent_b.parent_id = "PARENT-ID-B"
    parent_b.parent_key = "PARENT-KEY-B"
    parent_b.document_id = "DOC-001"

    parent_repository.get_by_id.side_effect = lambda parent_id: {
        "PARENT-ID-A": parent_a,
        "PARENT-ID-B": parent_b,
    }.get(parent_id)

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
        parent_candidate_limit=5,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[
            RankedItem(
                item_id="CHILD-001",
                rank=1,
                score=0.90,
                metadata={
                    "parent_id": "PARENT-ID-A",
                },
            ),
            RankedItem(
                item_id="CHILD-002",
                rank=2,
                score=0.85,
                metadata={
                    "parent_id": "PARENT-ID-A",
                },
            ),
            RankedItem(
                item_id="CHILD-003",
                rank=3,
                score=0.80,
                metadata={
                    "parent_id": "PARENT-ID-B",
                },
            ),
        ],
    )

    result = runner._expand_to_parents(
        child_query_result=child_result,
    )

    assert [item.item_id for item in result] == [
        "PARENT-KEY-A",
        "PARENT-KEY-B",
    ]

    # Parent A must retain the score of its strongest child,
    # which was the child ranked first.
    assert result[0].score == pytest.approx(0.90)


def test_parent_expansion_preserves_best_child_order(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify baseline parent ranking is determined by the earliest
    contributing child rank.

    Why this matters:
    Before CrossEncoder reranking, the only ranking signal available
    for a parent is the strength/order of its retrieved children.
    """

    parent_a = Mock(spec=ParentChunk)
    parent_a.parent_id = "PA"
    parent_a.parent_key = "PARENT-A"
    parent_a.document_id = "DOC-001"

    parent_b = Mock(spec=ParentChunk)
    parent_b.parent_id = "PB"
    parent_b.parent_key = "PARENT-B"
    parent_b.document_id = "DOC-002"

    parent_repository.get_by_id.side_effect = lambda parent_id: {
        "PA": parent_a,
        "PB": parent_b,
    }[parent_id]

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[
            RankedItem(
                item_id="C1",
                rank=1,
                score=0.92,
                metadata={
                    "parent_id": "PB",
                },
            ),
            RankedItem(
                item_id="C2",
                rank=2,
                score=0.88,
                metadata={
                    "parent_id": "PA",
                },
            ),
        ],
    )

    result = runner._expand_to_parents(
        child_query_result=child_result,
    )

    assert [item.item_id for item in result] == [
        "PARENT-B",
        "PARENT-A",
    ]


def test_parent_expansion_respects_parent_candidate_limit(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify parent-level candidate count is independently bounded.

    Why this matters:
    Ten child candidates may collapse into fewer parents, while the
    final parent evaluation may intentionally use a smaller top-k.
    """

    parents = {}

    for index in range(3):
        parent = Mock(spec=ParentChunk)
        parent.parent_id = f"PID-{index}"
        parent.parent_key = f"PKEY-{index}"
        parent.document_id = "DOC-001"

        parents[parent.parent_id] = parent

    parent_repository.get_by_id.side_effect = lambda parent_id: parents[parent_id]

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
        parent_candidate_limit=2,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[
            RankedItem(
                item_id=f"C{index}",
                rank=index + 1,
                metadata={
                    "parent_id": f"PID-{index}",
                },
            )
            for index in range(3)
        ],
    )

    result = runner._expand_to_parents(
        child_query_result=child_result,
    )

    assert len(result) == 2


def test_parent_expansion_rejects_missing_parent_id_metadata(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Fail immediately when the child benchmark contract does not carry
    the parent identity required for expansion.

    Regression protected:
    Accidentally stripping Qdrant parent metadata when converting a
    ScoredPoint into RankedItem.
    """

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[
            RankedItem(
                item_id="CHILD-001",
                rank=1,
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain parent_id metadata",
    ):
        runner._expand_to_parents(
            child_query_result=child_result,
        )


def test_parent_expansion_rejects_dangling_parent_reference(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify a child pointing to a missing MongoDB ParentChunk causes a
    hard benchmark failure.

    Why this matters:
    A dangling parent reference is a persistence-integrity defect,
    not a valid retrieval miss, and must never silently affect metrics.
    """

    parent_repository.get_by_id.return_value = None

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[
            RankedItem(
                item_id="CHILD-001",
                rank=1,
                metadata={
                    "parent_id": "MISSING-PARENT",
                },
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Dangling ChildChunk reference",
    ):
        runner._expand_to_parents(
            child_query_result=child_result,
        )


def test_parent_expansion_allows_empty_child_results(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify an empty child ranking cleanly becomes an empty parent
    ranking.

    Why this matters:
    A real retriever may return no candidates, and this should become
    zero retrieval metrics rather than an exception.
    """

    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
    )

    child_result = QueryRankingResult(
        query_id="Q001",
        query="Test query",
        retrieval_level=RetrievalLevel.CHILD,
        items=[],
    )

    result = runner._expand_to_parents(
        child_query_result=child_result,
    )

    assert result == []

    parent_repository.get_by_id.assert_not_called()


def test_parent_expansion_rejects_invalid_parent_limit(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Ensure parent-level top-k configuration is always positive.

    Why this matters:
    Zero or negative candidate limits have no meaningful ranking
    interpretation.
    """

    with pytest.raises(
        ValueError,
        match="parent_candidate_limit must be greater than 0",
    ):
        BaselineParentExpansionRunner(
            parent_repository=parent_repository,
            parent_candidate_limit=0,
        )


def test_expand_converts_child_run_to_parent_run(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Verify a completed child-level BenchmarkRunResult is converted into
    a parent-level BenchmarkRunResult without executing retrieval again.
    Why this matters:
    Child and parent metrics must be based on the exact same retrieval
    output so the comparison is valid.
    """
    parent = Mock(spec=ParentChunk)
    parent.parent_id = "PARENT-ID-001"
    parent.parent_key = "PARENT-KEY-001"
    parent.document_id = "DOC-001"
    parent_repository.get_by_id.return_value = parent
    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
        parent_candidate_limit=5,
    )
    child_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.DENSE,
        retrieval_level=RetrievalLevel.CHILD,
        results=[
            QueryRankingResult(
                query_id="Q001",
                query="Test query",
                retrieval_level=RetrievalLevel.CHILD,
                items=[
                    RankedItem(
                        item_id="CHILD-001",
                        rank=1,
                        score=0.90,
                        metadata={
                            "parent_id": ("PARENT-ID-001"),
                        },
                    ),
                ],
                latency_ms=10.0,
            ),
        ],
    )
    result = runner.expand(
        child_run_result=child_run,
    )
    assert result.retrieval_level == RetrievalLevel.PARENT
    assert result.strategy == RetrievalStrategy.DENSE
    assert result.results[0].ranked_ids() == [
        "PARENT-KEY-001",
    ]


def test_expand_rejects_non_child_run(
    parent_repository: Mock,
) -> None:
    """
    Purpose:
    Ensure parent expansion only accepts child-level benchmark results.
    Why this matters:
    Expanding a parent-level run again would represent an invalid
    benchmark pipeline.
    """
    runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
    )
    invalid_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.DENSE,
        retrieval_level=RetrievalLevel.PARENT,
        results=[],
    )
    with pytest.raises(
        ValueError,
        match="must use CHILD retrieval level",
    ):
        runner.expand(
            child_run_result=invalid_run,
        )
