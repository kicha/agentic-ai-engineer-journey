from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest

from app.models.parent_chunk import ParentChunk
from app.reranking.base import Reranker
from app.reranking.cross_encoder_reranker import (
    RerankedParentResult,
    RerankingResponse,
)
from app.retrieval.models import ParentRetrievalResult
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)
from benchmarks.runners.parent_reranking_runner import (
    ParentRerankingRunner,
)

# ============================================================
# TEST DOUBLES
# ============================================================


class FakeParentRepository:
    """
    Minimal in-memory replacement for ParentRepository.

    Why:
        ParentRerankingRunner should be tested without MongoDB.

    The fake preserves the important production contract:
        parent_id -> authoritative ParentChunk
    """

    def __init__(
        self,
        parents: dict[str, ParentChunk],
    ) -> None:
        self.parents = parents
        self.requested_parent_ids: list[str] = []

    def get_by_id(
        self,
        parent_id: str,
    ) -> ParentChunk | None:

        self.requested_parent_ids.append(parent_id)

        return self.parents.get(parent_id)


class FakeReranker(Reranker):
    """
    Deterministic parent reranker used by unit tests.

    It deliberately returns candidates in the order:

        original #3
        original #1
        original #2

    This makes it easy to verify that:

        hosted/local reranker index
            ->
        original parent candidate
            ->
        stable benchmark parent_key

    mapping remains correct.
    """

    def __init__(self) -> None:

        self.received_query: str | None = None

        self.received_candidates: list[ParentRetrievalResult] = []

        self.received_top_k: int | None = None

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[ParentRetrievalResult],
        top_k: int | None = None,
    ) -> RerankingResponse:

        self.received_query = query

        self.received_candidates = list(candidates)

        self.received_top_k = top_k

        reordered = [
            (3, 0.95),
            (1, 0.80),
            (2, 0.20),
        ]

        results: list[RerankedParentResult] = []

        effective_top_k = top_k if top_k is not None else len(reordered)

        for reranked_rank, (
            original_rank,
            reranker_score,
        ) in enumerate(
            reordered[:effective_top_k],
            start=1,
        ):

            candidate = candidates[original_rank - 1]

            results.append(
                RerankedParentResult(
                    original_rank=original_rank,
                    reranked_rank=reranked_rank,
                    vector_score=(candidate.best_score),
                    reranker_score=(reranker_score),
                    result=candidate,
                )
            )

        return RerankingResponse(
            query=query,
            candidate_count=len(candidates),
            returned_count=len(results),
            results=results,
        )


# ============================================================
# HELPERS
# ============================================================


def make_parent(
    *,
    parent_id: str,
    parent_key: str,
    content: str,
) -> ParentChunk:
    """
    Build the minimum real ParentChunk instance needed by the
    reranking unit test.

    model_construct() is appropriate here because ParentChunk's
    full ingestion schema is tested elsewhere.

    This test needs only:
        parent_id
        parent_key
        content
    """

    return ParentChunk.model_construct(
        parent_id=parent_id,
        parent_key=parent_key,
        content=content,
    )


def make_parent_run() -> BenchmarkRunResult:
    """
    Build one parent-level benchmark result containing three
    parent candidates in their pre-rerank order.
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query="How does HNSW search efficiently?",
        retrieval_level=(RetrievalLevel.PARENT),
        items=[
            RankedItem(
                item_id="parent-key-1",
                rank=1,
                score=0.91,
                metadata={
                    "parent_id": "parent-1",
                    "document_id": "DOC-001",
                    "source": "benchmark",
                },
            ),
            RankedItem(
                item_id="parent-key-2",
                rank=2,
                score=0.82,
                metadata={
                    "parent_id": "parent-2",
                    "document_id": "DOC-002",
                    "source": "benchmark",
                },
            ),
            RankedItem(
                item_id="parent-key-3",
                rank=3,
                score=0.73,
                metadata={
                    "parent_id": "parent-3",
                    "document_id": "DOC-003",
                    "source": "benchmark",
                },
            ),
        ],
        latency_ms=100.0,
    )

    return BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.HYBRID,
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )


def make_repository() -> FakeParentRepository:

    return FakeParentRepository(
        parents={
            "parent-1": make_parent(
                parent_id="parent-1",
                parent_key="parent-key-1",
                content="Parent one content.",
            ),
            "parent-2": make_parent(
                parent_id="parent-2",
                parent_key="parent-key-2",
                content="Parent two content.",
            ),
            "parent-3": make_parent(
                parent_id="parent-3",
                parent_key="parent-key-3",
                content="Parent three content.",
            ),
        }
    )


# ============================================================
# CONSTRUCTOR VALIDATION
# ============================================================


def test_parent_reranking_runner_rejects_non_positive_final_k() -> None:
    """
    final_k defines how many parents survive reranking.

    Zero or negative values have no meaningful interpretation
    and should fail immediately.
    """

    repository = make_repository()

    reranker = FakeReranker()

    with pytest.raises(
        ValueError,
        match="final_k must be greater than zero",
    ):
        ParentRerankingRunner(
            parent_repository=repository,
            reranker=reranker,
            final_k=0,
        )


# ============================================================
# RETRIEVAL LEVEL VALIDATION
# ============================================================


def test_parent_reranking_runner_requires_parent_level_run() -> None:
    """
    The canonical architecture requires:

        child retrieval
            ->
        parent expansion
            ->
        reranking

    Therefore ParentRerankingRunner must reject a child-level
    BenchmarkRunResult.
    """

    child_query_result = QueryRankingResult(
        query_id="Q001",
        query="test query",
        retrieval_level=(RetrievalLevel.CHILD),
        items=[],
        latency_ms=10.0,
    )

    child_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.HYBRID,
        retrieval_level=(RetrievalLevel.CHILD),
        results=[
            child_query_result,
        ],
    )

    runner = ParentRerankingRunner(
        parent_repository=(make_repository()),
        reranker=FakeReranker(),
        final_k=3,
    )

    with pytest.raises(
        ValueError,
        match=("ParentRerankingRunner requires " "a parent-level BenchmarkRunResult"),
    ):
        runner.run(
            parent_run_result=child_run,
        )


# ============================================================
# FULL RERANKING CONTRACT
# ============================================================


@patch("benchmarks.runners." "parent_reranking_runner." "perf_counter")
def test_parent_reranking_runner_reranks_existing_parent_candidates(
    mock_perf_counter,
) -> None:
    """
    Prove the complete benchmark adapter contract.

    The test verifies that:

    1. no child retrieval occurs,
    2. existing parent candidates are reused,
    3. MongoDB parent IDs are used to reload ParentChunks,
    4. original vector/fusion scores reach the reranker,
    5. reranker order becomes benchmark order,
    6. stable parent_key identities remain correct,
    7. metadata survives reranking,
    8. reranker trace metadata is added,
    9. benchmark_version is preserved,
    10. output strategy becomes HYBRID_RERANK.
    """

    # 25 ms reranking latency.
    mock_perf_counter.side_effect = [
        10.000,
        10.025,
    ]

    repository = make_repository()

    reranker = FakeReranker()

    runner = ParentRerankingRunner(
        parent_repository=repository,
        reranker=reranker,
        final_k=3,
    )

    parent_run = make_parent_run()

    result = runner.run(
        parent_run_result=parent_run,
    )

    # --------------------------------------------------------
    # BENCHMARK RUN IDENTITY
    # --------------------------------------------------------

    assert result.benchmark_version == "2.0"

    assert result.strategy == RetrievalStrategy.HYBRID_RERANK

    assert result.retrieval_level == RetrievalLevel.PARENT

    assert len(result.results) == 1

    # --------------------------------------------------------
    # AUTHORITATIVE PARENT LOOKUPS
    # --------------------------------------------------------

    assert repository.requested_parent_ids == [
        "parent-1",
        "parent-2",
        "parent-3",
    ]

    # --------------------------------------------------------
    # RERANKER INPUT CONTRACT
    # --------------------------------------------------------

    assert reranker.received_query == "How does HNSW search efficiently?"

    assert reranker.received_top_k == 3

    assert len(reranker.received_candidates) == 3

    assert [candidate.parent.content for candidate in reranker.received_candidates] == [
        "Parent one content.",
        "Parent two content.",
        "Parent three content.",
    ]

    assert [
        candidate.best_score for candidate in reranker.received_candidates
    ] == pytest.approx(
        [
            0.91,
            0.82,
            0.73,
        ]
    )

    # ParentRerankingRunner is not responsible for child-hit
    # reconstruction.
    assert all(
        candidate.matched_children == [] for candidate in reranker.received_candidates
    )

    # --------------------------------------------------------
    # OUTPUT QUERY CONTRACT
    # --------------------------------------------------------

    query_result = result.results[0]

    assert query_result.query_id == "Q001"

    assert query_result.query == "How does HNSW search efficiently?"

    assert query_result.retrieval_level == RetrievalLevel.PARENT

    # --------------------------------------------------------
    # RERANKED ORDER
    #
    # Fake reranker returns:
    #
    # original #3
    # original #1
    # original #2
    # --------------------------------------------------------

    assert [item.item_id for item in query_result.items] == [
        "parent-key-3",
        "parent-key-1",
        "parent-key-2",
    ]

    assert [item.rank for item in query_result.items] == [
        1,
        2,
        3,
    ]

    assert [item.score for item in query_result.items] == pytest.approx(
        [
            0.95,
            0.80,
            0.20,
        ]
    )

    # --------------------------------------------------------
    # ORIGINAL METADATA MUST SURVIVE
    # --------------------------------------------------------

    first_item = query_result.items[0]

    assert first_item.metadata["parent_id"] == "parent-3"

    assert first_item.metadata["document_id"] == "DOC-003"

    assert first_item.metadata["source"] == "benchmark"

    # --------------------------------------------------------
    # RERANK TRACE METADATA
    # --------------------------------------------------------

    assert first_item.metadata["pre_rerank_rank"] == 3

    assert first_item.metadata["pre_rerank_score"] == pytest.approx(0.73)

    assert first_item.metadata["reranker_score"] == pytest.approx(0.95)


# ============================================================
# LATENCY
# ============================================================


@patch("benchmarks.runners." "parent_reranking_runner." "perf_counter")
def test_parent_reranking_runner_adds_reranking_to_existing_latency(
    mock_perf_counter,
) -> None:
    """
    Reranking latency should be added to the previous pipeline
    latency rather than replacing it.

    Existing latency:
        100 ms

    Simulated reranking:
         25 ms

    Expected cumulative latency:
        125 ms
    """

    mock_perf_counter.side_effect = [
        20.000,
        20.025,
    ]

    runner = ParentRerankingRunner(
        parent_repository=(make_repository()),
        reranker=FakeReranker(),
        final_k=3,
    )

    result = runner.run(
        parent_run_result=(make_parent_run()),
    )

    assert result.results[0].latency_ms == pytest.approx(
        125.0,
        abs=0.01,
    )


# ============================================================
# MISSING parent_id
# ============================================================


def test_parent_reranking_runner_rejects_missing_parent_id() -> None:
    """
    A parent benchmark item without parent_id cannot be
    rehydrated from the authoritative Mongo repository.

    This is a benchmark transport-contract violation.
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query="test query",
        retrieval_level=(RetrievalLevel.PARENT),
        items=[
            RankedItem(
                item_id="parent-key-1",
                rank=1,
                score=0.90,
                metadata={
                    "document_id": "DOC-001",
                },
            ),
        ],
        latency_ms=10.0,
    )

    parent_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.HYBRID,
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )

    runner = ParentRerankingRunner(
        parent_repository=(make_repository()),
        reranker=FakeReranker(),
        final_k=1,
    )

    with pytest.raises(
        RuntimeError,
        match=("Parent benchmark item is missing " "parent_id metadata"),
    ):
        runner.run(
            parent_run_result=parent_run,
        )


# ============================================================
# DANGLING PARENT REFERENCE
# ============================================================


def test_parent_reranking_runner_rejects_dangling_parent_reference() -> None:
    """
    If benchmark metadata references a Mongo parent that no
    longer exists, reranking must fail loudly.

    Silently skipping it would corrupt benchmark rankings.
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query="test query",
        retrieval_level=(RetrievalLevel.PARENT),
        items=[
            RankedItem(
                item_id="missing-parent-key",
                rank=1,
                score=0.90,
                metadata={
                    "parent_id": ("missing-parent"),
                },
            ),
        ],
        latency_ms=10.0,
    )

    parent_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=RetrievalStrategy.HYBRID,
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )

    runner = ParentRerankingRunner(
        parent_repository=(make_repository()),
        reranker=FakeReranker(),
        final_k=1,
    )

    with pytest.raises(
        RuntimeError,
        match=("Dangling parent reference " "detected during reranking"),
    ):
        runner.run(
            parent_run_result=parent_run,
        )


# ============================================================
# FINAL K
# ============================================================


def test_parent_reranking_runner_applies_final_k() -> None:
    """
    Parent candidate breadth and final-context breadth are
    intentionally different.

    Here:
        candidate parents = 3
        final_k           = 2

    The runner must pass all candidates to the reranker but
    retain only the reranker's top two.
    """

    repository = make_repository()

    reranker = FakeReranker()

    runner = ParentRerankingRunner(
        parent_repository=repository,
        reranker=reranker,
        final_k=2,
    )

    result = runner.run(
        parent_run_result=(make_parent_run()),
    )

    assert len(reranker.received_candidates) == 3

    assert reranker.received_top_k == 2

    assert [item.item_id for item in result.results[0].items] == [
        "parent-key-3",
        "parent-key-1",
    ]

    assert [item.rank for item in result.results[0].items] == [
        1,
        2,
    ]
