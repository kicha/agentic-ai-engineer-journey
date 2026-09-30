from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest

from app.embeddings.base import (
    DenseEmbeddingService,
    DenseVector,
)
from app.models.parent_chunk import ParentChunk
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)
from benchmarks.runners.parent_mmr_runner import (
    ParentMMRRunner,
)

# ============================================================
# TEST DOUBLES
# ============================================================


class FakeParentRepository:
    """
    In-memory replacement for ParentRepository.

    The benchmark adapter must still perform the same logical
    lookup:

        parent_id -> authoritative ParentChunk

    without requiring a real MongoDB connection.
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


class FakeDenseEmbeddingService(DenseEmbeddingService):
    """
    Deterministic dense embedding service.

    It records query/document calls so the tests can prove that:

    - the query is embedded once,
    - all parent documents are embedded in one batch,
    - ParentMMRRunner does not perform unnecessary per-parent
      embedding calls.
    """

    def __init__(
        self,
        *,
        query_embedding: DenseVector,
        document_embeddings: dict[
            str,
            DenseVector,
        ],
    ) -> None:

        self.query_embedding = query_embedding

        self.document_embeddings = document_embeddings

        self.query_calls: list[str] = []

        self.document_calls: list[list[str]] = []

    @property
    def embedding_dimension(
        self,
    ) -> int:

        return len(self.query_embedding)

    def embed_query(
        self,
        query: str,
    ) -> DenseVector:

        self.query_calls.append(query)

        return list(self.query_embedding)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:

        text_list = list(texts)

        self.document_calls.append(text_list)

        return [list(self.document_embeddings[text]) for text in text_list]


class WrongCountEmbeddingService(FakeDenseEmbeddingService):
    """
    Simulate a broken embedding provider returning fewer vectors
    than requested.
    """

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[DenseVector]:

        embeddings = super().embed_documents(texts)

        return embeddings[:-1]


# ============================================================
# FIXTURES / BUILDERS
# ============================================================


def make_parent(
    *,
    parent_id: str,
    parent_key: str,
    content: str,
) -> ParentChunk:
    """
    Build only the ParentChunk fields needed by this unit test.

    Full ParentChunk validation belongs to the model tests.
    """

    return ParentChunk.model_construct(
        parent_id=parent_id,
        parent_key=parent_key,
        parent_index=0,
        content=content,
    )


def make_repository() -> FakeParentRepository:

    return FakeParentRepository(
        parents={
            "parent-1": make_parent(
                parent_id="parent-1",
                parent_key="parent-key-1",
                content="Parent one.",
            ),
            "parent-2": make_parent(
                parent_id="parent-2",
                parent_key="parent-key-2",
                content="Parent two.",
            ),
            "parent-3": make_parent(
                parent_id="parent-3",
                parent_key="parent-key-3",
                content="Parent three.",
            ),
            "parent-4": make_parent(
                parent_id="parent-4",
                parent_key="parent-key-4",
                content="Parent four.",
            ),
            "parent-5": make_parent(
                parent_id="parent-5",
                parent_key="parent-key-5",
                content="Parent five.",
            ),
        }
    )


def make_embedder() -> FakeDenseEmbeddingService:
    """
    Embeddings are deliberately chosen so MMR produces a
    deterministic result.

    Query:
        [1, 0, 0]

    Parent 1:
        highest dense relevance

    Parent 2:
        almost as relevant, but highly redundant with Parent 1

    Parent 3:
        slightly less relevant but more diverse

    With lambda=0.5, Parent 3 should be promoted ahead of
    Parent 2 after Parent 1 has been selected.
    """

    return FakeDenseEmbeddingService(
        query_embedding=[
            1.0,
            0.0,
            0.0,
        ],
        document_embeddings={
            "Parent one.": [
                0.95,
                0.312,
                0.0,
            ],
            "Parent two.": [
                0.94,
                0.341,
                0.0,
            ],
            "Parent three.": [
                0.80,
                0.0,
                0.60,
            ],
            "Parent four.": [
                0.70,
                0.0,
                0.714,
            ],
            "Parent five.": [
                0.20,
                0.98,
                0.0,
            ],
        },
    )


def make_reranked_run() -> BenchmarkRunResult:
    """
    Simulate the output produced by ParentRerankingRunner.

    Five parents are deliberately retained.

    Parent MMR will later reduce:

        5 reranked candidates
            ->
        final 3 parents
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query=("How does HNSW search efficiently?"),
        retrieval_level=(RetrievalLevel.PARENT),
        items=[
            RankedItem(
                item_id="parent-key-1",
                rank=1,
                score=0.95,
                metadata={
                    "parent_id": "parent-1",
                    "document_id": "DOC-001",
                    "pre_rerank_rank": 2,
                    "pre_rerank_score": 0.81,
                    "reranker_score": 0.95,
                },
            ),
            RankedItem(
                item_id="parent-key-2",
                rank=2,
                score=0.90,
                metadata={
                    "parent_id": "parent-2",
                    "document_id": "DOC-002",
                    "pre_rerank_rank": 1,
                    "pre_rerank_score": 0.85,
                    "reranker_score": 0.90,
                },
            ),
            RankedItem(
                item_id="parent-key-3",
                rank=3,
                score=0.80,
                metadata={
                    "parent_id": "parent-3",
                    "document_id": "DOC-003",
                    "pre_rerank_rank": 4,
                    "pre_rerank_score": 0.70,
                    "reranker_score": 0.80,
                },
            ),
            RankedItem(
                item_id="parent-key-4",
                rank=4,
                score=0.60,
                metadata={
                    "parent_id": "parent-4",
                    "document_id": "DOC-004",
                    "pre_rerank_rank": 3,
                    "pre_rerank_score": 0.74,
                    "reranker_score": 0.60,
                },
            ),
            RankedItem(
                item_id="parent-key-5",
                rank=5,
                score=0.30,
                metadata={
                    "parent_id": "parent-5",
                    "document_id": "DOC-005",
                    "pre_rerank_rank": 5,
                    "pre_rerank_score": 0.40,
                    "reranker_score": 0.30,
                },
            ),
        ],
        latency_ms=1000.0,
    )

    return BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=(RetrievalStrategy.HYBRID_RERANK),
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )


# ============================================================
# CONSTRUCTOR VALIDATION
# ============================================================


def test_parent_mmr_runner_rejects_invalid_lambda() -> None:
    """
    lambda_mult must remain within the mathematical MMR range:

        0 <= lambda <= 1
    """

    with pytest.raises(
        ValueError,
        match=("lambda_mult must be between 0 and 1"),
    ):
        ParentMMRRunner(
            parent_repository=(make_repository()),
            embedder=make_embedder(),
            lambda_mult=1.1,
            final_k=3,
        )


def test_parent_mmr_runner_rejects_non_positive_final_k() -> None:
    """
    A final context size of zero or less is invalid.
    """

    with pytest.raises(
        ValueError,
        match=("final_k must be greater than zero"),
    ):
        ParentMMRRunner(
            parent_repository=(make_repository()),
            embedder=make_embedder(),
            lambda_mult=0.7,
            final_k=0,
        )


# ============================================================
# INPUT LEVEL VALIDATION
# ============================================================


def test_parent_mmr_runner_requires_parent_level_run() -> None:
    """
    Canonical ordering requires Parent MMR to operate only
    after parent expansion/reranking.

    Child-level benchmark output must therefore be rejected.
    """

    child_result = QueryRankingResult(
        query_id="Q001",
        query="test query",
        retrieval_level=(RetrievalLevel.CHILD),
        items=[],
        latency_ms=10.0,
    )

    child_run = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=(RetrievalStrategy.HYBRID),
        retrieval_level=(RetrievalLevel.CHILD),
        results=[
            child_result,
        ],
    )

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=make_embedder(),
        lambda_mult=0.7,
        final_k=3,
    )

    with pytest.raises(
        ValueError,
        match=("ParentMMRRunner requires " "a parent-level BenchmarkRunResult"),
    ):
        runner.run(
            reranked_run_result=child_run,
        )


# ============================================================
# CANDIDATE COUNT
# ============================================================


def test_parent_mmr_runner_rejects_final_k_larger_than_candidate_pool() -> None:
    """
    MMR cannot select more parents than exist in the candidate
    pool.

    This also protects against accidentally feeding an already
    truncated reranker top-3 into MMR final_k=5.
    """

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=make_embedder(),
        lambda_mult=0.7,
        final_k=6,
    )

    with pytest.raises(
        ValueError,
        match=("final_k cannot exceed the number " "of available parent candidates"),
    ):
        runner.run(
            reranked_run_result=(make_reranked_run()),
        )


# ============================================================
# FULL MMR CONTRACT
# ============================================================


@patch("benchmarks.runners." "parent_mmr_runner." "perf_counter")
def test_parent_mmr_runner_selects_final_parents_and_preserves_identity(
    mock_perf_counter,
) -> None:
    """
    Prove the full Parent MMR benchmark adapter contract.

    This verifies:

    - authoritative parents are loaded from Mongo abstraction,
    - query is embedded once,
    - parents are embedded in one batch,
    - real ParentMMRSelector executes,
    - stable parent_key identities survive MMR,
    - MMR changes ranking based on relevance + redundancy,
    - reranker metadata survives,
    - MMR trace metadata is added,
    - benchmark version is preserved,
    - strategy becomes HYBRID_RERANK_MMR.
    """

    # Simulate 40 ms MMR-stage latency.
    mock_perf_counter.side_effect = [
        10.000,
        10.040,
    ]

    repository = make_repository()
    embedder = make_embedder()

    runner = ParentMMRRunner(
        parent_repository=repository,
        embedder=embedder,
        lambda_mult=0.5,
        final_k=3,
    )

    result = runner.run(
        reranked_run_result=(make_reranked_run()),
    )

    # --------------------------------------------------------
    # RUN IDENTITY
    # --------------------------------------------------------

    assert result.benchmark_version == "2.0"

    assert result.strategy == RetrievalStrategy.HYBRID_RERANK_MMR

    assert result.retrieval_level == RetrievalLevel.PARENT

    assert len(result.results) == 1

    # --------------------------------------------------------
    # AUTHORITATIVE PARENT LOOKUPS
    # --------------------------------------------------------

    assert repository.requested_parent_ids == [
        "parent-1",
        "parent-2",
        "parent-3",
        "parent-4",
        "parent-5",
    ]

    # --------------------------------------------------------
    # EMBEDDING CONTRACT
    # --------------------------------------------------------

    assert embedder.query_calls == ["How does HNSW search efficiently?"]

    # All five ParentChunks must be embedded in ONE batch.
    assert embedder.document_calls == [
        [
            "Parent one.",
            "Parent two.",
            "Parent three.",
            "Parent four.",
            "Parent five.",
        ]
    ]

    # --------------------------------------------------------
    # FINAL QUERY RESULT
    # --------------------------------------------------------

    query_result = result.results[0]

    assert query_result.retrieval_level == RetrievalLevel.PARENT

    assert len(query_result.items) == 3

    assert [item.rank for item in query_result.items] == [
        1,
        2,
        3,
    ]

    # --------------------------------------------------------
    # MMR BEHAVIOR
    #
    # Parent 1 has the strongest dense query similarity.
    #
    # Parent 2 is also highly relevant, but is extremely
    # redundant with Parent 1.
    #
    # With lambda=0.5, Parent 3 should therefore be promoted
    # ahead of Parent 2.
    # --------------------------------------------------------

    selected_ids = [item.item_id for item in query_result.items]

    assert selected_ids[0] == ("parent-key-1")

    assert selected_ids[1] == ("parent-key-3")

    assert selected_ids[2] == ("parent-key-2")

    # --------------------------------------------------------
    # STABLE IDENTITIES
    # --------------------------------------------------------

    assert set(selected_ids).issubset(
        {
            "parent-key-1",
            "parent-key-2",
            "parent-key-3",
            "parent-key-4",
            "parent-key-5",
        }
    )

    # --------------------------------------------------------
    # EXISTING RERANKER METADATA SURVIVES
    # --------------------------------------------------------

    first_item = query_result.items[0]

    assert first_item.metadata["parent_id"] == "parent-1"

    assert first_item.metadata["document_id"] == "DOC-001"

    assert first_item.metadata["pre_rerank_rank"] == 2

    assert first_item.metadata["reranker_score"] == pytest.approx(0.95)

    # --------------------------------------------------------
    # MMR TRACE METADATA
    # --------------------------------------------------------

    assert first_item.metadata["pre_mmr_rank"] == 1

    assert first_item.metadata["query_similarity"] > 0.0

    assert "max_redundancy" in first_item.metadata

    assert "relevance_contribution" in first_item.metadata

    assert "redundancy_penalty" in first_item.metadata

    assert "mmr_score" in first_item.metadata


# ============================================================
# LATENCY
# ============================================================


@patch("benchmarks.runners." "parent_mmr_runner." "perf_counter")
def test_parent_mmr_runner_adds_mmr_latency_to_existing_pipeline_latency(
    mock_perf_counter,
) -> None:
    """
    Existing reranked latency:
        1000 ms

    Simulated Parent MMR:
          50 ms

    Expected cumulative pipeline latency:
        1050 ms
    """

    mock_perf_counter.side_effect = [
        20.000,
        20.050,
    ]

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=make_embedder(),
        lambda_mult=0.5,
        final_k=3,
    )

    result = runner.run(
        reranked_run_result=(make_reranked_run()),
    )

    assert result.results[0].latency_ms == pytest.approx(
        1050.0,
        abs=0.01,
    )


# ============================================================
# MISSING PARENT ID
# ============================================================


def test_parent_mmr_runner_rejects_missing_parent_id() -> None:
    """
    Parent MMR cannot rehydrate authoritative ParentChunks when
    transport metadata has lost parent_id.
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
        latency_ms=100.0,
    )

    run_result = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=(RetrievalStrategy.HYBRID_RERANK),
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=make_embedder(),
        lambda_mult=0.7,
        final_k=1,
    )

    with pytest.raises(
        RuntimeError,
        match=("Parent benchmark item is missing " "parent_id metadata"),
    ):
        runner.run(
            reranked_run_result=(run_result),
        )


# ============================================================
# DANGLING PARENT
# ============================================================


def test_parent_mmr_runner_rejects_dangling_parent_reference() -> None:
    """
    Silently dropping a missing Mongo parent would change the
    candidate set and invalidate benchmark results.

    Therefore persistence corruption is a hard failure.
    """

    query_result = QueryRankingResult(
        query_id="Q001",
        query="test query",
        retrieval_level=(RetrievalLevel.PARENT),
        items=[
            RankedItem(
                item_id=("missing-parent-key"),
                rank=1,
                score=0.90,
                metadata={
                    "parent_id": ("missing-parent"),
                },
            ),
        ],
        latency_ms=100.0,
    )

    run_result = BenchmarkRunResult(
        benchmark_version="2.0",
        strategy=(RetrievalStrategy.HYBRID_RERANK),
        retrieval_level=(RetrievalLevel.PARENT),
        results=[
            query_result,
        ],
    )

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=make_embedder(),
        lambda_mult=0.7,
        final_k=1,
    )

    with pytest.raises(
        RuntimeError,
        match=("Dangling parent reference " "detected during Parent MMR"),
    ):
        runner.run(
            reranked_run_result=(run_result),
        )


# ============================================================
# EMBEDDING COUNT INTEGRITY
# ============================================================


def test_parent_mmr_runner_rejects_wrong_parent_embedding_count() -> None:
    """
    One dense vector must exist for every ParentChunk.

    If the provider returns fewer vectors, zip() would otherwise
    silently truncate the candidate set and corrupt evaluation.
    """

    normal_embedder = make_embedder()

    broken_embedder = WrongCountEmbeddingService(
        query_embedding=(normal_embedder.query_embedding),
        document_embeddings=(normal_embedder.document_embeddings),
    )

    runner = ParentMMRRunner(
        parent_repository=(make_repository()),
        embedder=broken_embedder,
        lambda_mult=0.7,
        final_k=3,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Embedding service returned an " "unexpected number of parent embeddings"
        ),
    ):
        runner.run(
            reranked_run_result=(make_reranked_run()),
        )
