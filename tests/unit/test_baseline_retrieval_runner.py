from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from qdrant_client.models import ScoredPoint

from app.retrieval.hybrid_child_retriever import (
    HybridChildRetriever,
)
from benchmarks.evaluation.benchmark_models import (
    RetrievalStrategy,
)
from benchmarks.evaluation.gold_models import (
    AnnotationStatus,
    Answerability,
    GoldDatasetV2,
    GoldQueryV2,
)
from benchmarks.evaluation.ranking_evaluator import (
    RetrievalLevel,
)
from benchmarks.runners.baseline_retrieval_runner import (
    BaselineRetrievalRunner,
)

# ============================================================
# TEST DATA BUILDERS
# ============================================================


def build_gold_dataset() -> GoldDatasetV2:
    """
    Purpose:
    Build a minimal in-memory Benchmark V2 dataset for runner tests.

    Why this matters:
    Unit tests for the runner should not depend on the real frozen JSON
    files. We want to isolate runner orchestration from filesystem,
    Qdrant, MongoDB, and embedding dependencies.
    """

    return GoldDatasetV2(
        schema_version="2.0",
        benchmark_version="2.0",
        corpus_sha256="test-sha",
        status="frozen",
        queries=[
            GoldQueryV2(
                query_id="Q001",
                query="What does ef_construct control?",
                query_class="definition",
                answerability=Answerability.ANSWERABLE,
                annotation_status=AnnotationStatus.CURATED,
                document_judgments=[],
                parent_judgments=[],
                child_judgments=[],
            ),
            GoldQueryV2(
                query_id="Q002",
                query="What does M control?",
                query_class="definition",
                answerability=Answerability.ANSWERABLE,
                annotation_status=AnnotationStatus.CURATED,
                document_judgments=[],
                parent_judgments=[],
                child_judgments=[],
            ),
        ],
    )


def make_scored_point(
    child_key: str,
    score: float,
) -> ScoredPoint:
    """
    Purpose:
    Construct the smallest realistic Qdrant ScoredPoint needed by
    BaselineRetrievalRunner.

    We use a real value object here rather than mocking ScoredPoint
    itself because it is just deterministic data, not an external
    service dependency.
    """

    return ScoredPoint(
        id=child_key,
        version=0,
        score=score,
        payload={
            "child_key": child_key,
        },
        vector=None,
    )


@pytest.fixture
def retriever() -> Mock:
    """
    Purpose:
    Mock the external retrieval boundary.

    Why this matters:
    A UNIT test must not require Qdrant, embedding models, or sparse
    encoders to be available. Those belong in integration tests.
    """

    return Mock(spec=HybridChildRetriever)


@pytest.fixture
def runner(
    retriever: Mock,
) -> BaselineRetrievalRunner:
    """
    Purpose:
    Build the unit under test with an injected mocked retriever.
    """

    return BaselineRetrievalRunner(
        retriever=retriever,
        child_candidate_limit=3,
    )


# ============================================================
# DENSE STRATEGY
# ============================================================


def test_dense_strategy_uses_dense_retriever(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify DENSE benchmark execution delegates only to
    HybridChildRetriever.retrieve_dense().

    Regression protected:
    Accidentally routing a Dense benchmark through Hybrid or Sparse
    retrieval after future runner refactoring.
    """

    retriever.retrieve_dense.return_value = [
        make_scored_point(
            child_key="CHILD-001",
            score=0.91,
        )
    ]

    dataset = build_gold_dataset()

    result = runner.run(
        gold_dataset=dataset,
        strategy=RetrievalStrategy.DENSE,
    )

    # There are two gold queries, therefore Dense retrieval must have
    # been invoked exactly twice.
    assert retriever.retrieve_dense.call_count == 2

    retriever.retrieve_sparse.assert_not_called()
    retriever.retrieve_hybrid.assert_not_called()

    assert result.strategy == RetrievalStrategy.DENSE

    assert result.retrieval_level == RetrievalLevel.CHILD


# ============================================================
# SPARSE STRATEGY
# ============================================================


def test_sparse_strategy_uses_sparse_retriever(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify SPARSE benchmarks call the sparse retrieval path only.

    This protects the baseline comparison between independently
    measured Dense, Sparse, and Hybrid retrieval.
    """

    retriever.retrieve_sparse.return_value = []

    result = runner.run(
        gold_dataset=build_gold_dataset(),
        strategy=RetrievalStrategy.SPARSE,
    )

    assert retriever.retrieve_sparse.call_count == 2
    retriever.retrieve_dense.assert_not_called()
    retriever.retrieve_hybrid.assert_not_called()

    assert result.strategy == RetrievalStrategy.SPARSE


# ============================================================
# HYBRID STRATEGY
# ============================================================


def test_hybrid_strategy_uses_hybrid_retriever(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify HYBRID benchmark execution delegates to the RRF Hybrid
    retrieval path rather than Dense or Sparse independently.
    """

    retriever.retrieve_hybrid.return_value = []

    result = runner.run(
        gold_dataset=build_gold_dataset(),
        strategy=RetrievalStrategy.HYBRID,
    )

    assert retriever.retrieve_hybrid.call_count == 2
    retriever.retrieve_dense.assert_not_called()
    retriever.retrieve_sparse.assert_not_called()

    assert result.strategy == RetrievalStrategy.HYBRID


# ============================================================
# STANDARDIZED OUTPUT
# ============================================================


def test_runner_converts_qdrant_hits_to_ranked_child_keys(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify Qdrant results are converted into deterministic child_key
    rankings using the common BenchmarkRunResult contract.

    Why this matters:
    Benchmark V2 gold references child_key, not volatile child UUIDs.
    Using the wrong identifier would make relevance evaluation invalid.
    """

    retriever.retrieve_dense.return_value = [
        make_scored_point(
            child_key="CHILD-A",
            score=0.90,
        ),
        make_scored_point(
            child_key="CHILD-B",
            score=0.80,
        ),
    ]

    result = runner.run(
        gold_dataset=build_gold_dataset(),
        strategy=RetrievalStrategy.DENSE,
    )

    first_query_result = result.results[0]

    assert first_query_result.ranked_ids() == [
        "CHILD-A",
        "CHILD-B",
    ]

    # Ranking positions are created from retrieval order and must use
    # normal IR one-based ranking semantics.
    assert [item.rank for item in first_query_result.items] == [
        1,
        2,
    ]

    # Preserve retrieval scores for diagnostics even though centralized
    # relevance metrics operate on ranked stable IDs.
    assert first_query_result.items[0].score == pytest.approx(0.90)


def test_runner_passes_candidate_limit_to_retriever(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify the configured candidate limit is forwarded consistently to
    the retrieval implementation.

    Why this matters:
    Comparing methods with accidental candidate-pool differences can
    make benchmark results misleading.
    """

    retriever.retrieve_dense.return_value = []

    runner.run(
        gold_dataset=build_gold_dataset(),
        strategy=RetrievalStrategy.DENSE,
    )

    first_call = retriever.retrieve_dense.call_args_list[0]

    assert first_call.kwargs["limit"] == 3


def test_runner_keeps_empty_retrieval_result(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Verify zero-result retrieval is represented as an empty ranking
    rather than treated as an exception.

    The centralized metric layer will correctly convert an empty
    ranking into zero relevance metrics.
    """

    retriever.retrieve_dense.return_value = []

    result = runner.run(
        gold_dataset=build_gold_dataset(),
        strategy=RetrievalStrategy.DENSE,
    )

    assert result.results[0].items == []


# ============================================================
# PAYLOAD VALIDATION
# ============================================================


def test_runner_rejects_hit_without_child_key(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Fail immediately if a Qdrant benchmark payload does not contain
    the deterministic child_key required by Benchmark V2.

    Regression protected:
    Accidentally evaluating transient child_id values or malformed
    legacy Qdrant payloads against V2 gold.
    """

    retriever.retrieve_dense.return_value = [
        ScoredPoint(
            id="temporary-id",
            version=0,
            score=0.90,
            payload={
                "child_id": "temporary-id",
            },
            vector=None,
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="does not contain child_key",
    ):
        runner.run(
            gold_dataset=build_gold_dataset(),
            strategy=RetrievalStrategy.DENSE,
        )


def test_runner_rejects_duplicate_child_keys(
    runner: BaselineRetrievalRunner,
    retriever: Mock,
) -> None:
    """
    Purpose:
    Protect ranking metrics from duplicate retrieval units.

    Why this matters:
    A repeated child could incorrectly occupy multiple ranking
    positions and distort Precision, Recall, MAP, and nDCG.
    """

    retriever.retrieve_dense.return_value = [
        make_scored_point(
            child_key="CHILD-001",
            score=0.90,
        ),
        make_scored_point(
            child_key="CHILD-001",
            score=0.85,
        ),
    ]

    with pytest.raises(
        RuntimeError,
        match="Duplicate child_key",
    ):
        runner.run(
            gold_dataset=build_gold_dataset(),
            strategy=RetrievalStrategy.DENSE,
        )


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================


def test_runner_rejects_non_positive_candidate_limit(
    retriever: Mock,
) -> None:
    """
    Purpose:
    Prevent meaningless retrieval configurations from entering the
    benchmark.

    A candidate limit of zero or less cannot produce a valid ranking.
    """

    with pytest.raises(
        ValueError,
        match="child_candidate_limit must be greater than 0",
    ):
        BaselineRetrievalRunner(
            retriever=retriever,
            child_candidate_limit=0,
        )
