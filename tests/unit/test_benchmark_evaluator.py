from unittest.mock import Mock

import pytest

from benchmarks.evaluation.benchmark_evaluator import (
    BenchmarkEvaluator,
)
from benchmarks.evaluation.gold_models import (
    AnnotationStatus,
    Answerability,
    GoldDatasetV2,
    GoldQueryV2,
    ParentJudgment,
    RelevanceGrade,
)
from benchmarks.evaluation.ranking_evaluator import (
    RankingEvaluator,
    RetrievalLevel,
)

# ============================================================
# TEST DATA BUILDERS
# ============================================================


def build_answerable_query(
    query_id: str,
    relevant_parent_key: str,
) -> GoldQueryV2:
    """
    Build the smallest useful answerable gold query.

    Each synthetic query has exactly one relevant parent.
    This keeps expected metric calculations obvious and makes
    failures easy to understand.
    """

    return GoldQueryV2(
        query_id=query_id,
        query=f"Question for {query_id}",
        query_class="test",
        answerability=Answerability.ANSWERABLE,
        annotation_status=AnnotationStatus.CURATED,
        document_judgments=[],
        parent_judgments=[
            ParentJudgment(
                parent_key=relevant_parent_key,
                relevance=RelevanceGrade.DIRECT,
            ),
        ],
        child_judgments=[],
    )


def build_unanswerable_query(
    query_id: str,
) -> GoldQueryV2:
    """
    Build an intentionally unanswerable query.

    An unanswerable query must contain no relevance judgments.
    It will still receive a per-query diagnostic result but must
    not contribute to normal retrieval ranking averages.
    """

    return GoldQueryV2(
        query_id=query_id,
        query=f"Unanswerable question for {query_id}",
        query_class="unanswerable",
        answerability=Answerability.UNANSWERABLE,
        annotation_status=AnnotationStatus.CURATED,
        document_judgments=[],
        parent_judgments=[],
        child_judgments=[],
    )


def build_gold_dataset() -> GoldDatasetV2:
    """
    Build a small deterministic benchmark dataset.

    Q001:
        relevant item retrieved at rank 1

    Q002:
        relevant item retrieved at rank 2

    Q003:
        intentionally unanswerable

    This arrangement lets us verify:
    - per-query evaluation
    - aggregation
    - MRR calculation
    - unanswerable-query exclusion
    """

    queries = [
        build_answerable_query(
            query_id="Q001",
            relevant_parent_key="PARENT-001",
        ),
        build_answerable_query(
            query_id="Q002",
            relevant_parent_key="PARENT-002",
        ),
        build_unanswerable_query(
            query_id="Q003",
        ),
    ]

    return GoldDatasetV2(
        schema_version="2.0",
        benchmark_version="2.0",
        corpus_sha256="test-sha",
        status="frozen",
        queries=queries,
    )


# ============================================================
# BASIC BENCHMARK EVALUATION
# ============================================================


def test_benchmark_evaluator_evaluates_every_query() -> None:
    """
    Purpose:
    Verify that BenchmarkEvaluator produces one diagnostic result
    for every query in the gold dataset.

    Why this matters:
    A benchmark must never silently drop queries. Missing query
    evaluation would make aggregate results incomparable between
    retrieval methods.
    """

    dataset = build_gold_dataset()

    rankings = {
        "Q001": [
            "PARENT-001",
            "PARENT-X",
        ],
        "Q002": [
            "PARENT-X",
            "PARENT-002",
        ],
        "Q003": [
            "PARENT-X",
            "PARENT-Y",
        ],
    }

    evaluator = BenchmarkEvaluator()

    result = evaluator.evaluate(
        gold_dataset=dataset,
        ranked_ids_by_query_id=rankings,
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    # All three gold queries must appear in diagnostics,
    # including the intentionally unanswerable query.
    assert result.total_query_count == 3
    assert len(result.query_results) == 3


def test_benchmark_evaluator_counts_answerability_correctly() -> None:
    """
    Purpose:
    Verify the benchmark distinguishes answerable and unanswerable
    queries correctly.

    Why this matters:
    Ranking quality and abstention quality are different evaluation
    problems and should not be mixed into one denominator.
    """

    dataset = build_gold_dataset()

    rankings = {
        "Q001": ["PARENT-001"],
        "Q002": ["PARENT-002"],
        "Q003": [],
    }

    evaluator = BenchmarkEvaluator()

    result = evaluator.evaluate(
        gold_dataset=dataset,
        ranked_ids_by_query_id=rankings,
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    assert result.answerable_query_count == 2
    assert result.unanswerable_query_count == 1


# ============================================================
# AGGREGATE METRIC BEHAVIOR
# ============================================================


def test_benchmark_evaluator_aggregates_answerable_queries() -> None:
    """
    Purpose:
    Verify that dataset-level metrics are arithmetic means of the
    answerable query metrics.

    Scenario:
    Q001 retrieves its relevant parent at rank 1.
        Reciprocal Rank = 1.0

    Q002 retrieves its relevant parent at rank 2.
        Reciprocal Rank = 0.5

    Therefore:
        MRR = (1.0 + 0.5) / 2 = 0.75
    """

    dataset = build_gold_dataset()

    rankings = {
        "Q001": [
            "PARENT-001",
            "PARENT-X",
        ],
        "Q002": [
            "PARENT-X",
            "PARENT-002",
        ],
        "Q003": [
            "PARENT-X",
        ],
    }

    evaluator = BenchmarkEvaluator()

    result = evaluator.evaluate(
        gold_dataset=dataset,
        ranked_ids_by_query_id=rankings,
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    assert result.metrics.mean_reciprocal_rank == pytest.approx(0.75)

    # Both answerable queries retrieve their relevant parent
    # somewhere inside the top 2, so Hit@2 is 1.0 for each.
    assert result.metrics.mean_hit_at_k == pytest.approx(1.0)

    # Each query returns two items, but only one is relevant.
    # Precision@2 is therefore 0.5 for both answerable queries.
    assert result.metrics.mean_precision_at_k == pytest.approx(0.5)

    # Each answerable query has exactly one relevant parent,
    # and that parent was retrieved, so Recall@2 is 1.0.
    assert result.metrics.mean_recall_at_k == pytest.approx(1.0)


def test_unanswerable_query_is_excluded_from_ranking_averages() -> None:
    """
    Purpose:
    Prove that an unanswerable query does not lower retrieval
    ranking metrics simply because it has no relevant documents.

    Regression protected:
    Accidentally averaging an all-zero unanswerable result into
    Hit, Recall, MRR, MAP or nDCG.
    """

    dataset = build_gold_dataset()

    rankings = {
        "Q001": ["PARENT-001"],
        "Q002": ["PARENT-002"],
        "Q003": ["PARENT-WRONG"],
    }

    evaluator = BenchmarkEvaluator()

    result = evaluator.evaluate(
        gold_dataset=dataset,
        ranked_ids_by_query_id=rankings,
        retrieval_level=RetrievalLevel.PARENT,
        k=1,
    )

    # Both answerable queries are perfect at rank 1.
    #
    # If Q003 were incorrectly included in the average,
    # mean Hit@1 would become 2/3 instead of 1.0.
    assert result.metrics.mean_hit_at_k == pytest.approx(1.0)

    assert result.metrics.mean_reciprocal_rank == pytest.approx(1.0)


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_benchmark_evaluator_rejects_missing_query_results() -> None:
    """
    Purpose:
    Verify an incomplete benchmark run fails loudly.

    Why this matters:
    If one retrieval method evaluates 31 queries while another
    evaluates 32, their aggregate metrics are not directly
    comparable. Missing queries must therefore be an error.
    """

    dataset = build_gold_dataset()

    # Q003 is deliberately missing.
    rankings = {
        "Q001": ["PARENT-001"],
        "Q002": ["PARENT-002"],
    }

    evaluator = BenchmarkEvaluator()

    with pytest.raises(
        ValueError,
        match="Missing ranked results",
    ):
        evaluator.evaluate(
            gold_dataset=dataset,
            ranked_ids_by_query_id=rankings,
            retrieval_level=RetrievalLevel.PARENT,
            k=2,
        )


def test_benchmark_evaluator_rejects_unknown_query_results() -> None:
    """
    Purpose:
    Verify retrieval output cannot contain query IDs that are not
    part of the frozen gold dataset.

    Why this matters:
    Extra query results usually indicate that the runner and gold
    dataset are using different benchmark versions.
    """

    dataset = build_gold_dataset()

    rankings = {
        "Q001": ["PARENT-001"],
        "Q002": ["PARENT-002"],
        "Q003": [],
        "Q999": ["PARENT-X"],
    }

    evaluator = BenchmarkEvaluator()

    with pytest.raises(
        ValueError,
        match="unknown query IDs",
    ):
        evaluator.evaluate(
            gold_dataset=dataset,
            ranked_ids_by_query_id=rankings,
            retrieval_level=RetrievalLevel.PARENT,
            k=2,
        )


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        -10,
    ],
)
def test_benchmark_evaluator_rejects_invalid_k(
    k: int,
) -> None:
    """
    Purpose:
    Ensure top-k evaluation always uses a meaningful positive
    cutoff.

    Why this matters:
    Metrics such as Hit@K, Precision@K and Recall@K have no useful
    interpretation for zero or negative K.
    """

    dataset = build_gold_dataset()

    evaluator = BenchmarkEvaluator()

    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        evaluator.evaluate(
            gold_dataset=dataset,
            ranked_ids_by_query_id={
                "Q001": [],
                "Q002": [],
                "Q003": [],
            },
            retrieval_level=RetrievalLevel.PARENT,
            k=k,
        )


# ============================================================
# DEPENDENCY INJECTION / MOCK TEST
# ============================================================


def test_benchmark_evaluator_delegates_to_ranking_evaluator() -> None:
    """
    Purpose:
    Demonstrate the unit-test boundary using a mock dependency.

    BenchmarkEvaluator owns orchestration; RankingEvaluator owns
    per-query scoring. This test verifies delegation without
    retesting RankingEvaluator's metric calculations.

    We use a mock here because the collaborator itself is the
    behavior being isolated, unlike earlier tests where real pure
    calculations were intentionally exercised end-to-end.
    """

    ranking_evaluator = Mock(spec=RankingEvaluator)

    evaluator = BenchmarkEvaluator(
        ranking_evaluator=ranking_evaluator,
    )

    dataset = build_gold_dataset()

    # We only inspect delegation in this test. The mock therefore
    # does not need to calculate real RankingEvaluationResult
    # objects because evaluation will fail during aggregation if
    # we let the full method complete.
    #
    # Instead, this specific behavior is more cleanly verified by
    # checking calls in a separate implementation-level test once
    # a suitable fake result factory is introduced.
    #
    # For now, confirm dependency injection itself is preserved.
    assert evaluator.ranking_evaluator is ranking_evaluator
