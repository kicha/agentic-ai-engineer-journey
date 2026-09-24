import pytest

from benchmarks.evaluation.gold_models import (
    AnnotationStatus,
    Answerability,
    ChildJudgment,
    DocumentJudgment,
    GoldQueryV2,
    ParentJudgment,
    RelevanceGrade,
)
from benchmarks.evaluation.ranking_evaluator import (
    RankingEvaluator,
    RetrievalLevel,
)

# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def gold_query() -> GoldQueryV2:
    return GoldQueryV2(
        query_id="Q999",
        query="What does ef_construct control?",
        query_class="definition",
        answerability=Answerability.ANSWERABLE,
        annotation_status=AnnotationStatus.CURATED,
        document_judgments=[
            DocumentJudgment(
                document_id="DOC-002",
                relevance=RelevanceGrade.DIRECT,
            ),
        ],
        parent_judgments=[
            ParentJudgment(
                parent_key="PARENT-001",
                relevance=RelevanceGrade.DIRECT,
            ),
            ParentJudgment(
                parent_key="PARENT-002",
                relevance=RelevanceGrade.SUPPORTING,
            ),
            ParentJudgment(
                parent_key="PARENT-003",
                relevance=RelevanceGrade.WEAK,
            ),
        ],
        child_judgments=[
            ChildJudgment(
                child_key="CHILD-001",
                relevance=RelevanceGrade.DIRECT,
            ),
            ChildJudgment(
                child_key="CHILD-002",
                relevance=RelevanceGrade.SUPPORTING,
            ),
        ],
    )


@pytest.fixture
def evaluator() -> RankingEvaluator:
    return RankingEvaluator()


# ============================================================
# PARENT EVALUATION
# ============================================================


def test_evaluate_parent_ranking(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "PARENT-001",
            "PARENT-X",
            "PARENT-002",
        ],
        retrieval_level=RetrievalLevel.PARENT,
        k=3,
    )

    assert result.query_id == "Q999"
    assert result.retrieval_level == RetrievalLevel.PARENT

    assert result.ranked_ids == [
        "PARENT-001",
        "PARENT-X",
        "PARENT-002",
    ]

    assert result.relevance == {
        "PARENT-001": 3,
        "PARENT-002": 2,
        "PARENT-003": 1,
    }

    assert result.metrics.hit_at_k == 1.0

    assert result.metrics.precision_at_k == pytest.approx(2 / 3)

    assert result.metrics.recall_at_k == pytest.approx(2 / 3)

    assert result.metrics.reciprocal_rank == 1.0


# ============================================================
# CHILD EVALUATION
# ============================================================


def test_evaluate_child_ranking(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "CHILD-X",
            "CHILD-002",
        ],
        retrieval_level=RetrievalLevel.CHILD,
        k=2,
    )

    assert result.relevance == {
        "CHILD-001": 3,
        "CHILD-002": 2,
    }

    assert result.metrics.hit_at_k == 1.0

    assert result.metrics.precision_at_k == pytest.approx(0.5)

    assert result.metrics.recall_at_k == pytest.approx(0.5)

    assert result.metrics.reciprocal_rank == pytest.approx(0.5)


# ============================================================
# DOCUMENT EVALUATION
# ============================================================


def test_evaluate_document_ranking(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "DOC-001",
            "DOC-002",
        ],
        retrieval_level=RetrievalLevel.DOCUMENT,
        k=2,
    )

    assert result.relevance == {
        "DOC-002": 3,
    }

    assert result.metrics.hit_at_k == 1.0

    assert result.metrics.reciprocal_rank == pytest.approx(0.5)


# ============================================================
# TOP-K BOUNDARY
# ============================================================


def test_evaluator_truncates_ranked_ids_to_k(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "PARENT-X",
            "PARENT-Y",
            "PARENT-001",
        ],
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    assert result.ranked_ids == [
        "PARENT-X",
        "PARENT-Y",
    ]

    assert result.metrics.hit_at_k == 0.0

    assert result.metrics.reciprocal_rank == 0.0


def test_relevant_item_below_k_does_not_affect_metrics(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "PARENT-X",
            "PARENT-Y",
            "PARENT-001",
        ],
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    assert result.metrics.hit_at_k == 0.0
    assert result.metrics.recall_at_k == 0.0
    assert result.metrics.reciprocal_rank == 0.0


# ============================================================
# UNANSWERABLE QUERY
# ============================================================


def test_unanswerable_query_returns_zero_metrics(
    evaluator: RankingEvaluator,
) -> None:

    gold_query = GoldQueryV2(
        query_id="Q998",
        query="Unsupported question",
        query_class="unanswerable",
        answerability=Answerability.UNANSWERABLE,
        annotation_status=AnnotationStatus.CURATED,
        document_judgments=[],
        parent_judgments=[],
        child_judgments=[],
    )

    result = evaluator.evaluate(
        gold_query=gold_query,
        ranked_ids=[
            "PARENT-001",
            "PARENT-002",
        ],
        retrieval_level=RetrievalLevel.PARENT,
        k=2,
    )

    assert result.relevance == {}
    assert result.metrics.hit_at_k == 0.0
    assert result.metrics.precision_at_k == 0.0
    assert result.metrics.recall_at_k == 0.0
    assert result.metrics.reciprocal_rank == 0.0
    assert result.metrics.average_precision_at_k == 0.0
    assert result.metrics.ndcg_at_k == 0.0


# ============================================================
# VALIDATION
# ============================================================


def test_evaluator_rejects_duplicate_ranked_ids(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
) -> None:

    with pytest.raises(
        ValueError,
        match="ranked_ids must not contain duplicate",
    ):
        evaluator.evaluate(
            gold_query=gold_query,
            ranked_ids=[
                "PARENT-001",
                "PARENT-001",
            ],
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
def test_evaluator_rejects_invalid_k(
    evaluator: RankingEvaluator,
    gold_query: GoldQueryV2,
    k: int,
) -> None:

    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        evaluator.evaluate(
            gold_query=gold_query,
            ranked_ids=[],
            retrieval_level=RetrievalLevel.PARENT,
            k=k,
        )
