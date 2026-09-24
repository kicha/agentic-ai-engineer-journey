from benchmarks.evaluation.gold_models import (
    Answerability,
    AnnotationStatus,
    ChildJudgment,
    DocumentJudgment,
    GoldQueryV2,
    ParentJudgment,
    RelevanceGrade,
)


def build_gold_query() -> GoldQueryV2:
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


def test_document_relevance_map() -> None:
    query = build_gold_query()

    assert query.document_relevance_map() == {
        "DOC-002": 3,
    }


def test_parent_relevance_map() -> None:
    query = build_gold_query()

    assert query.parent_relevance_map() == {
        "PARENT-001": 3,
        "PARENT-002": 1,
    }


def test_child_relevance_map() -> None:
    query = build_gold_query()

    assert query.child_relevance_map() == {
        "CHILD-001": 3,
        "CHILD-002": 2,
    }


def test_generic_relevance_map_returns_parent_relevance() -> None:
    query = build_gold_query()

    assert query.relevance_map("parent") == {
        "PARENT-001": 3,
        "PARENT-002": 1,
    }


def test_unanswerable_query_returns_empty_relevance_maps() -> None:
    query = GoldQueryV2(
        query_id="Q998",
        query="Unsupported question",
        query_class="unanswerable",
        answerability=Answerability.UNANSWERABLE,
        annotation_status=AnnotationStatus.CURATED,
        document_judgments=[],
        parent_judgments=[],
        child_judgments=[],
    )

    assert query.document_relevance_map() == {}
    assert query.parent_relevance_map() == {}
    assert query.child_relevance_map() == {}
