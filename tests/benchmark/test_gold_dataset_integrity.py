import json

from benchmarks.evaluation.gold_models import (
    AnnotationStatus,
    Answerability,
    GoldDatasetV2,
)
from benchmarks.paths import (
    GOLD_QUERIES_PATH,
)


def load_gold_dataset() -> GoldDatasetV2:
    with GOLD_QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return GoldDatasetV2.model_validate(data)


def test_gold_dataset_contains_32_queries() -> None:
    dataset = load_gold_dataset()

    assert len(dataset.queries) == 32


def test_all_gold_queries_are_curated() -> None:
    dataset = load_gold_dataset()

    for query in dataset.queries:
        assert query.annotation_status == AnnotationStatus.CURATED


def test_query_ids_are_unique() -> None:
    dataset = load_gold_dataset()

    query_ids = [query.query_id for query in dataset.queries]

    assert len(query_ids) == len(set(query_ids))


def test_answerable_queries_have_relevance_judgments() -> None:
    dataset = load_gold_dataset()

    for query in dataset.queries:
        if query.answerability != Answerability.ANSWERABLE:
            continue

        assert query.parent_judgments
        assert query.child_judgments


def test_unanswerable_queries_have_no_relevance_judgments() -> None:
    dataset = load_gold_dataset()

    for query in dataset.queries:
        if query.answerability != Answerability.UNANSWERABLE:
            continue

        assert not query.document_judgments
        assert not query.parent_judgments
        assert not query.child_judgments
