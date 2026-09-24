# tests/unit/test_ranking_metrics.py

import math

import pytest

from benchmarks.evaluation.ranking_metrics import (
    RankingMetrics,
)


def test_hit_at_k_returns_one_when_relevant_item_is_present() -> None:
    ranked_ids = [
        "P003",
        "P007",
        "P001",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
    }

    result = RankingMetrics.hit_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result == 1.0


def test_hit_at_k_returns_zero_when_no_relevant_item_is_present() -> None:
    ranked_ids = [
        "P003",
        "P007",
        "P008",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
    }

    result = RankingMetrics.hit_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result == 0.0


def test_precision_at_k() -> None:
    ranked_ids = [
        "P001",
        "P003",
        "P009",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
    }

    result = RankingMetrics.precision_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result == pytest.approx(2 / 3)


def test_recall_at_k() -> None:
    ranked_ids = [
        "P001",
        "P003",
        "P009",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
        "P011": 1,
    }

    result = RankingMetrics.recall_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result == pytest.approx(2 / 3)


def test_reciprocal_rank_when_first_relevant_is_rank_two() -> None:
    ranked_ids = [
        "P003",
        "P001",
        "P009",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
    }

    result = RankingMetrics.reciprocal_rank(
        ranked_ids=ranked_ids,
        relevance=relevance,
    )

    assert result == pytest.approx(0.5)


def test_reciprocal_rank_returns_zero_when_no_relevant_item_exists() -> None:
    ranked_ids = [
        "P003",
        "P004",
    ]

    relevance = {
        "P001": 3,
    }

    result = RankingMetrics.reciprocal_rank(
        ranked_ids=ranked_ids,
        relevance=relevance,
    )

    assert result == 0.0


def test_average_precision_at_k() -> None:
    ranked_ids = [
        "P001",
        "P003",
        "P009",
        "P011",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
        "P011": 1,
    }

    result = RankingMetrics.average_precision_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=4,
    )

    expected = (1.0 + (2 / 3) + (3 / 4)) / 3

    assert result == pytest.approx(expected)


def test_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    ranked_ids = [
        "P001",
        "P009",
        "P011",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
        "P011": 1,
    }

    result = RankingMetrics.ndcg_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_at_k_penalizes_weaker_ordering() -> None:
    ranked_ids = [
        "P011",
        "P009",
        "P001",
    ]

    relevance = {
        "P001": 3,
        "P009": 2,
        "P011": 1,
    }

    result = RankingMetrics.ndcg_at_k(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    ideal_dcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)

    actual_dcg = 1 / math.log2(2) + 3 / math.log2(3) + 7 / math.log2(4)

    expected = actual_dcg / ideal_dcg

    assert result == pytest.approx(expected)


def test_metrics_return_zero_for_empty_ranked_results() -> None:
    ranked_ids: list[str] = []

    relevance = {
        "P001": 3,
    }

    result = RankingMetrics.evaluate(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=3,
    )

    assert result.hit_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.average_precision_at_k == 0.0
    assert result.ndcg_at_k == 0.0


def test_metrics_return_zero_for_unanswerable_query() -> None:
    ranked_ids = [
        "P001",
        "P002",
    ]

    relevance: dict[str, int] = {}

    result = RankingMetrics.evaluate(
        ranked_ids=ranked_ids,
        relevance=relevance,
        k=2,
    )

    assert result.hit_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.average_precision_at_k == 0.0
    assert result.ndcg_at_k == 0.0


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        -10,
    ],
)
def test_metrics_reject_invalid_k(
    k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        RankingMetrics.hit_at_k(
            ranked_ids=[],
            relevance={},
            k=k,
        )
