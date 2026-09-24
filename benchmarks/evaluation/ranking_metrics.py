# benchmarks/evaluation/ranking_metrics.py

from __future__ import annotations

import math
from pydantic import BaseModel, ConfigDict
from typing import Mapping, Sequence


class RankingMetricsResult(BaseModel):
    model_config = ConfigDict(
        frozen=True,
    )

    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision_at_k: float
    ndcg_at_k: float


class RankingMetrics:
    @staticmethod
    def hit_at_k(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> float:
        RankingMetrics._validate_k(k)

        top_k = ranked_ids[:k]

        return float(
            any(
                RankingMetrics._is_relevant(
                    item_id=item_id,
                    relevance=relevance,
                )
                for item_id in top_k
            )
        )

    @staticmethod
    def precision_at_k(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> float:
        RankingMetrics._validate_k(k)

        top_k = ranked_ids[:k]

        if not top_k:
            return 0.0

        relevant_count = sum(
            1
            for item_id in top_k
            if RankingMetrics._is_relevant(
                item_id=item_id,
                relevance=relevance,
            )
        )

        return relevant_count / k

    @staticmethod
    def recall_at_k(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> float:
        RankingMetrics._validate_k(k)

        total_relevant = sum(1 for grade in relevance.values() if grade > 0)

        if total_relevant == 0:
            return 0.0

        top_k = ranked_ids[:k]

        retrieved_relevant = sum(
            1
            for item_id in top_k
            if RankingMetrics._is_relevant(
                item_id=item_id,
                relevance=relevance,
            )
        )

        return retrieved_relevant / total_relevant

    @staticmethod
    def reciprocal_rank(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
    ) -> float:
        for rank, item_id in enumerate(
            ranked_ids,
            start=1,
        ):
            if RankingMetrics._is_relevant(
                item_id=item_id,
                relevance=relevance,
            ):
                return 1.0 / rank

        return 0.0

    @staticmethod
    def average_precision_at_k(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> float:
        RankingMetrics._validate_k(k)

        total_relevant = sum(1 for grade in relevance.values() if grade > 0)

        if total_relevant == 0:
            return 0.0

        top_k = ranked_ids[:k]

        relevant_seen = 0
        precision_sum = 0.0

        for rank, item_id in enumerate(
            top_k,
            start=1,
        ):
            if not RankingMetrics._is_relevant(
                item_id=item_id,
                relevance=relevance,
            ):
                continue

            relevant_seen += 1

            precision_sum += relevant_seen / rank

        denominator = min(
            total_relevant,
            k,
        )

        if denominator == 0:
            return 0.0

        return precision_sum / denominator

    @staticmethod
    def ndcg_at_k(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> float:
        RankingMetrics._validate_k(k)

        top_k = ranked_ids[:k]

        dcg = RankingMetrics._dcg(
            ranked_ids=top_k,
            relevance=relevance,
        )

        ideal_grades = sorted(
            (grade for grade in relevance.values() if grade > 0),
            reverse=True,
        )[:k]

        if not ideal_grades:
            return 0.0

        ideal_dcg = sum(
            RankingMetrics._discounted_gain(
                grade=grade,
                rank=rank,
            )
            for rank, grade in enumerate(
                ideal_grades,
                start=1,
            )
        )

        if ideal_dcg == 0.0:
            return 0.0

        return dcg / ideal_dcg

    @staticmethod
    def evaluate(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
        k: int,
    ) -> RankingMetricsResult:
        return RankingMetricsResult(
            hit_at_k=RankingMetrics.hit_at_k(
                ranked_ids=ranked_ids,
                relevance=relevance,
                k=k,
            ),
            precision_at_k=RankingMetrics.precision_at_k(
                ranked_ids=ranked_ids,
                relevance=relevance,
                k=k,
            ),
            recall_at_k=RankingMetrics.recall_at_k(
                ranked_ids=ranked_ids,
                relevance=relevance,
                k=k,
            ),
            reciprocal_rank=RankingMetrics.reciprocal_rank(
                ranked_ids=ranked_ids,
                relevance=relevance,
            ),
            average_precision_at_k=(
                RankingMetrics.average_precision_at_k(
                    ranked_ids=ranked_ids,
                    relevance=relevance,
                    k=k,
                )
            ),
            ndcg_at_k=RankingMetrics.ndcg_at_k(
                ranked_ids=ranked_ids,
                relevance=relevance,
                k=k,
            ),
        )

    @staticmethod
    def _is_relevant(
        item_id: str,
        relevance: Mapping[str, int],
    ) -> bool:
        return (
            relevance.get(
                item_id,
                0,
            )
            > 0
        )

    @staticmethod
    def _dcg(
        ranked_ids: Sequence[str],
        relevance: Mapping[str, int],
    ) -> float:
        return sum(
            RankingMetrics._discounted_gain(
                grade=relevance.get(
                    item_id,
                    0,
                ),
                rank=rank,
            )
            for rank, item_id in enumerate(
                ranked_ids,
                start=1,
            )
        )

    @staticmethod
    def _discounted_gain(
        grade: int,
        rank: int,
    ) -> float:
        if grade <= 0:
            return 0.0

        gain = (2**grade) - 1

        discount = math.log2(rank + 1)

        return gain / discount

    @staticmethod
    def _validate_k(
        k: int,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
