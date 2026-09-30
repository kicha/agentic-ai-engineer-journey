from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel

from app.models.parent_chunk import ParentChunk


class ParentMMRCandidate(BaseModel):
    parent: ParentChunk
    query_similarity: float
    reranker_score: float
    reranker_rank: int
    embedding: List[float]


class ParentMMRResult(BaseModel):
    rank: int
    parent: ParentChunk
    reranker_rank: int
    reranker_score: float
    query_similarity: float
    max_redundancy: float
    relevance_contribution: float
    redundancy_penalty: float
    mmr_score: float


class ParentMMRSelector:
    """Select relevant but non-redundant parents in dense embedding space.

    Reranker score/rank are retained for traceability only. MMR relevance uses
    query-to-parent cosine similarity, while redundancy uses parent-to-parent
    cosine similarity. Score spaces are never mixed.
    """

    def __init__(self, lambda_mult: float = 0.7) -> None:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be between 0 and 1.")
        self.lambda_mult = lambda_mult

    @staticmethod
    def _cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0.0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / denominator)

    def select(
        self,
        candidates: List[ParentMMRCandidate],
        top_k: int,
    ) -> List[ParentMMRResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not candidates:
            return []

        remaining = candidates.copy()
        first = max(remaining, key=lambda candidate: candidate.query_similarity)
        remaining.remove(first)
        selected = [first]
        results = [
            ParentMMRResult(
                rank=1,
                parent=first.parent,
                reranker_rank=first.reranker_rank,
                reranker_score=first.reranker_score,
                query_similarity=first.query_similarity,
                max_redundancy=0.0,
                relevance_contribution=first.query_similarity,
                redundancy_penalty=0.0,
                mmr_score=first.query_similarity,
            )
        ]

        while remaining and len(selected) < top_k:
            best_candidate: ParentMMRCandidate | None = None
            best_mmr_score = float("-inf")
            best_max_redundancy = 0.0
            best_relevance_contribution = 0.0
            best_redundancy_penalty = 0.0

            for candidate in remaining:
                candidate_vector = np.asarray(candidate.embedding, dtype=np.float32)
                max_redundancy = max(
                    self._cosine_similarity(
                        candidate_vector,
                        np.asarray(selected_candidate.embedding, dtype=np.float32),
                    )
                    for selected_candidate in selected
                )
                relevance_contribution = (
                    self.lambda_mult * candidate.query_similarity
                )
                redundancy_penalty = (
                    (1.0 - self.lambda_mult) * max_redundancy
                )
                mmr_score = relevance_contribution - redundancy_penalty

                if mmr_score > best_mmr_score:
                    best_candidate = candidate
                    best_mmr_score = mmr_score
                    best_max_redundancy = max_redundancy
                    best_relevance_contribution = relevance_contribution
                    best_redundancy_penalty = redundancy_penalty

            if best_candidate is None:
                break

            selected.append(best_candidate)
            remaining.remove(best_candidate)
            results.append(
                ParentMMRResult(
                    rank=len(results) + 1,
                    parent=best_candidate.parent,
                    reranker_rank=best_candidate.reranker_rank,
                    reranker_score=best_candidate.reranker_score,
                    query_similarity=best_candidate.query_similarity,
                    max_redundancy=best_max_redundancy,
                    relevance_contribution=best_relevance_contribution,
                    redundancy_penalty=best_redundancy_penalty,
                    mmr_score=best_mmr_score,
                )
            )

        return results
