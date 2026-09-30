from __future__ import annotations

from typing import Dict, List, Sequence

from qdrant_client.models import ScoredPoint

from app.repositories.parent_repository import ParentRepository
from app.retrieval.hybrid_child_retriever import HybridChildRetriever
from app.retrieval.models import ChildSearchHit, ParentRetrievalResult


class HybridParentChildRetriever:
    """Hybrid child retrieval followed by unique MongoDB parent expansion."""

    def __init__(
        self,
        hybrid_child_retriever: HybridChildRetriever,
        parent_repository: ParentRepository,
    ) -> None:
        self.hybrid_child_retriever = hybrid_child_retriever
        self.parent_repository = parent_repository

    def retrieve(
        self,
        query: str,
        child_top_k: int = 15,
        parent_top_k: int | None = None,
    ) -> List[ParentRetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query cannot be empty.")
        if child_top_k <= 0:
            raise ValueError("child_top_k must be greater than zero.")

        hits = self.hybrid_child_retriever.retrieve_hybrid(
            query=normalized_query,
            limit=child_top_k,
        )
        return self.expand_hits(hits=hits, parent_top_k=parent_top_k)

    def expand_hits(
        self,
        *,
        hits: Sequence[ScoredPoint],
        parent_top_k: int | None = None,
    ) -> List[ParentRetrievalResult]:
        if parent_top_k is not None and parent_top_k <= 0:
            raise ValueError("parent_top_k must be greater than zero.")
        if not hits:
            return []

        children_by_parent: Dict[str, List[ChildSearchHit]] = {}
        for hit in hits:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            if not parent_id:
                raise RuntimeError(
                    "Hybrid Qdrant child payload is missing 'parent_id'. "
                    f"Point ID: {hit.id}"
                )

            child_hit = ChildSearchHit(
                child_id=str(payload.get("child_id", hit.id)),
                parent_id=str(parent_id),
                document_id=str(payload.get("document_id", "")),
                child_index=int(payload.get("child_index", -1)),
                content=str(payload.get("content", "")),
                score=float(hit.score),
            )
            children_by_parent.setdefault(str(parent_id), []).append(child_hit)

        parent_results: List[ParentRetrievalResult] = []
        for parent_id, child_hits in children_by_parent.items():
            parent = self.parent_repository.get_by_id(parent_id)
            if parent is None:
                raise RuntimeError(
                    "Dangling ChildChunk reference detected. "
                    f"Missing parent_id={parent_id}"
                )

            child_hits.sort(key=lambda child: child.score, reverse=True)
            parent_results.append(
                ParentRetrievalResult(
                    parent=parent,
                    best_score=child_hits[0].score,
                    matched_children=child_hits,
                )
            )

        parent_results.sort(key=lambda result: result.best_score, reverse=True)
        if parent_top_k is not None:
            return parent_results[:parent_top_k]
        return parent_results
