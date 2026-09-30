from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.models import ParentRetrievalResult

    from app.reranking.cross_encoder_reranker import (
        RerankingResponse,
    )


class Reranker(ABC):
    """
    Contract for parent-level reranking providers.
    """

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence["ParentRetrievalResult"],
        top_k: int | None = None,
    ) -> "RerankingResponse":
        raise NotImplementedError
