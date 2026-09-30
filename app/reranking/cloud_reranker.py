from __future__ import annotations
from typing import Sequence

import httpx

from app.config.settings import CloudConfig, RerankingConfig
from app.reranking.base import Reranker
from app.reranking.cross_encoder_reranker import (
    RerankedParentResult,
    RerankingResponse,
)
from app.retrieval.models import ParentRetrievalResult


class CloudReranker(Reranker):

    def __init__(
        self, reranking_config: RerankingConfig, cloud_config: CloudConfig
    ) -> None:
        self.model_name = reranking_config.model_name
        self.top_k = reranking_config.top_k

        self.reranker_url = cloud_config.reranker_url
        self.api_key = cloud_config.api_key
        self.timeout_seconds = cloud_config.timeout_seconds

        if not self.api_key:
            raise ValueError("CLOUD_EMBEDDING_API_KEY is required.")

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[ParentRetrievalResult],
        top_k: int | None = None,
    ) -> RerankingResponse:

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            return RerankingResponse(
                query=query,
                candidate_count=0,
                returned_count=0,
                results=[],
            )

        documents = [candidate.parent.content for candidate in candidates]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": (top_k if top_k is not None else len(documents)),
        }

        headers = {
            "Authorization": (f"Bearer {self.api_key}"),
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                self.reranker_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        body = response.json()
        api_results = body.get("results")

        if api_results is None:
            raise RuntimeError("Cloud reranker returned no results.")

        reranked_results: list[RerankedParentResult] = []

        for reranked_index, item in enumerate(
            api_results,
            start=1,
        ):

            original_index = int(item["index"])

            if original_index < 0 or original_index >= len(candidates):
                raise RuntimeError(
                    "Cloud reranker returned " "an invalid document index."
                )

            candidate = candidates[original_index]

            reranked_results.append(
                RerankedParentResult(
                    original_rank=(original_index + 1),
                    reranked_rank=(reranked_index),
                    vector_score=(candidate.best_score),
                    reranker_score=float(item["relevance_score"]),
                    result=candidate,
                )
            )

        return RerankingResponse(
            query=query,
            candidate_count=len(candidates),
            returned_count=(len(reranked_results)),
            results=reranked_results,
        )
