from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.reranking.cloud_reranker as cloud_reranker_module
from app.config.settings import (
    CloudConfig,
    RerankerProvider,
    RerankingConfig,
)
from app.reranking.cloud_reranker import CloudReranker
from app.retrieval.models import ParentRetrievalResult

# ============================================================
# TEST HELPERS
# ============================================================


def build_reranking_config() -> RerankingConfig:
    """
    Return an explicit cloud reranking configuration.

    The model is intentionally supplied explicitly because the
    RerankingConfig default currently points to the LOCAL
    MS-MARCO CrossEncoder model.
    """

    return RerankingConfig(
        provider=RerankerProvider.CLOUD,
        model_name="jina-reranker-v2-base-multilingual",
        top_k=5,
    )


def build_cloud_config() -> CloudConfig:
    """
    Return cloud configuration with a fake API key.

    Unit tests must never depend on a real environment secret.
    """

    return CloudConfig(
        api_key="test-api-key",
        reranker_url="https://example.test/v1/rerank",
        timeout_seconds=10.0,
    )


def build_candidate(
    *,
    content: str,
    best_score: float,
) -> ParentRetrievalResult:
    """
    Construct the minimum ParentRetrievalResult needed by the
    reranker test.

    model_construct() is intentional here.

    CloudReranker only needs:
        candidate.parent.content
        candidate.best_score

    The complete ParentChunk graph is tested elsewhere, so this
    unit test should not duplicate all ParentChunk construction.
    """

    parent = SimpleNamespace(
        content=content,
    )

    return ParentRetrievalResult.model_construct(
        parent=parent,
        best_score=best_score,
    )


class FakeResponse:
    """
    Minimal httpx-like response used by CloudReranker tests.
    """

    def __init__(
        self,
        body: dict[str, Any],
        *,
        status_error: Exception | None = None,
    ) -> None:
        self._body = body
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    def json(self) -> dict[str, Any]:
        return self._body


class FakeHttpClient:
    """
    Context-manager-compatible fake replacement for httpx.Client.
    """

    response_body: dict[str, Any] = {}
    status_error: Exception | None = None

    last_url: str | None = None
    last_headers: dict[str, str] | None = None
    last_payload: dict[str, Any] | None = None
    last_timeout: float | None = None

    def __init__(
        self,
        *,
        timeout: float,
    ) -> None:
        type(self).last_timeout = timeout

    def __enter__(self) -> "FakeHttpClient":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeResponse:

        type(self).last_url = url
        type(self).last_headers = headers
        type(self).last_payload = json

        return FakeResponse(
            body=type(self).response_body,
            status_error=type(self).status_error,
        )


@pytest.fixture(autouse=True)
def reset_fake_http_client() -> None:
    """
    Prevent state leaking between tests.
    """

    FakeHttpClient.response_body = {}
    FakeHttpClient.status_error = None

    FakeHttpClient.last_url = None
    FakeHttpClient.last_headers = None
    FakeHttpClient.last_payload = None
    FakeHttpClient.last_timeout = None


# ============================================================
# CONSTRUCTOR
# ============================================================


def test_cloud_reranker_requires_api_key() -> None:
    """
    A cloud reranker cannot operate without credentials.

    This protects us from accidentally reaching the provider with
    an unauthenticated configuration.
    """

    reranking_config = build_reranking_config()

    cloud_config = CloudConfig(
        api_key=None,
        reranker_url="https://example.test/v1/rerank",
    )

    with pytest.raises(
        ValueError,
        match="CLOUD_EMBEDDING_API_KEY is required",
    ):
        CloudReranker(
            reranking_config=reranking_config,
            cloud_config=cloud_config,
        )


# ============================================================
# INPUT VALIDATION
# ============================================================


def test_cloud_reranker_rejects_empty_query() -> None:
    """
    Empty queries must fail before making a network request.
    """

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate = build_candidate(
        content="HNSW uses graph traversal.",
        best_score=0.80,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        reranker.rerank(
            query="   ",
            candidates=[candidate],
        )


def test_cloud_reranker_rejects_non_positive_top_k() -> None:
    """
    top_k must represent a positive number of returned parents.
    """

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate = build_candidate(
        content="HNSW uses graph traversal.",
        best_score=0.80,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        reranker.rerank(
            query="How does HNSW search?",
            candidates=[candidate],
            top_k=0,
        )


def test_cloud_reranker_returns_empty_response_for_no_candidates() -> None:
    """
    No candidates is a valid retrieval outcome.

    The reranker should return a typed empty response rather than
    attempting a provider call.
    """

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    response = reranker.rerank(
        query="How does HNSW search?",
        candidates=[],
    )

    assert response.query == "How does HNSW search?"
    assert response.candidate_count == 0
    assert response.returned_count == 0
    assert response.results == []


# ============================================================
# PROVIDER REQUEST
# ============================================================


def test_cloud_reranker_sends_expected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify our provider adapter sends the expected model, query,
    documents, top_n, authorization header and timeout.

    This is the main API-contract unit test.
    """

    FakeHttpClient.response_body = {
        "results": [
            {
                "index": 0,
                "relevance_score": 0.91,
            },
            {
                "index": 1,
                "relevance_score": 0.42,
            },
        ]
    }

    monkeypatch.setattr(
        cloud_reranker_module.httpx,
        "Client",
        FakeHttpClient,
    )

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidates = [
        build_candidate(
            content="HNSW navigates a hierarchical graph.",
            best_score=0.81,
        ),
        build_candidate(
            content="Metadata can be stored in payload fields.",
            best_score=0.63,
        ),
    ]

    reranker.rerank(
        query="How does HNSW search efficiently?",
        candidates=candidates,
        top_k=2,
    )

    assert FakeHttpClient.last_url == "https://example.test/v1/rerank"

    assert FakeHttpClient.last_timeout == 10.0

    assert FakeHttpClient.last_headers == {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json",
    }

    assert FakeHttpClient.last_payload == {
        "model": "jina-reranker-v2-base-multilingual",
        "query": "How does HNSW search efficiently?",
        "documents": [
            "HNSW navigates a hierarchical graph.",
            "Metadata can be stored in payload fields.",
        ],
        "top_n": 2,
    }


# ============================================================
# RERANKING / INDEX MAPPING
# ============================================================


def test_cloud_reranker_maps_provider_indices_to_original_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Provider results contain zero-based indices into the original
    candidate list.

    This test protects the most important mapping in CloudReranker:
    hosted rank -> original ParentRetrievalResult.
    """

    FakeHttpClient.response_body = {
        "results": [
            {
                "index": 2,
                "relevance_score": 0.95,
            },
            {
                "index": 0,
                "relevance_score": 0.80,
            },
            {
                "index": 1,
                "relevance_score": 0.20,
            },
        ]
    }

    monkeypatch.setattr(
        cloud_reranker_module.httpx,
        "Client",
        FakeHttpClient,
    )

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate_1 = build_candidate(
        content="Candidate one",
        best_score=0.90,
    )

    candidate_2 = build_candidate(
        content="Candidate two",
        best_score=0.80,
    )

    candidate_3 = build_candidate(
        content="Candidate three",
        best_score=0.70,
    )

    response = reranker.rerank(
        query="test query",
        candidates=[
            candidate_1,
            candidate_2,
            candidate_3,
        ],
        top_k=3,
    )

    assert response.candidate_count == 3
    assert response.returned_count == 3

    # Provider index 2 means original candidate #3.
    assert response.results[0].original_rank == 3
    assert response.results[0].reranked_rank == 1
    assert response.results[0].reranker_score == pytest.approx(0.95)
    assert response.results[0].vector_score == pytest.approx(0.70)
    assert response.results[0].result is candidate_3

    # Provider index 0 means original candidate #1.
    assert response.results[1].original_rank == 1
    assert response.results[1].reranked_rank == 2
    assert response.results[1].reranker_score == pytest.approx(0.80)
    assert response.results[1].result is candidate_1

    # Provider index 1 means original candidate #2.
    assert response.results[2].original_rank == 2
    assert response.results[2].reranked_rank == 3
    assert response.results[2].reranker_score == pytest.approx(0.20)
    assert response.results[2].result is candidate_2


def test_cloud_reranker_rejects_invalid_provider_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A provider index outside the submitted candidate list indicates
    a corrupted or incompatible provider response.

    We should fail loudly instead of returning the wrong parent.
    """

    FakeHttpClient.response_body = {
        "results": [
            {
                "index": 99,
                "relevance_score": 0.99,
            },
        ]
    }

    monkeypatch.setattr(
        cloud_reranker_module.httpx,
        "Client",
        FakeHttpClient,
    )

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate = build_candidate(
        content="Candidate one",
        best_score=0.90,
    )

    with pytest.raises(
        RuntimeError,
        match="invalid document index",
    ):
        reranker.rerank(
            query="test query",
            candidates=[candidate],
        )


def test_cloud_reranker_rejects_missing_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A successful HTTP response without a results collection does
    not satisfy the provider contract.
    """

    FakeHttpClient.response_body = {}

    monkeypatch.setattr(
        cloud_reranker_module.httpx,
        "Client",
        FakeHttpClient,
    )

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate = build_candidate(
        content="Candidate one",
        best_score=0.90,
    )

    with pytest.raises(
        RuntimeError,
        match="Cloud reranker returned no results",
    ):
        reranker.rerank(
            query="test query",
            candidates=[candidate],
        )


# ============================================================
# HTTP FAILURE
# ============================================================


def test_cloud_reranker_propagates_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Provider/network failures must not be silently converted into
    fake reranking results.
    """

    request = httpx.Request(
        method="POST",
        url="https://example.test/v1/rerank",
    )

    response = httpx.Response(
        status_code=500,
        request=request,
    )

    FakeHttpClient.status_error = httpx.HTTPStatusError(
        "Provider failure",
        request=request,
        response=response,
    )

    monkeypatch.setattr(
        cloud_reranker_module.httpx,
        "Client",
        FakeHttpClient,
    )

    reranker = CloudReranker(
        reranking_config=build_reranking_config(),
        cloud_config=build_cloud_config(),
    )

    candidate = build_candidate(
        content="Candidate one",
        best_score=0.90,
    )

    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank(
            query="test query",
            candidates=[candidate],
        )
