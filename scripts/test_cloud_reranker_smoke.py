from __future__ import annotations

from types import SimpleNamespace

from app.config.settings import (
    CloudConfig,
    RerankerProvider,
    RerankingConfig,
)
from app.reranking.cloud_reranker import CloudReranker
from app.retrieval.models import ParentRetrievalResult

# ============================================================
# HELPERS
# ============================================================


def build_candidate(
    *,
    content: str,
    best_score: float,
) -> ParentRetrievalResult:
    """
    Build the minimal ParentRetrievalResult needed by the
    CloudReranker.

    This smoke test is validating the real hosted reranker API,
    not ParentChunk construction, so we intentionally keep the
    candidate object minimal.
    """

    parent = SimpleNamespace(
        content=content,
    )

    return ParentRetrievalResult.model_construct(
        parent=parent,
        best_score=best_score,
    )


# ============================================================
# SMOKE TEST
# ============================================================


def main() -> None:

    print("=" * 80)
    print("CLOUD RERANKER SMOKE TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------

    reranking_config = RerankingConfig(
        provider=RerankerProvider.CLOUD,
        top_k=3,
    )

    cloud_config = CloudConfig()

    print()
    print("Provider       :", reranking_config.provider)
    print("Reranker model :", reranking_config.model_name)
    print("Reranker URL   :", cloud_config.reranker_url)
    print(
        "API key set    :",
        bool(cloud_config.api_key),
    )

    # --------------------------------------------------------
    # 2. Create real cloud reranker
    # --------------------------------------------------------

    reranker = CloudReranker(
        reranking_config=reranking_config,
        cloud_config=cloud_config,
    )

    # --------------------------------------------------------
    # 3. Small controlled candidate set
    # --------------------------------------------------------

    query = (
        "Why does HNSW avoid comparing the query "
        "against every vector in the database?"
    )

    candidates = [
        build_candidate(
            content=(
                "HNSW uses a hierarchical graph to navigate "
                "toward promising regions of the vector space. "
                "Instead of comparing the query against every "
                "stored vector, the search follows graph "
                "connections through progressively denser layers."
            ),
            best_score=0.71,
        ),
        build_candidate(
            content=(
                "Vector databases can store metadata together "
                "with vectors. Metadata can include source, "
                "category, document identifier, section, and "
                "other application-specific attributes."
            ),
            best_score=0.64,
        ),
        build_candidate(
            content=(
                "The HNSW M parameter controls the number of "
                "graph connections maintained for each node. "
                "Higher M may improve recall but increases "
                "memory usage and graph construction cost."
            ),
            best_score=0.68,
        ),
    ]

    # --------------------------------------------------------
    # 4. Show original order
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("ORIGINAL CANDIDATES")
    print("-" * 80)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print()
        print(f"Original rank : {index}")
        print(f"Vector score  : " f"{candidate.best_score:.4f}")
        print("Content       : " f"{candidate.parent.content}")

    # --------------------------------------------------------
    # 5. REAL hosted API call
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("CALLING HOSTED RERANKER...")
    print("-" * 80)

    response = reranker.rerank(
        query=query,
        candidates=candidates,
        top_k=3,
    )

    # --------------------------------------------------------
    # 6. Validate response
    # --------------------------------------------------------

    if response.candidate_count != 3:
        raise RuntimeError("Unexpected candidate_count returned.")

    if response.returned_count != 3:
        raise RuntimeError("Unexpected returned_count returned.")

    if len(response.results) != 3:
        raise RuntimeError("Expected exactly 3 reranked results.")

    # --------------------------------------------------------
    # 7. Print reranked order
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for item in response.results:
        print()
        print(f"Reranked rank : " f"{item.reranked_rank}")
        print(f"Original rank : " f"{item.original_rank}")
        print(f"Vector score  : " f"{item.vector_score:.4f}")
        print(f"Reranker score: " f"{item.reranker_score:.6f}")
        print("Content       : " f"{item.result.parent.content}")

    # --------------------------------------------------------
    # 8. Simple semantic sanity check
    # --------------------------------------------------------

    top_result = response.results[0]

    top_content = top_result.result.parent.content.lower()

    expected_terms = (
        "instead of comparing",
        "graph",
    )

    if not all(term in top_content for term in expected_terms):
        raise RuntimeError(
            "Smoke test completed technically, "
            "but the expected HNSW search-process "
            "candidate was not ranked first."
        )

    # --------------------------------------------------------
    # 9. PASS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CLOUD RERANKER SMOKE TEST PASSED")
    print("=" * 80)

    print()
    print("Verified:")
    print("  - cloud API authentication")
    print("  - real provider endpoint")
    print("  - configured reranker model")
    print("  - request payload compatibility")
    print("  - response parsing")
    print("  - provider index -> parent mapping")
    print("  - relevance scores returned")
    print("  - semantically sensible top result")


if __name__ == "__main__":
    main()
