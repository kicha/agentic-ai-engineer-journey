from pathlib import Path

from app.config.settings import ChunkingConfig, HybridSearchConfig, QdrantConfig

from app.embeddings.local_dense_embedding_service import (
    LocalDenseEmbeddingService,
)

from app.embeddings.local_sparse_embedding_service import (
    LocalSparseEmbeddingService,
)

from benchmarks.evaluation.retrieval_benchmark import (
    GoldQueryLoader,
    RetrievalBenchmarkEvaluator,
)

from app.retrieval.hybrid_child_retriever import (
    HybridChildRetriever,
)

from app.vectorstore.hybrid_qdrant_store import (
    HybridQdrantStore,
)
from benchmarks.paths import LEGACY_GOLD_QUERIES_PATH, BENCHMARK_RESULTS_DIR

# ============================================================
# PATHS
# ============================================================

GOLD_QUERY_PATH = LEGACY_GOLD_QUERIES_PATH
OUTPUT_PATH = BENCHMARK_RESULTS_DIR / "retrieval_benchmark_results.json"


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()
    print("=" * 100)
    print("AGENTIC RAG RETRIEVAL BENCHMARK")
    print("=" * 100)

    # ========================================================
    # CONFIG
    # ========================================================

    config = ChunkingConfig()
    hybrid_config = HybridSearchConfig()
    qdrant_config = QdrantConfig()

    # ========================================================
    # EMBEDDING SERVICES
    # ========================================================

    dense_embedder = LocalDenseEmbeddingService(config.embedding_model_name)

    sparse_embedder = LocalSparseEmbeddingService(
        model_name=(config.sparse_embedding_model)
    )

    vector_size = dense_embedder.embedding_dimension

    # ========================================================
    # HYBRID VECTOR STORE
    # ========================================================

    hybrid_store = HybridQdrantStore(
        config=hybrid_config, vector_size=vector_size, qdrant_config=qdrant_config
    )

    # ========================================================
    # RETRIEVER
    # ========================================================

    retriever = HybridChildRetriever(
        dense_embedding_service=(dense_embedder),
        sparse_embedding_service=(sparse_embedder),
        vector_store=(hybrid_store),
    )

    # ========================================================
    # LOAD GOLD QUERIES
    # ========================================================

    gold_queries = GoldQueryLoader.load(GOLD_QUERY_PATH)

    print(f"\nGold queries loaded: " f"{len(gold_queries)}")

    # ========================================================
    # EVALUATOR
    # ========================================================

    evaluator = RetrievalBenchmarkEvaluator(
        retriever=retriever,
        vector_store=hybrid_store,
        # ----------------------------------------------
        # Our currently verified benchmark collection.
        # ----------------------------------------------
        #####   HARDCODED VALUE 196
        expected_point_count=196,
        # ----------------------------------------------
        # Retrieve a broad CHILD pool first.
        #
        # Then collapse those child hits into unique
        # document rankings.
        # ----------------------------------------------
        child_candidate_limit=50,
        # ----------------------------------------------
        # Metrics require top 10 documents.
        # ----------------------------------------------
        document_limit=10,
    )

    # ========================================================
    # RUN
    # ========================================================

    report = evaluator.evaluate(gold_queries=gold_queries)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    evaluator.print_summary(report)

    # ========================================================
    # SAVE FULL RESULTS
    # ========================================================

    evaluator.save_report(
        report=report,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

