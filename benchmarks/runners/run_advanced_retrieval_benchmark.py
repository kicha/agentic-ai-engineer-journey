from app.config.settings import (
    ChunkingConfig,
    HybridSearchConfig,
    QdrantConfig,
    RerankingConfig,
)

from app.embeddings.local_dense_embedding_service import (
    LocalDenseEmbeddingService,
)

from app.embeddings.local_sparse_embedding_service import (
    LocalSparseEmbeddingService,
)

from benchmarks.evaluation.retrieval_benchmark import (
    GoldQueryLoader,
)

from benchmarks.evaluation.advanced_retrieval_benchmark import (
    AdvancedRetrievalBenchmarkEvaluator,
)

from app.repositories.parent_repository import (
    ParentRepository,
)

from app.vectorstore.hybrid_qdrant_store import (
    HybridQdrantStore,
)

from app.retrieval.hybrid_child_retriever import (
    HybridChildRetriever,
)

from app.retrieval.hybrid_parent_child_retriever import (
    HybridParentChildRetriever,
)

from app.reranking.cross_encoder_reranker import (
    LocalCrossEncoderReranker,
)

from app.retrieval.hybrid_rerank_mmr_retriever import (
    HybridRerankMMRRetriever,
)
from benchmarks.paths import GOLD_QUERIES_PATH

# ============================================================
# PATHS
# ============================================================

GOLD_QUERY_PATH = GOLD_QUERIES_PATH


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    config = ChunkingConfig()
    hybrid_config = HybridSearchConfig()
    qdrant_config = QdrantConfig()
    reranking_config = RerankingConfig()

    # ========================================================
    # EMBEDDERS
    # ========================================================

    dense_embedder = LocalDenseEmbeddingService(config.embedding_model_name)
    sparse_embedder = LocalSparseEmbeddingService(
        model_name=(config.sparse_embedding_model)
    )

    vector_size = dense_embedder.embedding_dimension

    # ========================================================
    # QDRANT
    # ========================================================

    hybrid_store = HybridQdrantStore(
        config=hybrid_config, vector_size=vector_size, qdrant_config=qdrant_config
    )

    # ========================================================
    # MONGODB
    # ========================================================

    parent_repository = ParentRepository(config)

    try:

        # ====================================================
        # HYBRID CHILD
        # ====================================================

        hybrid_child_retriever = HybridChildRetriever(
            dense_embedding_service=(dense_embedder),
            sparse_embedding_service=(sparse_embedder),
            vector_store=(hybrid_store),
        )

        # ====================================================
        # HYBRID PARENT EXPANSION
        # ====================================================

        hybrid_parent_retriever = HybridParentChildRetriever(
            hybrid_child_retriever=(hybrid_child_retriever),
            parent_repository=(parent_repository),
        )

        # ====================================================
        # CROSSENCODER
        # ====================================================

        reranker = LocalCrossEncoderReranker(model_name=reranking_config.model_name)

        # ====================================================
        # FULL PIPELINE
        # ====================================================

        hybrid_rerank_mmr = HybridRerankMMRRetriever(
            hybrid_parent_child_retriever=hybrid_parent_retriever,
            reranker=(reranker),
            embedder=(dense_embedder),
            lambda_mult=0.7,
        )

        # ====================================================
        # GOLD QUERIES
        # ====================================================

        gold_queries = GoldQueryLoader.load(GOLD_QUERY_PATH)

        # ====================================================
        # EVALUATOR
        # ====================================================

        evaluator = AdvancedRetrievalBenchmarkEvaluator(
            hybrid_parent_retriever=(hybrid_parent_retriever),
            cross_encoder=(reranker),
            hybrid_rerank_mmr_retriever=(hybrid_rerank_mmr),
            # broad child pool
            hybrid_child_top_k=30,
            # prune parent candidates
            cross_encoder_top_k=10,
            # final context budget
            final_top_k=3,
        )

        # ====================================================
        # RUN
        # ====================================================

        results = evaluator.evaluate(gold_queries)
        evaluator.print_summary(results)

    finally:

        parent_repository.close()


if __name__ == "__main__":
    main()

