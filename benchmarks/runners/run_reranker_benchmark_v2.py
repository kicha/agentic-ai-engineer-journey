from __future__ import annotations

import json
import logging

from app.application_factory import build_runtime_configs
from app.config.settings import (
    CloudConfig,
    HybridSearchConfig,
    QdrantDeployment,
    RerankerProvider,
    RerankingConfig,
)
from app.embeddings.factory import (
    create_dense_embedding_service,
    create_sparse_embedding_service,
)
from benchmarks.runners.parent_mmr_runner import ParentMMRRunner
from app.embeddings.base import DenseEmbeddingService
from app.repositories.parent_repository import ParentRepository
from app.reranking.factory import create_reranker
from app.retrieval.hybrid_child_retriever import HybridChildRetriever
from app.utils.logging_config import configure_logging
from app.vectorstore.hybrid_qdrant_store import HybridQdrantStore
from benchmarks.evaluation.benchmark_evaluator import (
    BenchmarkEvaluationResult,
    BenchmarkEvaluator,
)
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    RetrievalStrategy,
)
from benchmarks.evaluation.gold_models import GoldDatasetV2
from benchmarks.paths import GOLD_QUERIES_PATH
from benchmarks.runners.baseline_parent_expansion_runner import (
    BaselineParentExpansionRunner,
)
from benchmarks.runners.baseline_retrieval_runner import BaselineRetrievalRunner

from benchmarks.runners.parent_reranking_runner import (
    ParentRerankingRunner,
)

logger = logging.getLogger(__name__)


# ============================================================
# BENCHMARK CONSTANTS
# ============================================================

CLOUD_COLLECTION_NAME = HybridSearchConfig().cloud_collection_name

CHILD_CANDIDATE_LIMIT = 10
PARENT_CANDIDATE_LIMIT = 5

# Number of parents retained AFTER reranking
# so that MMR still has a candidate pool to work with.
RERANKER_CANDIDATE_K = 5

# True final context size after Parent MMR.
FINAL_PARENT_K = 3

MMR_LAMBDA = 0.7

DEFAULT_RERANKING_CONFIG = RerankingConfig()
CLOUD_RERANKER_MODEL = DEFAULT_RERANKING_CONFIG.model_name
LOCAL_RERANKER_MODEL = DEFAULT_RERANKING_CONFIG.local_model_name


# ============================================================
# GOLD DATASET
# ============================================================


def load_gold_dataset() -> GoldDatasetV2:
    """
    Load the frozen Benchmark V2 gold judgments.
    """

    raw_data = json.loads(
        GOLD_QUERIES_PATH.read_text(
            encoding="utf-8",
        )
    )

    return GoldDatasetV2.model_validate(raw_data)


# ============================================================
# LOCAL RERANKER
# ============================================================


def build_local_reranker():
    """
    Construct the existing local CrossEncoder reranker.
    """

    reranking_config = RerankingConfig(
        provider=RerankerProvider.LOCAL,
        model_name=LOCAL_RERANKER_MODEL,
        top_k=RERANKER_CANDIDATE_K,
    )

    return create_reranker(
        config=reranking_config,
    )


# ============================================================
# CLOUD HYBRID CHILD RETRIEVER
# ============================================================


def build_cloud_child_retriever() -> tuple[
    HybridChildRetriever,
    DenseEmbeddingService,
]:
    """
    Build the same cloud retrieval stack used by the accepted
    Benchmark V2 cloud baseline.

    Dense:
        hosted cloud embedding provider

    Sparse:
        Qdrant Cloud BM25 inference

    Vector database:
        Qdrant Cloud

    Collection:
        configured Qdrant Cloud collection
    """

    runtime = build_runtime_configs(
        deployment=QdrantDeployment.CLOUD,
        collection_name=CLOUD_COLLECTION_NAME,
    )

    dense_embedding_service = create_dense_embedding_service(
        config=runtime.embedding,
        cloud_config=runtime.cloud,
    )

    sparse_embedding_service = create_sparse_embedding_service(
        config=runtime.embedding,
    )

    vector_store = HybridQdrantStore(
        config=runtime.hybrid,
        vector_size=dense_embedding_service.embedding_dimension,
        qdrant_config=runtime.qdrant,
        collection_name=runtime.collection_name,
    )

    child_retriever = HybridChildRetriever(
        vector_store=vector_store,
        dense_embedding_service=(dense_embedding_service),
        sparse_embedding_service=(sparse_embedding_service),
    )

    return (
        child_retriever,
        dense_embedding_service,
    )


# ============================================================
# PARENT REPOSITORY
# ============================================================


def build_parent_repository() -> ParentRepository:
    """
    MongoDB remains authoritative for ParentChunk content.
    """

    runtime = build_runtime_configs(deployment=QdrantDeployment.CLOUD)
    return ParentRepository(config=runtime.mongo)


# ============================================================
# CLOUD RERANKER
# ============================================================


def build_cloud_reranker():
    """
    Construct the real hosted Jina parent reranker.
    """

    reranking_config = RerankingConfig(
        provider=RerankerProvider.CLOUD,
        model_name=CLOUD_RERANKER_MODEL,
        top_k=RERANKER_CANDIDATE_K,
    )

    return create_reranker(
        config=reranking_config,
        cloud_config=CloudConfig(),
    )


# ============================================================
# CENTRALIZED EVALUATION
# ============================================================


def evaluate_run(
    *,
    evaluator: BenchmarkEvaluator,
    gold_dataset: GoldDatasetV2,
    run_result: BenchmarkRunResult,
    evaluation_k: int,
) -> BenchmarkEvaluationResult:
    """
    Evaluate a standardized BenchmarkRunResult.

    No metric logic is duplicated here.
    """

    return evaluator.evaluate(
        gold_dataset=gold_dataset,
        ranked_ids_by_query_id=(run_result.ranked_ids_by_query_id()),
        retrieval_level=(run_result.retrieval_level),
        k=evaluation_k,
    )


# ============================================================
# REPORTING
# ============================================================


def log_evaluation(
    *,
    strategy: RetrievalStrategy,
    stage: str,
    evaluation_result: BenchmarkEvaluationResult,
) -> None:
    """
    Log one standardized benchmark metric summary.
    """

    metrics = evaluation_result.metrics

    logger.info(
        f"strategy={strategy.value} "
        f"stage={stage} "
        f"k={evaluation_result.k} "
        f"hit_at_k="
        f"{metrics.mean_hit_at_k:.4f} "
        f"precision_at_k="
        f"{metrics.mean_precision_at_k:.4f} "
        f"recall_at_k="
        f"{metrics.mean_recall_at_k:.4f} "
        f"mrr="
        f"{metrics.mean_reciprocal_rank:.4f} "
        f"map_at_k="
        f"{metrics.mean_average_precision_at_k:.4f} "
        f"ndcg_at_k="
        f"{metrics.mean_ndcg_at_k:.4f}"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Run Benchmark V2 through:

        Cloud Hybrid child retrieval
            ↓
        Parent expansion
            ↓
        Hosted Jina parent reranking
            ↓
        Parent @3 evaluation

    The same parent candidate pool is evaluated before and
    after reranking.
    """

    configure_logging()

    logger.info("Starting Benchmark V2 hosted " "parent reranker evaluation")

    logger.info(
        f"cloud_collection="
        f"{CLOUD_COLLECTION_NAME} "
        f"child_candidate_limit="
        f"{CHILD_CANDIDATE_LIMIT} "
        f"parent_candidate_limit="
        f"{PARENT_CANDIDATE_LIMIT} "
        f"final_parent_k="
        f"{FINAL_PARENT_K} "
        f"reranker_model="
        f"{CLOUD_RERANKER_MODEL}"
    )

    gold_dataset = load_gold_dataset()

    child_retriever, dense_embedding_service = build_cloud_child_retriever()

    parent_repository = build_parent_repository()

    try:

        # ----------------------------------------------------
        # RUNNERS
        # ----------------------------------------------------

        child_runner = BaselineRetrievalRunner(
            retriever=child_retriever,
            child_candidate_limit=(CHILD_CANDIDATE_LIMIT),
        )

        parent_runner = BaselineParentExpansionRunner(
            parent_repository=(parent_repository),
            parent_candidate_limit=(PARENT_CANDIDATE_LIMIT),
        )

        logger.info("Loading local reranker")
        local_reranker = build_local_reranker()

        logger.info("Creating hosted reranker")
        cloud_reranker = build_cloud_reranker()

        local_reranking_runner = ParentRerankingRunner(
            parent_repository=(parent_repository),
            reranker=local_reranker,
            final_k=RERANKER_CANDIDATE_K,
        )

        cloud_reranking_runner = ParentRerankingRunner(
            parent_repository=(parent_repository),
            reranker=cloud_reranker,
            final_k=RERANKER_CANDIDATE_K,
        )

        parent_mmr_runner = ParentMMRRunner(
            parent_repository=(parent_repository),
            embedder=(dense_embedding_service),
            lambda_mult=MMR_LAMBDA,
            final_k=FINAL_PARENT_K,
        )

        evaluator = BenchmarkEvaluator()

        # ====================================================
        # STAGE 1
        # CLOUD HYBRID CHILD RETRIEVAL
        # ====================================================

        logger.info("Stage 1: cloud hybrid " "child retrieval")

        child_run = child_runner.run(
            gold_dataset=gold_dataset,
            strategy=(RetrievalStrategy.HYBRID),
        )

        # ====================================================
        # STAGE 2
        # PARENT EXPANSION
        # ====================================================

        logger.info("Stage 2: parent " "expansion/deduplication")

        parent_run = parent_runner.expand(
            child_run_result=child_run,
        )

        # ----------------------------------------------------
        # PRE-RERANK BASELINE
        # ----------------------------------------------------

        baseline_evaluation = evaluate_run(
            evaluator=evaluator,
            gold_dataset=gold_dataset,
            run_result=parent_run,
            evaluation_k=FINAL_PARENT_K,
        )

        log_evaluation(
            strategy=RetrievalStrategy.HYBRID,
            stage="parent_baseline_no_rerank",
            evaluation_result=(baseline_evaluation),
        )

        # ============================================================
        # STAGE 3A
        # LOCAL CROSS-ENCODER RERANKING
        # ============================================================

        logger.info("Stage 3A: local CrossEncoder " "parent reranking")

        local_reranked_run = local_reranking_runner.run(
            parent_run_result=parent_run,
        )

        local_evaluation = evaluate_run(
            evaluator=evaluator,
            gold_dataset=gold_dataset,
            run_result=local_reranked_run,
            evaluation_k=FINAL_PARENT_K,
        )

        log_evaluation(
            strategy=(RetrievalStrategy.HYBRID_RERANK),
            stage="parent_post_local_rerank",
            evaluation_result=(local_evaluation),
        )

        # ============================================================
        # STAGE 4A
        # LOCAL RERANKER -> PARENT MMR
        # ============================================================

        logger.info("Stage 4A: Parent MMR after " "local CrossEncoder reranking")

        local_mmr_run = parent_mmr_runner.run(
            reranked_run_result=(local_reranked_run),
        )

        local_mmr_evaluation = evaluate_run(
            evaluator=evaluator,
            gold_dataset=gold_dataset,
            run_result=local_mmr_run,
            evaluation_k=FINAL_PARENT_K,
        )

        log_evaluation(
            strategy=(RetrievalStrategy.HYBRID_RERANK_MMR),
            stage=("parent_post_local_rerank_mmr"),
            evaluation_result=(local_mmr_evaluation),
        )
        # ============================================================
        # STAGE 3B
        # HOSTED JINA RERANKING
        # ============================================================

        logger.info("Stage 3B: hosted Jina " "parent reranking")

        cloud_reranked_run = cloud_reranking_runner.run(
            parent_run_result=parent_run,
        )

        cloud_evaluation = evaluate_run(
            evaluator=evaluator,
            gold_dataset=gold_dataset,
            run_result=cloud_reranked_run,
            evaluation_k=FINAL_PARENT_K,
        )

        log_evaluation(
            strategy=(RetrievalStrategy.HYBRID_RERANK),
            stage="parent_post_cloud_rerank",
            evaluation_result=(cloud_evaluation),
        )

        # ============================================================
        # STAGE 4B
        # CLOUD RERANKER -> PARENT MMR
        # ============================================================

        logger.info("Stage 4B: Parent MMR after " "hosted Jina reranking")

        cloud_mmr_run = parent_mmr_runner.run(
            reranked_run_result=(cloud_reranked_run),
        )

        cloud_mmr_evaluation = evaluate_run(
            evaluator=evaluator,
            gold_dataset=gold_dataset,
            run_result=cloud_mmr_run,
            evaluation_k=FINAL_PARENT_K,
        )

        log_evaluation(
            strategy=(RetrievalStrategy.HYBRID_RERANK_MMR),
            stage=("parent_post_cloud_rerank_mmr"),
            evaluation_result=(cloud_mmr_evaluation),
        )

        # ============================================================
        # COMPARISON
        # ============================================================

        def log_comparison(
            *,
            baseline: BenchmarkEvaluationResult,
            local: BenchmarkEvaluationResult,
            cloud: BenchmarkEvaluationResult,
        ) -> None:

            baseline_metrics = baseline.metrics
            local_metrics = local.metrics
            cloud_metrics = cloud.metrics

            logger.info("RERANKER COMPARISON " f"k={baseline.k}")

            logger.info(
                "baseline "
                f"hit={baseline_metrics.mean_hit_at_k:.4f} "
                f"precision="
                f"{baseline_metrics.mean_precision_at_k:.4f} "
                f"recall="
                f"{baseline_metrics.mean_recall_at_k:.4f} "
                f"mrr="
                f"{baseline_metrics.mean_reciprocal_rank:.4f} "
                f"map="
                f"{baseline_metrics.mean_average_precision_at_k:.4f} "
                f"ndcg="
                f"{baseline_metrics.mean_ndcg_at_k:.4f}"
            )

            logger.info(
                "local "
                f"hit={local_metrics.mean_hit_at_k:.4f} "
                f"precision="
                f"{local_metrics.mean_precision_at_k:.4f} "
                f"recall="
                f"{local_metrics.mean_recall_at_k:.4f} "
                f"mrr="
                f"{local_metrics.mean_reciprocal_rank:.4f} "
                f"map="
                f"{local_metrics.mean_average_precision_at_k:.4f} "
                f"ndcg="
                f"{local_metrics.mean_ndcg_at_k:.4f}"
            )

            logger.info(
                "cloud "
                f"hit={cloud_metrics.mean_hit_at_k:.4f} "
                f"precision="
                f"{cloud_metrics.mean_precision_at_k:.4f} "
                f"recall="
                f"{cloud_metrics.mean_recall_at_k:.4f} "
                f"mrr="
                f"{cloud_metrics.mean_reciprocal_rank:.4f} "
                f"map="
                f"{cloud_metrics.mean_average_precision_at_k:.4f} "
                f"ndcg="
                f"{cloud_metrics.mean_ndcg_at_k:.4f}"
            )

            log_comparison(
                baseline=baseline_evaluation,
                local=local_evaluation,
                cloud=cloud_evaluation,
            )

        # # ====================================================
        # # STAGE 3
        # # HOSTED PARENT RERANKING
        # # ====================================================

        # logger.info("Stage 3: hosted parent " "reranking")

        # reranked_run = reranking_runner.run(
        #     parent_run_result=parent_run,
        # )

        # # ====================================================
        # # STAGE 4
        # # RERANKED PARENT EVALUATION
        # # ====================================================

        # logger.info("Stage 4: reranked " "parent evaluation")

        # reranked_evaluation = evaluate_run(
        #     evaluator=evaluator,
        #     gold_dataset=gold_dataset,
        #     run_result=reranked_run,
        #     evaluation_k=FINAL_PARENT_K,
        # )

        # log_evaluation(
        #     strategy=(RetrievalStrategy.HYBRID_RERANK),
        #     stage="parent_post_rerank",
        #     evaluation_result=(reranked_evaluation),
        # )

        # logger.info("Benchmark V2 hosted " "reranker evaluation complete")

    finally:
        parent_repository.close()


if __name__ == "__main__":
    main()
