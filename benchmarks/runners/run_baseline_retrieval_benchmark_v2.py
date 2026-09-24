from __future__ import annotations

import json
import logging

from app.config.settings import ChunkingConfig, HybridSearchConfig
from app.embeddings.embedding_service import (
    EmbeddingService,
)
from app.embeddings.sparse_embedding_service import SparseEmbeddingService
from app.repositories.parent_repository import ParentRepository
from app.retrieval.hybrid_child_retriever import HybridChildRetriever
from app.utils.logging_config import configure_logging
from app.vectorstore.hybrid_qdrant_store import HybridQdrantStore
from benchmarks.evaluation.benchmark_evaluator import (
    BenchmarkEvaluator,
    BenchmarkEvaluationResult,
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

logger = logging.getLogger(__name__)


# ============================================================
# GOLD DATASET
# ============================================================


def load_gold_dataset() -> GoldDatasetV2:
    """
    Load the frozen Benchmark V2 gold dataset.

    This file contains the manually curated document, parent, and child
    relevance judgments used by the centralized evaluation framework.
    """

    raw_data = json.loads(
        GOLD_QUERIES_PATH.read_text(
            encoding="utf-8",
        )
    )

    return GoldDatasetV2.model_validate(raw_data)


# ============================================================
# RETRIEVAL DEPENDENCIES
# ============================================================


def build_child_retriever() -> HybridChildRetriever:
    """
    Construct the real child retrieval stack.

    This executable entry point is responsible for dependency wiring.
    The reusable benchmark runner remains independently unit-testable.
    """

    config = ChunkingConfig()
    hybrid_search_config = HybridSearchConfig()

    embedding_service = EmbeddingService(
        model_name=config.embedding_model_name,
    )
    sparse_embedding_service = SparseEmbeddingService()
    vector_size = embedding_service.model.get_embedding_dimension()

    vector_store = HybridQdrantStore(
        config=hybrid_search_config,
        vector_size=vector_size,
    )

    return HybridChildRetriever(
        vector_store=vector_store,
        dense_embedding_service=embedding_service,
        sparse_embedding_service=sparse_embedding_service,
    )


def build_parent_repository() -> ParentRepository:
    """
    Construct the authoritative MongoDB parent repository.

    ParentChunks are not reconstructed from Qdrant payloads. The
    benchmark expands retrieved child references into the same
    authoritative ParentChunks used by the production architecture.
    """

    config = ChunkingConfig()

    return ParentRepository(
        config=config,
    )


# ============================================================
# CENTRALIZED EVALUATION
# ============================================================


def evaluate_run(
    evaluator: BenchmarkEvaluator,
    gold_dataset: GoldDatasetV2,
    run_result: BenchmarkRunResult,
    evaluation_k: int,
) -> BenchmarkEvaluationResult:
    """
    Evaluate one standardized benchmark run.

    The same method works for both CHILD and PARENT runs because
    BenchmarkRunResult carries its own retrieval level.
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
    strategy: RetrievalStrategy,
    stage: str,
    evaluation_result: BenchmarkEvaluationResult,
) -> None:
    """
    Log one standardized metric summary.

    Keeping reporting separate from metric calculation prevents the
    executable runner from reimplementing evaluation logic.
    """

    metrics = evaluation_result.metrics

    logger.info(
        f"strategy={strategy.value} "
        f"stage={stage} "
        f"k={evaluation_result.k} "
        f"hit_at_k={metrics.mean_hit_at_k:.4f} "
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
# ONE STRATEGY
# ============================================================


def run_strategy(
    child_runner: BaselineRetrievalRunner,
    parent_runner: BaselineParentExpansionRunner,
    evaluator: BenchmarkEvaluator,
    gold_dataset: GoldDatasetV2,
    strategy: RetrievalStrategy,
    child_evaluation_k: int,
    parent_evaluation_k: int,
) -> None:
    """
    Run and evaluate one retrieval strategy at two stages.

    Stage 1:
        child retrieval
        -> child-level diagnostic metrics

    Stage 2:
        same child results
        -> parent expansion/deduplication
        -> parent-level primary context metrics

    No second retrieval is executed for parent evaluation.
    """

    logger.info(f"Starting benchmark strategy={strategy.value}")

    # --------------------------------------------------------
    # CHILD RETRIEVAL
    # --------------------------------------------------------

    child_run_result = child_runner.run(
        gold_dataset=gold_dataset,
        strategy=strategy,
    )

    child_evaluation = evaluate_run(
        evaluator=evaluator,
        gold_dataset=gold_dataset,
        run_result=child_run_result,
        evaluation_k=child_evaluation_k,
    )

    log_evaluation(
        strategy=strategy,
        stage="child",
        evaluation_result=child_evaluation,
    )

    # --------------------------------------------------------
    # PARENT EXPANSION
    #
    # IMPORTANT:
    # This uses the SAME child results produced above.
    # --------------------------------------------------------

    parent_run_result = parent_runner.expand(
        child_run_result=child_run_result,
    )

    parent_evaluation = evaluate_run(
        evaluator=evaluator,
        gold_dataset=gold_dataset,
        run_result=parent_run_result,
        evaluation_k=parent_evaluation_k,
    )

    log_evaluation(
        strategy=strategy,
        stage="parent",
        evaluation_result=parent_evaluation,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Execute the Benchmark V2 baseline retrieval comparison.

    Strategies:
        Dense
        Sparse
        Hybrid

    Each strategy is evaluated at two levels:

        CHILD
            candidate-generation quality

        PARENT
            recovered context quality

    CrossEncoder and Parent MMR are intentionally NOT included in this
    baseline benchmark.
    """

    configure_logging()

    gold_dataset = load_gold_dataset()

    child_retriever = build_child_retriever()

    parent_repository = build_parent_repository()

    child_runner = BaselineRetrievalRunner(
        retriever=child_retriever,
        child_candidate_limit=10,
    )

    parent_runner = BaselineParentExpansionRunner(
        parent_repository=parent_repository,
        parent_candidate_limit=5,
    )

    evaluator = BenchmarkEvaluator()

    # --------------------------------------------------------
    # Child metrics are diagnostic.
    #
    # Parent metrics are the more important measure because the
    # ParentChunk is what ultimately becomes RAG context.
    # --------------------------------------------------------

    child_evaluation_k = 5
    parent_evaluation_k = 3

    try:

        for strategy in (
            RetrievalStrategy.DENSE,
            RetrievalStrategy.SPARSE,
            RetrievalStrategy.HYBRID,
        ):
            run_strategy(
                child_runner=child_runner,
                parent_runner=parent_runner,
                evaluator=evaluator,
                gold_dataset=gold_dataset,
                strategy=strategy,
                child_evaluation_k=(child_evaluation_k),
                parent_evaluation_k=(parent_evaluation_k),
            )

    finally:

        # Keep resource cleanup at the executable boundary.
        parent_repository.close()


if __name__ == "__main__":
    main()
