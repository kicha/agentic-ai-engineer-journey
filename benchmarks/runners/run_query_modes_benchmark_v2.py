from __future__ import annotations

import argparse
import json
import logging
from time import perf_counter

from app.application_factory import build_retrieval_stack
from app.config.settings import (
    QdrantDeployment,
    QueryTransformationMode,
    QueryingConfig,
)
from app.utils.logging_config import configure_logging
from benchmarks.evaluation.benchmark_evaluator import BenchmarkEvaluator
from benchmarks.evaluation.benchmark_models import (
    BenchmarkRunResult,
    QueryRankingResult,
    RankedItem,
    RetrievalStrategy,
)
from benchmarks.evaluation.gold_models import GoldDatasetV2
from benchmarks.evaluation.ranking_evaluator import RetrievalLevel
from benchmarks.paths import GOLD_QUERIES_PATH

logger = logging.getLogger(__name__)

STRATEGY_BY_MODE = {
    QueryTransformationMode.ORIGINAL: RetrievalStrategy.HYBRID_RERANK_MMR,
    QueryTransformationMode.REWRITE: RetrievalStrategy.QUERY_REWRITE,
    QueryTransformationMode.MULTI_QUERY: RetrievalStrategy.MULTI_QUERY,
    QueryTransformationMode.HYDE: RetrievalStrategy.HYDE,
}


def load_gold_dataset() -> GoldDatasetV2:
    return GoldDatasetV2.model_validate(
        json.loads(GOLD_QUERIES_PATH.read_text(encoding="utf-8"))
    )


def run_mode(
    *,
    mode: QueryTransformationMode,
    deployment: QdrantDeployment,
    collection_name: str | None = None,
) -> tuple[BenchmarkRunResult, object]:
    gold_dataset = load_gold_dataset()
    querying_config = QueryingConfig(mode=mode)
    stack = build_retrieval_stack(
        deployment=deployment,
        collection_name=collection_name,
        querying_config=querying_config,
    )

    query_results: list[QueryRankingResult] = []
    try:
        for gold_query in gold_dataset.queries:
            started = perf_counter()
            retrieval_result = stack.orchestrator.retrieve(
                gold_query.query,
                mode=mode,
            )
            latency_ms = (perf_counter() - started) * 1000.0
            items = [
                RankedItem(
                    item_id=result.parent.parent_key,
                    rank=index,
                    score=result.mmr_score,
                    metadata={
                        "parent_id": str(result.parent.parent_id),
                        "document_id": result.parent.document_id,
                        "reranker_rank": result.reranker_rank,
                        "reranker_score": result.reranker_score,
                        "query_similarity": result.query_similarity,
                        "mmr_score": result.mmr_score,
                        "query_mode": mode.value,
                    },
                )
                for index, result in enumerate(retrieval_result.results, start=1)
            ]
            query_results.append(
                QueryRankingResult(
                    query_id=gold_query.query_id,
                    query=gold_query.query,
                    retrieval_level=RetrievalLevel.PARENT,
                    items=items,
                    latency_ms=latency_ms,
                )
            )
            logger.info(
                f"query_id={gold_query.query_id} mode={mode.value} "
                f"parents={len(items)} latency_ms={latency_ms:.2f}"
            )
    finally:
        stack.close()

    run_result = BenchmarkRunResult(
        benchmark_version=gold_dataset.benchmark_version,
        strategy=STRATEGY_BY_MODE[mode],
        retrieval_level=RetrievalLevel.PARENT,
        results=query_results,
    )
    evaluation = BenchmarkEvaluator().evaluate(
        gold_dataset=gold_dataset,
        ranked_ids_by_query_id=run_result.ranked_ids_by_query_id(),
        retrieval_level=RetrievalLevel.PARENT,
        k=querying_config.final_parent_k,
    )
    return run_result, evaluation


def log_evaluation(mode: QueryTransformationMode, evaluation) -> None:
    metrics = evaluation.metrics
    logger.info(
        f"mode={mode.value} k={evaluation.k} "
        f"hit_at_k={metrics.mean_hit_at_k:.4f} "
        f"precision_at_k={metrics.mean_precision_at_k:.4f} "
        f"recall_at_k={metrics.mean_recall_at_k:.4f} "
        f"mrr={metrics.mean_reciprocal_rank:.4f} "
        f"map_at_k={metrics.mean_average_precision_at_k:.4f} "
        f"ndcg_at_k={metrics.mean_ndcg_at_k:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true")
    group.add_argument("--cloud", action="store_true")
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in QueryTransformationMode],
        default=[
            QueryTransformationMode.REWRITE.value,
            QueryTransformationMode.MULTI_QUERY.value,
            QueryTransformationMode.HYDE.value,
        ],
    )
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    deployment = (
        QdrantDeployment.LOCAL if args.local else QdrantDeployment.CLOUD
    )
    for mode_value in args.modes:
        mode = QueryTransformationMode(mode_value)
        _, evaluation = run_mode(
            mode=mode,
            deployment=deployment,
            collection_name=args.collection,
        )
        log_evaluation(mode, evaluation)


if __name__ == "__main__":
    main()
