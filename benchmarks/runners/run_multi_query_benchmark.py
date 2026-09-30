"""Compatibility entry point for the canonical Benchmark V2 query-mode runner."""

from app.config.settings import QdrantConfig, QueryTransformationMode
from benchmarks.runners.run_query_modes_benchmark_v2 import log_evaluation, run_mode


def main() -> None:
    mode = QueryTransformationMode.MULTI_QUERY
    _, evaluation = run_mode(
        mode=mode,
        deployment=QdrantConfig().deployment,
    )
    log_evaluation(mode, evaluation)


if __name__ == "__main__":
    main()
