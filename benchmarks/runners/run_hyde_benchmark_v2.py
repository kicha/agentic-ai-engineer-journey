"""Canonical Benchmark V2 HyDE runner."""

from app.config.settings import QdrantConfig, QueryTransformationMode
from benchmarks.runners.run_query_modes_benchmark_v2 import log_evaluation, run_mode


def main() -> None:
    mode = QueryTransformationMode.HYDE
    _, evaluation = run_mode(
        mode=mode,
        deployment=QdrantConfig().deployment,
    )
    log_evaluation(mode, evaluation)


if __name__ == "__main__":
    main()
