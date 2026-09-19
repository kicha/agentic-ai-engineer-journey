from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# BENCHMARK ROOT
# ============================================================

BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"


# ============================================================
# BENCHMARK DATA
# ============================================================

BENCHMARK_DATA_DIR = BENCHMARK_ROOT / "data"
BENCHMARK_DOCUMENTS_DIR = BENCHMARK_DATA_DIR / "documents"
BENCHMARK_FROZEN_DIR = BENCHMARK_DATA_DIR / "frozen"
BENCHMARK_WORK_DIR = BENCHMARK_DATA_DIR / "work"


# ============================================================
# FROZEN BENCHMARK ARTIFACTS
# ============================================================

BENCHMARK_MANIFEST_PATH = BENCHMARK_FROZEN_DIR / "manifest.json"
BENCHMARK_UNITS_PATH = BENCHMARK_FROZEN_DIR / "benchmark_v2_units.json"
GOLD_QUERIES_PATH = BENCHMARK_FROZEN_DIR / "gold_queries_v2.json"


# ============================================================
# WORKING / AUTHORING ARTIFACTS
# ============================================================

GOLD_QUERIES_DRAFT_PATH = BENCHMARK_WORK_DIR / "gold_queries_v2_draft.json"
ANNOTATION_CANDIDATES_JSON_PATH = (
    BENCHMARK_WORK_DIR / "gold_annotation_candidates_v2.json"
)
ANNOTATION_CANDIDATES_MARKDOWN_PATH = (
    BENCHMARK_WORK_DIR / "gold_annotation_candidates_v2.md"
)


# ============================================================
# RESULTS
# ============================================================

BENCHMARK_RESULTS_DIR = BENCHMARK_ROOT / "results"
