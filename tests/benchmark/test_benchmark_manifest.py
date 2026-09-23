import hashlib
import json

from benchmarks.evaluation.gold_models import (
    GoldDatasetV2,
)
from benchmarks.paths import (
    BENCHMARK_MANIFEST_PATH,
    GOLD_QUERIES_PATH,
    BENCHMARK_DOCUMENTS_DIR,
)
from app.utils.document_id import (
    extract_document_id,
)
from benchmarks.utils.corpus_fingerprint import (
    compute_corpus_sha256,
)

# ============================================================
# FROZEN BENCHMARK V2 CONTRACT
# ============================================================

EXPECTED_BENCHMARK_VERSION = "2.0"
EXPECTED_DOCUMENT_COUNT = 24
EXPECTED_GOLD_QUERY_COUNT = 32
EXPECTED_TOTAL_CHARACTERS = 16027
EXPECTED_APPROX_WORDS = 2149

EXPECTED_CORPUS_SHA256 = (
    "16c01b5dfb1753d0b55dafdf6164d977" "383c0419c4ebfe8b376189aaf7ef9e62"
)


# ============================================================
# LOADERS
# ============================================================


def load_manifest() -> dict:
    with BENCHMARK_MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_gold_dataset() -> GoldDatasetV2:
    with GOLD_QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return GoldDatasetV2.model_validate(data)


# ============================================================
# MANIFEST IDENTITY
# ============================================================


def test_manifest_has_expected_benchmark_version() -> None:
    manifest = load_manifest()

    assert manifest["benchmark_version"] == EXPECTED_BENCHMARK_VERSION


def test_manifest_has_expected_document_count() -> None:
    manifest = load_manifest()

    assert manifest["document_count"] == EXPECTED_DOCUMENT_COUNT

    assert len(manifest["documents"]) == EXPECTED_DOCUMENT_COUNT


def test_manifest_has_expected_gold_query_count() -> None:
    manifest = load_manifest()

    assert manifest["gold_query_count"] == EXPECTED_GOLD_QUERY_COUNT


def test_manifest_has_expected_corpus_statistics() -> None:
    manifest = load_manifest()

    assert manifest["total_characters"] == EXPECTED_TOTAL_CHARACTERS

    assert manifest["approx_words"] == EXPECTED_APPROX_WORDS


def test_manifest_has_expected_corpus_sha256() -> None:
    manifest = load_manifest()

    assert manifest["corpus_sha256"] == EXPECTED_CORPUS_SHA256


# ============================================================
# DOCUMENT INTEGRITY
# ============================================================


def test_manifest_document_ids_are_unique() -> None:
    manifest = load_manifest()

    document_ids = [document["document_id"] for document in manifest["documents"]]

    assert len(document_ids) == len(set(document_ids))


def test_manifest_document_sources_are_unique() -> None:
    manifest = load_manifest()

    sources = [document["source"] for document in manifest["documents"]]

    assert len(sources) == len(set(sources))


def test_manifest_contains_expected_document_ids() -> None:
    manifest = load_manifest()

    actual_document_ids = {
        document["document_id"] for document in manifest["documents"]
    }

    expected_document_ids = {
        f"DOC-{index:03d}"
        for index in range(
            1,
            EXPECTED_DOCUMENT_COUNT + 1,
        )
    }

    assert actual_document_ids == expected_document_ids


def test_manifest_document_statistics_sum_to_totals() -> None:
    manifest = load_manifest()

    total_characters = sum(document["characters"] for document in manifest["documents"])

    total_words = sum(document["words"] for document in manifest["documents"])

    assert total_characters == manifest["total_characters"]

    assert total_words == manifest["approx_words"]


# ============================================================
# CROSS-ARTIFACT INTEGRITY
# ============================================================


def test_manifest_matches_gold_dataset_identity() -> None:
    manifest = load_manifest()
    gold_dataset = load_gold_dataset()

    assert gold_dataset.benchmark_version == manifest["benchmark_version"]

    assert gold_dataset.corpus_sha256 == manifest["corpus_sha256"]

    assert gold_dataset.query_count == manifest["gold_query_count"]


def test_actual_corpus_sha_matches_frozen_manifest() -> None:
    manifest = load_manifest()

    actual_sha256 = compute_corpus_sha256(BENCHMARK_DOCUMENTS_DIR)

    assert actual_sha256 == manifest["corpus_sha256"]
