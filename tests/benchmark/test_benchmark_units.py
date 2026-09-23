import json

from benchmarks.evaluation.gold_models import (
    GoldDatasetV2,
)
from benchmarks.paths import (
    BENCHMARK_UNITS_PATH,
    GOLD_QUERIES_PATH,
)

# ============================================================
# FROZEN BENCHMARK V2 CONTRACT
# ============================================================

EXPECTED_BENCHMARK_VERSION = "2.0"
EXPECTED_DOCUMENT_COUNT = 24
EXPECTED_PARENT_COUNT = 120
EXPECTED_CHILD_COUNT = 196


# ============================================================
# LOADERS
# ============================================================


def load_units() -> dict:
    with BENCHMARK_UNITS_PATH.open(
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
# HELPERS
# ============================================================


def flatten_children(
    units: dict,
) -> list[dict]:

    return [child for parent in units["parents"] for child in parent["children"]]


# ============================================================
# BASIC COUNTS
# ============================================================


def test_units_have_expected_benchmark_version() -> None:
    units = load_units()

    assert units["benchmark_version"] == EXPECTED_BENCHMARK_VERSION


def test_units_have_expected_parent_count() -> None:
    units = load_units()

    assert units["parent_count"] == EXPECTED_PARENT_COUNT

    assert len(units["parents"]) == EXPECTED_PARENT_COUNT


def test_units_have_expected_child_count() -> None:
    units = load_units()

    children = flatten_children(units)

    assert units["child_count"] == EXPECTED_CHILD_COUNT

    assert len(children) == EXPECTED_CHILD_COUNT


def test_units_contain_expected_document_count() -> None:
    units = load_units()

    document_ids = {parent["document_id"] for parent in units["parents"]}

    assert len(document_ids) == EXPECTED_DOCUMENT_COUNT


# ============================================================
# PARENT IDENTITY
# ============================================================


def test_parent_ids_are_unique() -> None:
    units = load_units()

    parent_ids = [parent["parent_id"] for parent in units["parents"]]

    assert len(parent_ids) == len(set(parent_ids))


def test_parent_keys_are_unique() -> None:
    units = load_units()

    parent_keys = [parent["parent_key"] for parent in units["parents"]]

    assert len(parent_keys) == len(set(parent_keys))


# ============================================================
# CHILD IDENTITY
# ============================================================


def test_child_ids_are_unique() -> None:
    units = load_units()

    children = flatten_children(units)

    child_ids = [child["child_id"] for child in children]

    assert len(child_ids) == len(set(child_ids))


def test_child_keys_are_unique() -> None:
    units = load_units()

    children = flatten_children(units)

    child_keys = [child["child_key"] for child in children]

    assert len(child_keys) == len(set(child_keys))


# ============================================================
# PARENT / CHILD LINKAGE
# ============================================================


def test_every_child_matches_its_parent_identity() -> None:
    units = load_units()

    for parent in units["parents"]:
        for child in parent["children"]:

            assert child["parent_id"] == parent["parent_id"]

            assert child["parent_key"] == parent["parent_key"]

            assert child["document_id"] == parent["document_id"]


def test_every_parent_has_at_least_one_child() -> None:
    units = load_units()

    for parent in units["parents"]:
        assert parent["children"]


# ============================================================
# INDEX INTEGRITY
# ============================================================


def test_parent_indexes_are_unique_within_document() -> None:
    units = load_units()

    indexes_by_document: dict[str, list[int]] = {}

    for parent in units["parents"]:
        document_id = parent["document_id"]

        indexes_by_document.setdefault(
            document_id,
            [],
        ).append(parent["parent_index"])

    for indexes in indexes_by_document.values():
        assert len(indexes) == len(set(indexes))


def test_parent_indexes_are_contiguous_within_document() -> None:
    units = load_units()

    indexes_by_document: dict[str, list[int]] = {}

    for parent in units["parents"]:
        document_id = parent["document_id"]

        indexes_by_document.setdefault(
            document_id,
            [],
        ).append(parent["parent_index"])

    for indexes in indexes_by_document.values():
        sorted_indexes = sorted(indexes)

        expected_indexes = list(range(len(sorted_indexes)))

        assert sorted_indexes == expected_indexes


def test_child_indexes_are_contiguous_within_parent() -> None:
    units = load_units()

    for parent in units["parents"]:

        child_indexes = sorted(child["child_index"] for child in parent["children"])

        expected_indexes = list(range(len(child_indexes)))

        assert child_indexes == expected_indexes


# ============================================================
# GOLD → FROZEN UNIT REFERENTIAL INTEGRITY
# ============================================================


def test_all_gold_parent_keys_exist_in_frozen_units() -> None:
    units = load_units()
    gold_dataset = load_gold_dataset()

    actual_parent_keys = {parent["parent_key"] for parent in units["parents"]}

    gold_parent_keys = {
        judgment.parent_key
        for query in gold_dataset.queries
        for judgment in query.parent_judgments
    }

    missing_parent_keys = gold_parent_keys - actual_parent_keys

    assert not missing_parent_keys


def test_all_gold_child_keys_exist_in_frozen_units() -> None:
    units = load_units()
    gold_dataset = load_gold_dataset()

    children = flatten_children(units)

    actual_child_keys = {child["child_key"] for child in children}

    gold_child_keys = {
        judgment.child_key
        for query in gold_dataset.queries
        for judgment in query.child_judgments
    }

    missing_child_keys = gold_child_keys - actual_child_keys

    assert not missing_child_keys
