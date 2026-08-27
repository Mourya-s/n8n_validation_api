"""
Tests for CatalogValidator (validators/catalog_validator.py).

All Databricks calls are mocked here so none of these tests require a
live Databricks environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from table_validator.validators.catalog_validator import CatalogValidator
from table_validator.config.schema import ValidationType
from table_validator.models import (
    CatalogValidationRequest,
    DataCompareMode,
    ValidationStatus,
    ValidationTier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _schema_df(columns):
    """columns: list of (name, data_type, is_nullable)"""
    return pd.DataFrame(
        [
            {
                "column_name": c[0],
                "data_type": c[1],
                "is_nullable": c[2],
                "ordinal_position": i + 1,
            }
            for i, c in enumerate(columns)
        ]
    )


def _make_connector(**overrides) -> MagicMock:
    mock = MagicMock()
    mock.catalog_exists.return_value = True
    mock.get_schemas.return_value = ["bronze"]
    mock.get_tables.return_value = ["customers"]
    mock.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "string", True)]
    )
    mock.get_row_count.return_value = 100
    mock.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
        "name": {"null_count": 2, "distinct_count": 95, "min": None, "max": None},
    }
    mock.is_min_max_eligible.side_effect = lambda dt: dt.lower().startswith("int")
    # Default fingerprint: identical on both sides, so a fully-matching
    # fixture legitimately stops at Tier 2 (no row-hash SQL) - matching
    # the fail-fast funnel's real behavior. Tests that need to reach
    # Tier 4 must override get_table_fingerprint to disagree.
    mock.get_table_fingerprint.return_value = {
        "row_count": 100, "hash_sum": 12345, "hash_xor": 67890,
    }
    mock.get_row_hashes_by_row_number.side_effect = lambda catalog, schema, table, cols, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")], key="row_number")
    )
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def _mismatched_fingerprint_connector(**overrides) -> MagicMock:
    """A connector fixture whose fingerprint always disagrees between
    source and target, forcing the funnel past Tier 2 into Tier 4 - for
    tests that need to exercise row-hash diff logic."""
    def fingerprint_side_effect(catalog, schema, table, columns, spec=None):
        return (
            {"row_count": 100, "hash_sum": 111, "hash_xor": 222}
            if catalog == "cat_source"
            else {"row_count": 100, "hash_sum": 999, "hash_xor": 888}
        )
    overrides.setdefault("get_table_fingerprint", MagicMock(side_effect=fingerprint_side_effect))
    return _make_connector(**overrides)


def _request(**overrides) -> CatalogValidationRequest:
    defaults = dict(source_catalog="cat_source", target_catalog="cat_target")
    defaults.update(overrides)
    return CatalogValidationRequest(**defaults)


# ---------------------------------------------------------------------------
# Stage 1: catalog exists / missing
# ---------------------------------------------------------------------------
def test_catalog_exists_both_sides_pass_through():
    connector = _make_connector()
    validator = CatalogValidator(connector)
    result = validator.compare_catalogs(_request())
    assert result.status in (ValidationStatus.PASS, ValidationStatus.FAIL)  # proceeds past stage 1


def test_source_catalog_missing_fails_fast():
    connector = _make_connector()
    connector.catalog_exists.side_effect = lambda name: name != "cat_source"
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    assert result.status == ValidationStatus.FAIL
    assert "cat_source" in result.error
    connector.get_schemas.assert_not_called()


def test_target_catalog_missing_fails_fast():
    connector = _make_connector()
    connector.catalog_exists.side_effect = lambda name: name != "cat_target"
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    assert result.status == ValidationStatus.FAIL
    assert "cat_target" in result.error


# ---------------------------------------------------------------------------
# Stage 2: schema missing
# ---------------------------------------------------------------------------
def test_missing_schema_in_target_reported_but_does_not_crash():
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "gold"] if catalog == "cat_source" else ["bronze"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    assert "gold" in result.missing_schemas
    assert result.status == ValidationStatus.FAIL
    # bronze (the common schema) should still have been validated
    assert any(s.schema_name == "bronze" for s in result.schemas)


# ---------------------------------------------------------------------------
# Stage 3/4: table missing / extra
# ---------------------------------------------------------------------------
def test_missing_and_extra_tables_reported():
    connector = _make_connector()
    connector.get_tables.side_effect = lambda catalog, schema: (
        ["customers", "orders"] if catalog == "cat_source" else ["customers", "employees"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    schema_result = result.schemas[0]
    assert schema_result.missing_tables == ["orders"]
    assert schema_result.extra_tables == ["employees"]
    # only the common table (customers) gets detailed validation
    assert [t.table for t in schema_result.tables] == ["customers"]


def test_explicit_table_scope_excludes_unrelated_missing_and_extra_tables():
    """Regression test: when the user explicitly names one table to
    compare (request.tables=["customers"]), an unrelated table difference
    elsewhere in the same schema (e.g. "orders" missing, "employees"
    extra) must NOT be reported as missing/extra and must NOT fail the
    targeted run - the user never asked to compare those tables. Before
    this fix, missing_tables/extra_tables were computed across the whole
    schema regardless of request.tables, so an unrelated table on either
    side falsely failed a run scoped to a single named table."""
    connector = _make_connector()
    connector.get_tables.side_effect = lambda catalog, schema: (
        ["customers", "orders"] if catalog == "cat_source" else ["customers", "employees"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(tables=["customers"]))
    schema_result = result.schemas[0]

    assert schema_result.missing_tables == []
    assert schema_result.extra_tables == []
    assert [t.table for t in schema_result.tables] == ["customers"]
    assert result.status != ValidationStatus.FAIL or schema_result.tables[0].status == ValidationStatus.FAIL


def test_explicit_schema_scope_excludes_unrelated_missing_and_extra_schemas():
    """Same regression as above, one level up: request.schemas=["bronze"]
    must not let an unrelated schema difference elsewhere in the catalog
    (e.g. "gold" missing, "silver" extra) fail this targeted run."""
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "gold"] if catalog == "cat_source" else ["bronze", "silver"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(schemas=["bronze"]))

    assert result.missing_schemas == []
    assert result.extra_schemas == []
    assert [s.schema_name for s in result.schemas] == ["bronze"]


# ---------------------------------------------------------------------------
# Stage 5/6: missing / extra columns
# ---------------------------------------------------------------------------
def test_missing_column_detected():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False), ("age", "int", True)])
        if catalog == "cat_source"
        else _schema_df([("id", "int", False)])
    )
    connector.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 100, "min": 1, "max": 100},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.missing_columns == ["age"]
    assert table.columns_status == ValidationStatus.FAIL
    assert table.status == ValidationStatus.FAIL
    # BLOCKING: Tier 0 aborts immediately, no further tier runs.
    assert table.schema_blocking is True
    assert table.tier_reached == ValidationTier.SCHEMA_BLOCKED
    connector.get_row_count.assert_not_called()
    connector.get_table_fingerprint.assert_not_called()


def test_extra_column_detected():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False)])
        if catalog == "cat_source"
        else _schema_df([("id", "int", False), ("phone", "string", True)])
    )
    connector.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 100, "min": 1, "max": 100},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.extra_columns == ["phone"]
    assert table.columns_status == ValidationStatus.FAIL
    # BLOCKING: Tier 0 aborts immediately, no further tier runs.
    assert table.schema_blocking is True
    assert table.tier_reached == ValidationTier.SCHEMA_BLOCKED
    connector.get_row_count.assert_not_called()
    connector.get_table_fingerprint.assert_not_called()


# ---------------------------------------------------------------------------
# Tier 0: data type mismatch - same-family widening is NON-BLOCKING
# (reported, but does not abort the table); cross-family is BLOCKING.
# ---------------------------------------------------------------------------
def test_datatype_widening_within_family_is_non_blocking():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("age", "int", True)])
        if catalog == "cat_source"
        else _schema_df([("age", "bigint", True)])
    )
    connector.get_column_statistics.return_value = {
        "age": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 90},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]
    col = table.columns[0]

    # Same-family widening (int -> bigint) is a real, reportable
    # difference but must not be BLOCKING - execution continues past
    # Tier 0 into Tier 1+ (fingerprint gets called).
    assert col.data_type_status == ValidationStatus.PASS
    assert col.source_data_type == "int"
    assert col.target_data_type == "bigint"
    assert table.schema_blocking is False
    connector.get_table_fingerprint.assert_called()


def test_datatype_cross_family_change_is_blocking():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("age", "int", True)])
        if catalog == "cat_source"
        else _schema_df([("age", "string", True)])
    )
    connector.get_column_statistics.return_value = {
        "age": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 90},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.schema_blocking is True
    assert table.tier_reached == ValidationTier.SCHEMA_BLOCKED
    assert table.status == ValidationStatus.FAIL
    connector.get_row_count.assert_not_called()
    connector.get_table_fingerprint.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 8: nullable mismatch
# ---------------------------------------------------------------------------
def test_nullable_mismatch_detected():
    """NON-BLOCKING contract: a nullable-only difference is recorded but
    must NOT abort the table - execution continues into Tier 1+ (the
    spec's explicit acceptance criterion: reported as a schema mismatch,
    but the funnel still verifies row-level data on its own)."""
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False)])
        if catalog == "cat_source"
        else _schema_df([("id", "int", True)])
    )
    connector.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 100, "min": 1, "max": 100},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]
    col = table.columns[0]

    assert col.nullable_status == ValidationStatus.FAIL
    assert table.schema_blocking is False
    assert table.tier_reached != ValidationTier.SCHEMA_BLOCKED
    connector.get_table_fingerprint.assert_called()


def test_nullable_check_can_be_disabled():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False)])
        if catalog == "cat_source"
        else _schema_df([("id", "int", True)])
    )
    connector.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 100, "min": 1, "max": 100},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(validate_nullable=False))
    col = result.schemas[0].tables[0].columns[0]

    assert col.nullable_status == ValidationStatus.SKIPPED


# ---------------------------------------------------------------------------
# Stage 9: column order
# ---------------------------------------------------------------------------
def test_column_order_mismatch_detected_when_enabled():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False), ("name", "string", True), ("age", "int", True)])
        if catalog == "cat_source"
        else _schema_df([("id", "int", False), ("age", "int", True), ("name", "string", True)])
    )
    connector.get_column_statistics.return_value = {
        c: {"null_count": 0, "distinct_count": 1, "min": None, "max": None}
        for c in ("id", "name", "age")
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(validate_column_order=True))
    table = result.schemas[0].tables[0]

    assert table.column_order_status == ValidationStatus.FAIL


def test_column_order_ignored_when_disabled():
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False), ("name", "string", True)])
        if catalog == "cat_source"
        else _schema_df([("name", "string", True), ("id", "int", False)])
    )
    connector.get_column_statistics.return_value = {
        c: {"null_count": 0, "distinct_count": 1, "min": None, "max": None}
        for c in ("id", "name")
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(validate_column_order=False))
    table = result.schemas[0].tables[0]

    assert table.column_order_status == ValidationStatus.SKIPPED
    # a table shouldn't fail solely because of order when disabled
    assert table.status != ValidationStatus.FAIL or table.row_count_status == ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Stage 10: row count mismatch
# ---------------------------------------------------------------------------
def test_row_count_mismatch_detected():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_count.side_effect = lambda catalog, schema, table: (
        1000 if catalog == "cat_source" else 950
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.row_count_source == 1000
    assert table.row_count_target == 950
    assert table.row_count_difference == -50
    assert table.row_count_status == ValidationStatus.FAIL
    # A Tier 1 mismatch is no longer terminal - the funnel keeps going so
    # the exact differing row(s) can still be located.
    connector.get_table_fingerprint.assert_called()
    connector.get_row_hashes_by_row_number.assert_called()
    assert table.tier_reached in (ValidationTier.ROW_HASH, ValidationTier.COLUMN_DIFF)


# ---------------------------------------------------------------------------
# Stage 11: null count mismatch
# ---------------------------------------------------------------------------
def test_null_count_mismatch_detected():
    connector = _make_connector()
    connector.get_column_statistics.side_effect = lambda catalog, schema, table, cols, mm: (
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 100, "distinct_count": 90, "min": None, "max": None}}
        if catalog == "cat_source"
        else
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 120, "distinct_count": 90, "min": None, "max": None}}
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    name_col = next(c for c in result.schemas[0].tables[0].columns if c.column == "name")

    assert name_col.null_count_status == ValidationStatus.FAIL
    assert name_col.source_null_count == 100
    assert name_col.target_null_count == 120


# ---------------------------------------------------------------------------
# Stage 12: distinct count mismatch
# ---------------------------------------------------------------------------
def test_distinct_count_mismatch_detected():
    connector = _make_connector()
    connector.get_column_statistics.side_effect = lambda catalog, schema, table, cols, mm: (
        {"id": {"null_count": 0, "distinct_count": 15, "min": None, "max": None},
         "name": {"null_count": 0, "distinct_count": 15, "min": None, "max": None}}
        if catalog == "cat_source"
        else
        {"id": {"null_count": 0, "distinct_count": 17, "min": None, "max": None},
         "name": {"null_count": 0, "distinct_count": 15, "min": None, "max": None}}
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    id_col = next(c for c in result.schemas[0].tables[0].columns if c.column == "id")

    assert id_col.distinct_count_status == ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Stage 13: min/max mismatch
# ---------------------------------------------------------------------------
def test_min_max_mismatch_detected_for_eligible_columns():
    connector = _make_connector()
    connector.get_table_schema.return_value = _schema_df([("amount", "int", True)])
    connector.is_min_max_eligible.return_value = True
    connector.get_column_statistics.side_effect = lambda catalog, schema, table, cols, mm: (
        {"amount": {"null_count": 0, "distinct_count": 5, "min": 10, "max": 50000}}
        if catalog == "cat_source"
        else
        {"amount": {"null_count": 0, "distinct_count": 5, "min": 10, "max": 49000}}
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    col = result.schemas[0].tables[0].columns[0]

    assert col.min_max_status == ValidationStatus.FAIL
    assert col.source_max == 50000
    assert col.target_max == 49000


# ---------------------------------------------------------------------------
# Successful / failed full-table validation
# ---------------------------------------------------------------------------
def test_fully_matching_table_passes():
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.status == ValidationStatus.PASS
    assert result.status == ValidationStatus.PASS


def test_failed_table_marks_overall_catalog_as_fail():
    connector = _make_connector()
    connector.get_row_count.side_effect = lambda catalog, schema, table: (
        100 if catalog == "cat_source" else 90
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    assert result.schemas[0].tables[0].status == ValidationStatus.FAIL
    assert result.status == ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Stage 18: technical error handling - one bad table doesn't crash the run
# ---------------------------------------------------------------------------
def test_technical_error_on_one_table_does_not_crash_others():
    connector = _make_connector()
    connector.get_tables.return_value = ["customers", "orders"]

    def schema_side_effect(catalog, schema, table):
        if table == "orders":
            raise RuntimeError("Permission denied")
        return _schema_df([("id", "int", False)])

    connector.get_table_schema.side_effect = schema_side_effect
    connector.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 1, "min": None, "max": None},
    }
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    tables_by_name = {t.table: t for t in result.schemas[0].tables}

    assert tables_by_name["orders"].status == ValidationStatus.ERROR
    assert "Permission denied" in tables_by_name["orders"].error
    assert tables_by_name["customers"].status == ValidationStatus.PASS
    assert result.status == ValidationStatus.ERROR


def test_catalog_existence_check_failure_returns_error_not_exception():
    connector = _make_connector()
    connector.catalog_exists.side_effect = RuntimeError("connection dropped")
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())

    assert result.status == ValidationStatus.ERROR
    assert "connection dropped" in result.error


# ---------------------------------------------------------------------------
# calculate_overall_status - pure logic, no mocking needed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([ValidationStatus.PASS, ValidationStatus.PASS], ValidationStatus.PASS),
        ([ValidationStatus.PASS, ValidationStatus.FAIL], ValidationStatus.FAIL),
        ([ValidationStatus.PASS, ValidationStatus.ERROR], ValidationStatus.ERROR),
        ([ValidationStatus.FAIL, ValidationStatus.ERROR], ValidationStatus.ERROR),
        ([ValidationStatus.SKIPPED, ValidationStatus.SKIPPED], ValidationStatus.SKIPPED),
        ([ValidationStatus.PASS, ValidationStatus.SKIPPED], ValidationStatus.PASS),
        ([], ValidationStatus.SKIPPED),
    ],
)
def test_calculate_overall_status(statuses, expected):
    assert CatalogValidator.calculate_overall_status(statuses) == expected


# ---------------------------------------------------------------------------
# _classify_type_family - pure logic, no mocking needed. Same-family
# widening is NON-BLOCKING (PASS); cross-family is BLOCKING (FAIL).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source_type,target_type,expected",
    [
        ("int", "int", ValidationStatus.PASS),
        ("int", "bigint", ValidationStatus.PASS),
        ("tinyint", "bigint", ValidationStatus.PASS),
        ("decimal(10,2)", "double", ValidationStatus.PASS),
        ("float", "decimal(38,0)", ValidationStatus.PASS),
        ("varchar(50)", "string", ValidationStatus.PASS),
        ("date", "timestamp", ValidationStatus.PASS),
        ("string", "int", ValidationStatus.FAIL),
        ("int", "string", ValidationStatus.FAIL),
        ("boolean", "int", ValidationStatus.FAIL),
        ("string", "date", ValidationStatus.FAIL),
        ("binary", "string", ValidationStatus.FAIL),
    ],
)
def test_classify_type_family(source_type, target_type, expected):
    assert CatalogValidator._classify_type_family(source_type, target_type) == expected


# ---------------------------------------------------------------------------
# Tier 4: duplicate-key detection (a configured key that isn't actually
# unique must be flagged, not silently collapsed to its last occurrence).
# ---------------------------------------------------------------------------
def test_duplicate_key_detected_in_row_hash_comparison():
    source_hashes = _hash_df([(1, "aaa"), (1, "bbb"), (2, "ccc")])
    target_hashes = _hash_df([(1, "aaa"), (2, "ccc")])

    mismatches, count, pct = CatalogValidator.compare_row_hashes(
        source_hashes, target_hashes, ["id"]
    )

    assert count == 1
    assert mismatches[0].status == "DUPLICATE_KEY"
    assert mismatches[0].primary_key == "1"


# ---------------------------------------------------------------------------
# Tier 3: partition candidate selection (pure logic, no mocks needed).
# ---------------------------------------------------------------------------
def test_partition_candidates_excludes_primary_key_and_sorts_alphabetically():
    candidates = CatalogValidator._partition_candidates(
        ["id", "region", "amount", "created_at"], key_columns=["id"],
    )
    assert candidates == ["amount", "created_at", "region"]


def test_partition_candidates_with_no_key_configured_offers_all_columns():
    candidates = CatalogValidator._partition_candidates(
        ["id", "region"], key_columns=None,
    )
    assert candidates == ["id", "region"]


# ---------------------------------------------------------------------------
# Tier 3: partitioned Tier 4 - large table + confirmed mismatch triggers
# the partition_prompt callback; bucket fingerprints eliminate matching
# buckets so only culprit buckets get row-hash-scanned.
# ---------------------------------------------------------------------------
def _bucket_fingerprint_df(rows):
    """rows: list of (bucket_value, row_count, hash_sum, hash_xor)"""
    return pd.DataFrame(
        [{"bucket_value": r[0], "row_count": r[1], "hash_sum": r[2], "hash_xor": r[3]} for r in rows]
    )


def test_large_mismatched_table_invokes_partition_prompt():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_count.return_value = 2_000_000  # over the default threshold
    connector.get_table_fingerprint_by_bucket.return_value = _bucket_fingerprint_df(
        [("east", 1_000_000, 111, 222), ("west", 1_000_000, 111, 222)]
    )
    prompt = MagicMock(return_value=None)  # user declines
    validator = CatalogValidator(connector, partition_prompt=prompt)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    prompt.assert_called_once()
    context = prompt.call_args[0][0]
    assert context.schema_name == "bronze"
    assert context.table == "customers"
    assert context.row_count == 2_000_000
    assert "id" in context.candidate_columns
    # User declined -> falls back to unpartitioned Tier 4, exactly as today.
    assert table.partitioned is False
    assert table.partition_skip_reason == "user declined or non-interactive run"
    connector.get_row_hashes_by_row_number.assert_called()


def test_small_mismatched_table_never_invokes_partition_prompt():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_count.return_value = 100  # well under the default threshold
    prompt = MagicMock(return_value="region")
    validator = CatalogValidator(connector, partition_prompt=prompt)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    prompt.assert_not_called()
    assert table.partitioned is False
    assert table.partition_skip_reason is None


def test_matching_bucket_is_never_row_hash_scanned():
    """The core performance guarantee: a bucket whose fingerprint matches
    between source and target must never be passed to get_row_hashes* -
    only the culprit bucket's data actually gets scanned."""
    connector = _mismatched_fingerprint_connector()
    connector.get_row_count.return_value = 2_000_000
    connector.get_table_fingerprint_by_bucket.return_value = _bucket_fingerprint_df(
        [("east", 1_000_000, 111, 222), ("west", 1_000_000, 111, 222)]
    )

    def fingerprint_by_bucket_side_effect(catalog, schema, table, columns, bucket_column):
        if catalog == "cat_source":
            return _bucket_fingerprint_df([("east", 1_000_000, 111, 222), ("west", 1_000_000, 333, 444)])
        return _bucket_fingerprint_df([("east", 1_000_000, 111, 222), ("west", 1_000_000, 999, 888)])

    connector.get_table_fingerprint_by_bucket.side_effect = fingerprint_by_bucket_side_effect
    connector.get_row_hashes_by_row_number.side_effect = lambda catalog, schema, table, cols, bucket_predicate=None: (
        _hash_df([(1, "aaa")], key="row_number")
    )
    prompt = MagicMock(return_value="region")
    validator = CatalogValidator(connector, partition_prompt=prompt)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.partitioned is True
    assert table.partition_column == "region"
    assert table.partition_buckets_total == 2
    assert table.partition_buckets_culprit == 1  # only "west" differs

    scanned_buckets = [
        call.kwargs.get("bucket_predicate")
        for call in connector.get_row_hashes_by_row_number.call_args_list
    ]
    assert all(bp is not None and bp[1] == "west" for bp in scanned_buckets)
    assert not any(bp[1] == "east" for bp in scanned_buckets)


def test_partition_prompt_callback_exception_falls_back_to_unpartitioned():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_count.return_value = 2_000_000
    prompt = MagicMock(side_effect=RuntimeError("prompt crashed"))
    validator = CatalogValidator(connector, partition_prompt=prompt)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.partitioned is False
    connector.get_row_hashes_by_row_number.assert_called()
    connector.get_table_fingerprint_by_bucket.assert_not_called()


# ---------------------------------------------------------------------------
# Data compare mode: default STATISTICS mode skips row-level compare
# ---------------------------------------------------------------------------
def test_matching_fingerprint_stops_before_row_hash_by_default():
    """Core regression guard for the fail-fast funnel: when Tier 2's
    whole-table fingerprint matches (the _make_connector() default), the
    funnel must stop there - no row-hash SQL of any kind runs. This is
    the direct fix for the production timeout the tiered rewrite exists
    to prevent (a table with no primary key no longer falls into the
    expensive ROW_NUMBER() fallback just because ROW validation is on)."""
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.tier_reached == ValidationTier.FINGERPRINT
    connector.get_row_hashes.assert_not_called()
    connector.get_row_hashes_by_row_number.assert_not_called()
    connector.key_based_row_diff.assert_not_called()
    assert table.data.status == ValidationStatus.PASS
    assert table.status == ValidationStatus.PASS


def test_mismatched_fingerprint_falls_back_to_row_number_when_no_key_configured():
    connector = _mismatched_fingerprint_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.data.mode == DataCompareMode.STATISTICS
    connector.key_based_row_diff.assert_not_called()
    # No primary key configured -> the row-hash stage falls back to a
    # ROW_NUMBER()-based comparison rather than skipping entirely.
    connector.get_row_hashes_by_row_number.assert_called()
    assert table.data.key_columns == ["row_number"]
    assert "row_number" in table.data.note.lower() if table.data.note else True


def test_full_mode_without_key_uses_row_number_fallback():
    connector = _mismatched_fingerprint_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(data_compare_mode=DataCompareMode.FULL)
    )
    table = result.schemas[0].tables[0]

    assert table.data.key_columns == ["row_number"]
    assert table.data.note is not None
    assert "row_number" in table.data.note.lower() or "row-number" in table.data.note.lower()


def test_stats_only_mode_stops_before_fingerprint_even_on_match():
    """--mode=stats (max_tier=STATISTICAL): the funnel must stop after
    Tier 1 even when statistics match - Tier 2's fingerprint (and
    therefore Tier 4/5) must never run. This closes a real gap in the
    pre-tiered pipeline, where STATISTICS mode only skipped compare_data
    but the always-on row-hash stage still ran regardless."""
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(max_tier=ValidationTier.STATISTICAL))
    table = result.schemas[0].tables[0]

    assert table.tier_reached == ValidationTier.STATISTICAL
    connector.get_table_fingerprint.assert_not_called()
    connector.get_row_hashes_by_row_number.assert_not_called()


def test_tier1_mismatch_drills_down_to_exact_row_in_full_mode():
    """Core regression guard: a Tier 1 statistical mismatch (distinct
    count differs here, mirroring a real report the user hit - e.g.
    'first_name' has 21 distinct values in source vs 22 in target) must
    no longer leave Data Mismatches/Row Hash Mismatches empty. In the
    default (--mode=full) ceiling, the funnel now continues past Tier 1
    into Tier 2/4/5 so the exact differing row lands in
    sample_changed_detail, not just a bare statistic."""
    connector = _mismatched_fingerprint_connector()
    connector.get_column_statistics.side_effect = lambda catalog, schema, table, cols, mm: (
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 2, "distinct_count": 21, "min": None, "max": None}}
        if catalog == "cat_source"
        else
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 2, "distinct_count": 22, "min": None, "max": None}}
    )
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")])
        if catalog == "cat_source"
        else _hash_df([(1, "aaa"), (2, "zzz")])
    )
    connector.get_row_detail_for_keys.return_value = [
        {
            "key": {"id": 2},
            "mismatched_columns": ["name"],
            "source_values": {"name": "Old Name"},
            "target_values": {"name": "New Name"},
            "source_row_hash": "bbb",
            "target_row_hash": "zzz",
        }
    ]
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]
    name_col = next(c for c in table.columns if c.column == "name")

    assert name_col.distinct_count_status == ValidationStatus.FAIL
    # The funnel kept going - Tier 2/4/5 all ran.
    connector.get_table_fingerprint.assert_called()
    connector.get_row_hashes.assert_called()
    connector.get_row_detail_for_keys.assert_called_once()
    assert table.tier_reached == ValidationTier.COLUMN_DIFF
    assert table.status == ValidationStatus.FAIL
    # The exact row/column/values are now in the report, not empty.
    assert len(table.data.sample_changed_detail) == 1
    detail = table.data.sample_changed_detail[0]
    assert detail.mismatch_column == "name"
    assert detail.source_value == "Old Name"
    assert detail.target_value == "New Name"
    assert table.data.row_hash_mismatch_count == 1


def test_tier1_mismatch_stays_fail_even_if_fingerprint_and_row_hash_find_nothing():
    """Edge case: Tier 1 confirms a real difference (e.g. a min/max-only
    finding on a column excluded from hashing), but the fingerprint
    happens to match and the row-hash join finds zero mismatched keys
    (both use _make_connector()'s identical default fixture data). The
    table must stay FAIL - Tier 1's confirmed finding must never be
    silently overridden by a downstream tier that simply didn't detect
    the same difference through a different lens."""
    connector = _make_connector()
    connector.get_column_statistics.side_effect = lambda catalog, schema, table, cols, mm: (
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 2, "distinct_count": 95, "min": None, "max": None}}
        if catalog == "cat_source"
        else
        {"id": {"null_count": 0, "distinct_count": 100, "min": None, "max": None},
         "name": {"null_count": 5, "distinct_count": 95, "min": None, "max": None}}
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]
    name_col = next(c for c in table.columns if c.column == "name")

    assert name_col.null_count_status == ValidationStatus.FAIL
    # Fingerprint matched (default fixture) but Tier 1 already confirmed
    # a difference, so Tier 4 still ran...
    connector.get_row_hashes.assert_called()
    # ...and even though Tier 4 found nothing, the table stays FAIL.
    assert table.data.status == ValidationStatus.FAIL
    assert table.status == ValidationStatus.FAIL


# ---------------------------------------------------------------------------
# Row-hash comparison stage (CatalogValidator.compare_row_hashes /
# DatabricksConnector.get_row_hashes) - separate mechanism from
# key_based_row_diff/_changed_row_detail (FULL mode), primary way to detect
# row-level mismatches whenever a primary key is configured, independent
# of data_compare_mode.
# ---------------------------------------------------------------------------
def _hash_df(rows, key="id"):
    """rows: list of (key_value, row_hash)"""
    return pd.DataFrame([{key: r[0], "row_hash": r[1]} for r in rows])


def test_row_hashes_matching_produces_no_mismatch():
    # Fingerprint mismatched (forces Tier 4 to run) but the actual
    # per-key row hashes match - Tier 4 itself must report PASS.
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")])
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]

    assert table.data.row_hash_mismatches == []
    assert table.data.row_hash_mismatch_count == 0
    assert table.data.row_hash_mismatch_percentage == 0.0
    assert table.data.status == ValidationStatus.PASS


def test_row_hashes_differing_for_shared_key_is_mismatch():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")])
        if catalog == "cat_source"
        else _hash_df([(1, "aaa"), (2, "zzz")])
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]

    assert table.data.row_hash_mismatch_count == 1
    mismatch = table.data.row_hash_mismatches[0]
    assert mismatch.primary_key == "2"
    assert mismatch.source_hash == "bbb"
    assert mismatch.target_hash == "zzz"
    assert mismatch.status == "MISMATCH"
    assert table.data.row_hash_mismatch_percentage == 50.0
    assert table.data.status == ValidationStatus.FAIL


def test_tier5_column_diff_runs_for_mismatched_key_and_names_exact_column():
    """When Tier 4 finds a real ROW_HASH_MISMATCH (not MISSING_IN_*), Tier 5
    must fetch that exact key's full row from both sides and name the
    mismatched column/values - this is the acceptance criterion "one
    changed cell in a large table is found down to exact PK/column/values"
    from the tiered-funnel spec."""
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")])
        if catalog == "cat_source"
        else _hash_df([(1, "aaa"), (2, "zzz")])
    )
    connector.get_row_detail_for_keys.return_value = [
        {
            "key": {"id": 2},
            "mismatched_columns": ["name"],
            "source_values": {"name": "old-value"},
            "target_values": {"name": "new-value"},
            "source_row_hash": "bbb",
            "target_row_hash": "zzz",
        }
    ]
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]

    connector.get_row_detail_for_keys.assert_called_once()
    call_kwargs = connector.get_row_detail_for_keys.call_args.kwargs
    assert call_kwargs["key_column"] == "id"
    assert call_kwargs["key_values"] == ["2"]

    assert table.tier_reached == ValidationTier.COLUMN_DIFF
    assert len(table.data.sample_changed_detail) == 1
    detail = table.data.sample_changed_detail[0]
    assert detail.mismatch_column == "name"
    assert detail.source_value == "old-value"
    assert detail.target_value == "new-value"
    assert detail.primary_key == {"id": 2}
    assert detail.verified is True


def test_tier5_column_diff_runs_for_row_number_fallback_when_mismatched():
    """No primary key configured -> row-number fallback is used for Tier
    4, but a real mismatch must still let Tier 5 attempt best-effort
    column-level detail, clearly tagged verified=False (the user's
    explicit ask: don't leave Data Mismatches empty just because there's
    no real key - show it, but flag it as unverified)."""
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes_by_row_number.side_effect = (
        lambda catalog, schema, table, cols, bucket_predicate=None: (
            _hash_df([(1, "aaa"), (2, "bbb")], key="row_number")
            if catalog == "cat_source"
            else _hash_df([(1, "aaa"), (2, "zzz")], key="row_number")
        )
    )
    connector.get_row_detail_for_row_numbers.return_value = [
        {
            "key": {"row_number": 2},
            "mismatched_columns": ["name"],
            "source_values": {"name": "old-value"},
            "target_values": {"name": "new-value"},
            "source_row_hash": "bbb",
            "target_row_hash": "zzz",
        }
    ]
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())  # no primary_keys configured
    table = result.schemas[0].tables[0]

    connector.get_row_detail_for_row_numbers.assert_called_once()
    call_kwargs = connector.get_row_detail_for_row_numbers.call_args.kwargs
    assert call_kwargs["row_numbers"] == [2]
    assert call_kwargs["order_by_columns"] == ["id", "name"]
    assert call_kwargs["value_columns"] == ["id", "name"]

    assert table.tier_reached == ValidationTier.COLUMN_DIFF
    assert len(table.data.sample_changed_detail) == 1
    detail = table.data.sample_changed_detail[0]
    assert detail.mismatch_column == "name"
    assert detail.source_value == "old-value"
    assert detail.target_value == "new-value"
    assert detail.primary_key == {"row_number": 2}
    assert detail.verified is False
    assert "unverified" in table.data.note.lower()


def test_tier5_row_number_fallback_error_is_recorded_not_raised():
    """If the best-effort re-fetch itself fails, the run must not crash -
    same try/except contract as the real-key Tier 5 path."""
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes_by_row_number.side_effect = (
        lambda catalog, schema, table, cols, bucket_predicate=None: (
            _hash_df([(1, "aaa"), (2, "bbb")], key="row_number")
            if catalog == "cat_source"
            else _hash_df([(1, "aaa"), (2, "zzz")], key="row_number")
        )
    )
    connector.get_row_detail_for_row_numbers.side_effect = RuntimeError("boom")
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.data.sample_changed_detail == []
    assert table.data.error is not None
    assert "boom" in table.data.error


def test_row_hashes_key_missing_from_target():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa"), (2, "bbb")])
        if catalog == "cat_source"
        else _hash_df([(1, "aaa")])
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]

    assert table.data.row_hash_mismatch_count == 1
    mismatch = table.data.row_hash_mismatches[0]
    assert mismatch.primary_key == "2"
    assert mismatch.source_hash == "bbb"
    assert mismatch.target_hash == ""
    assert mismatch.status == "MISSING_IN_TARGET"
    assert table.data.status == ValidationStatus.FAIL


def test_row_hashes_key_missing_from_source():
    connector = _mismatched_fingerprint_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk, bucket_predicate=None: (
        _hash_df([(1, "aaa")])
        if catalog == "cat_source"
        else _hash_df([(1, "aaa"), (2, "bbb")])
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(primary_keys={"bronze.customers": ["id"]})
    )
    table = result.schemas[0].tables[0]

    assert table.data.row_hash_mismatch_count == 1
    mismatch = table.data.row_hash_mismatches[0]
    assert mismatch.primary_key == "2"
    assert mismatch.source_hash == ""
    assert mismatch.target_hash == "bbb"
    assert mismatch.status == "MISSING_IN_SOURCE"
    assert table.data.status == ValidationStatus.FAIL


def test_row_hashes_use_row_number_fallback_when_no_primary_key_configured():
    connector = _mismatched_fingerprint_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    # No real key configured -> get_row_hashes (real-key path) is never
    # called; get_row_hashes_by_row_number (fallback) is used instead.
    connector.get_row_hashes.assert_not_called()
    connector.get_row_hashes_by_row_number.assert_called()
    assert table.data.row_hash_mismatches == []
    assert table.data.row_hash_mismatch_count == 0
    # Matching row-number hashes on both sides is a real PASS signal.
    assert table.data.status == ValidationStatus.PASS

# ---------------------------------------------------------------------------
# "Compare all matching" discovery: compare_schemas / compare_tables
# intersection logic in isolation (schemas/tables present in both, only
# source, only target), plus the resulting visibility logging and
# guardrail scope count when no explicit schemas/tables restriction is
# given (i.e. schema_name/table left blank -> "compare everything common").
# ---------------------------------------------------------------------------
def test_compare_schemas_intersection_only_source_only_target():
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "gold", "only_source"] if catalog == "cat_source"
        else ["bronze", "gold", "only_target"]
    )
    validator = CatalogValidator(connector)

    common, missing, extra = validator.compare_schemas("cat_source", "cat_target")

    assert common == ["bronze", "gold"]
    assert missing == ["only_source"]
    assert extra == ["only_target"]


def test_compare_schemas_excludes_information_schema_from_all_three_buckets():
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "information_schema"] if catalog == "cat_source"
        else ["bronze"]
    )
    validator = CatalogValidator(connector)

    common, missing, extra = validator.compare_schemas("cat_source", "cat_target")

    assert common == ["bronze"]
    assert missing == []  # information_schema never counted as missing
    assert extra == []


def test_compare_tables_intersection_only_source_only_target():
    connector = _make_connector()
    connector.get_tables.side_effect = lambda catalog, schema: (
        ["customers", "orders", "only_source_table"] if catalog == "cat_source"
        else ["customers", "orders", "only_target_table"]
    )
    validator = CatalogValidator(connector)

    common, missing, extra = validator.compare_tables("cat_source", "cat_target", "bronze")

    assert common == ["customers", "orders"]
    assert missing == ["only_source_table"]
    assert extra == ["only_target_table"]


def test_no_schemas_restriction_logs_missing_and_extra_schemas(caplog):
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "only_source"] if catalog == "cat_source" else ["bronze", "only_target"]
    )
    validator = CatalogValidator(connector)

    with caplog.at_level("WARNING"):
        validator.compare_catalogs(_request())  # request.schemas is None -> compare all

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("only_source" in w for w in warnings)
    assert any("only_target" in w for w in warnings)


def test_no_tables_restriction_logs_missing_and_extra_tables(caplog):
    connector = _make_connector()
    connector.get_tables.side_effect = lambda catalog, schema: (
        ["customers", "only_source_table"] if catalog == "cat_source"
        else ["customers", "only_target_table"]
    )
    validator = CatalogValidator(connector)

    with caplog.at_level("WARNING"):
        validator.compare_catalogs(_request())  # request.tables is None -> compare all

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("only_source_table" in w for w in warnings)
    assert any("only_target_table" in w for w in warnings)


def test_explicit_schemas_restriction_suppresses_all_matching_visibility_log(caplog):
    """When request.schemas is explicitly set, we're not doing a "compare
    everything" run, so the missing/extra visibility warnings and the
    guardrail scope log should not fire (there's no ambiguous scope to
    warn about - the user named exactly what they want)."""
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "only_source"] if catalog == "cat_source" else ["bronze", "only_target"]
    )
    validator = CatalogValidator(connector)

    with caplog.at_level("WARNING"):
        validator.compare_catalogs(_request(schemas=["bronze"]))

    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert not any("only_source" in w or "only_target" in w for w in warnings)


def test_catalog_wide_run_logs_guardrail_scope_count(caplog):
    connector = _make_connector()
    connector.get_schemas.return_value = ["bronze", "gold"]
    connector.get_tables.return_value = ["customers"]
    validator = CatalogValidator(connector)

    with caplog.at_level("INFO"):
        validator.compare_catalogs(_request())  # no schemas/tables restriction

    info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any("2 schema(s), 2 table(s) total" in m for m in info_messages)


# ---------------------------------------------------------------------------
# enabled_validations: config.validations must actually gate what runs and
# what counts toward overall status, not just be echoed cosmetically.
# ---------------------------------------------------------------------------
def test_row_only_selection_skips_column_reporting_but_tier1_stats_still_run():
    """With enabled_validations={ROW}, COLUMN's *reporting* (columns_status,
    column_order_status, data_types_status, nullable_status) must stay
    SKIPPED/cosmetic-empty - but get_column_statistics itself now legitimately
    runs as part of Tier 1 (the statistical tier that lets ROW's fail-fast
    funnel stop before an expensive row-hash diff), since Tier 1 is part of
    ROW's pipeline, not COLUMN's. This is a deliberate scope change from the
    pre-tiered pipeline, where get_column_statistics was purely a COLUMN-only
    SQL call.

    table.columns must still be populated with Tier 1's null/distinct/
    min-max findings even though COLUMN is disabled - the Suggestions
    sheet explains a ROW-only FAIL by walking this list, and leaving it
    empty here previously produced an unhelpful "Unclassified" suggestion
    for a table that failed purely on a Tier 1 statistic."""
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(enabled_validations={ValidationType.ROW}))

    connector.get_column_statistics.assert_called()
    table = result.schemas[0].tables[0]
    assert table.columns_status == ValidationStatus.SKIPPED
    assert table.column_order_status == ValidationStatus.SKIPPED
    assert table.data_types_status == ValidationStatus.SKIPPED
    assert table.nullable_status == ValidationStatus.SKIPPED
    # null/distinct/min-max ARE populated - they're Tier 1's own findings,
    # gated by ROW (via the row_enabled early-return), not by COLUMN.
    assert table.null_counts_status != ValidationStatus.SKIPPED
    assert table.distinct_counts_status != ValidationStatus.SKIPPED
    assert [c.column for c in table.columns] == ["id", "name"]
    # Tier 0's per-column type/nullable detection stays COLUMN-gated -
    # only Tier 1's own fields (null/distinct/min-max) are populated here.
    assert table.columns[0].data_type_status is None
    assert table.columns[0].nullable_status is None
    assert table.columns[0].null_count_status is not None


def test_row_only_selection_still_runs_row_count_and_row_hash():
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(enabled_validations={ValidationType.ROW}))

    connector.get_row_count.assert_called()
    table = result.schemas[0].tables[0]
    assert table.row_count_status == ValidationStatus.PASS
    assert table.data is not None
    assert table.data.status != ValidationStatus.SKIPPED


def test_column_only_selection_skips_row_count_and_row_hash():
    """With enabled_validations={COLUMN}, row-level checks (row count,
    row-hash comparison) must not run - verified by asserting
    get_row_count/get_row_hashes_by_row_number are never called."""
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(enabled_validations={ValidationType.COLUMN}))

    connector.get_row_count.assert_not_called()
    connector.get_row_hashes_by_row_number.assert_not_called()
    table = result.schemas[0].tables[0]
    assert table.row_count_status == ValidationStatus.SKIPPED
    assert table.data.status == ValidationStatus.SKIPPED
    assert table.columns_status != ValidationStatus.SKIPPED


def test_column_type_mismatch_does_not_fail_table_when_column_deselected():
    """A real column-level discrepancy (data type mismatch) must not
    affect overall table status when COLUMN wasn't selected to run."""
    connector = _make_connector()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df([("id", "int", False)])
        if catalog == "cat_source"
        else _schema_df([("id", "bigint", False)])  # type mismatch
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request(enabled_validations={ValidationType.ROW}))
    table = result.schemas[0].tables[0]

    assert table.data_types_status == ValidationStatus.SKIPPED
    assert table.status != ValidationStatus.FAIL


def test_missing_schema_does_not_fail_overall_status_when_schema_deselected():
    connector = _make_connector()
    connector.get_schemas.side_effect = lambda catalog: (
        ["bronze", "gold"] if catalog == "cat_source" else ["bronze"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(enabled_validations={ValidationType.ROW})
    )

    assert "gold" in result.missing_schemas  # still reported for programmatic access
    assert result.status != ValidationStatus.FAIL


def test_missing_table_does_not_fail_schema_status_when_schema_deselected():
    connector = _make_connector()
    connector.get_tables.side_effect = lambda catalog, schema: (
        ["customers", "orders"] if catalog == "cat_source" else ["customers"]
    )
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(enabled_validations={ValidationType.ROW})
    )
    schema_result = result.schemas[0]

    assert schema_result.missing_tables == ["orders"]  # still reported
    assert schema_result.status != ValidationStatus.FAIL


def test_default_enabled_validations_matches_prior_full_pipeline_behavior():
    """No enabled_validations passed (or all four) -> identical behavior
    to before this feature existed, i.e. everything runs and counts."""
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())  # enabled_validations defaults to all four
    table = result.schemas[0].tables[0]

    connector.get_column_statistics.assert_called()
    connector.get_row_count.assert_called()
    assert table.columns_status != ValidationStatus.SKIPPED
    assert table.row_count_status != ValidationStatus.SKIPPED


def test_excel_report_omits_column_sections_when_column_not_selected():
    """The Excel report must not include a Column Validation sheet, and
    Table Validation must not include column-level columns, when
    "column" wasn't in enabled_validations."""
    from table_validator.reports.excel_report import (
        TABLE_HEADERS,
        _build_table_rows,
        _filter_table_columns,
    )

    connector = _make_connector()
    validator = CatalogValidator(connector)
    result = validator.compare_catalogs(_request(enabled_validations={ValidationType.ROW}))

    rows = _build_table_rows(result)
    headers, filtered_rows, _status_cols = _filter_table_columns(
        TABLE_HEADERS, rows, {"row"}
    )

    assert "Data Types" not in headers
    assert "Null Counts" not in headers
    assert "Schema Match" not in headers
    assert "Row Count (Src)" in headers
    assert "Data Match" in headers
    assert len(filtered_rows[0]) == len(headers)
