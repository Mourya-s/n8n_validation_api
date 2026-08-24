"""
Tests for CatalogValidator (comparison_engine.py).

Kept as a separate file from the existing test_api.py since that file's
conventions weren't available - all Databricks calls are mocked here so
none of these tests require a live Databricks environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from comparison_engine import CatalogValidator
from models import (
    CatalogValidationRequest,
    DataCompareMode,
    ValidationStatus,
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
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


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


# ---------------------------------------------------------------------------
# Stage 7: data type mismatch
# ---------------------------------------------------------------------------
def test_datatype_mismatch_detected():
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
    col = result.schemas[0].tables[0].columns[0]

    assert col.data_type_status == ValidationStatus.FAIL
    assert col.source_data_type == "int"
    assert col.target_data_type == "bigint"


# ---------------------------------------------------------------------------
# Stage 8: nullable mismatch
# ---------------------------------------------------------------------------
def test_nullable_mismatch_detected():
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
    col = result.schemas[0].tables[0].columns[0]

    assert col.nullable_status == ValidationStatus.FAIL


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
    connector = _make_connector()
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
# Data compare mode: default STATISTICS mode skips row-level compare
# ---------------------------------------------------------------------------
def test_default_mode_skips_row_level_data_comparison():
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.data.mode == DataCompareMode.STATISTICS
    assert table.data.status == ValidationStatus.SKIPPED
    connector.key_based_row_diff.assert_not_called()


def test_full_mode_without_key_is_skipped_safely():
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(
        _request(data_compare_mode=DataCompareMode.FULL)
    )
    table = result.schemas[0].tables[0]

    assert table.data.status == ValidationStatus.SKIPPED
    assert "key" in table.data.note.lower()


# ---------------------------------------------------------------------------
# Row-hash comparison stage (CatalogValidator.compare_row_hashes /
# DatabricksConnector.get_row_hashes) - separate mechanism from
# key_based_row_diff/_changed_row_detail (FULL mode), primary way to detect
# row-level mismatches whenever a primary key is configured, independent
# of data_compare_mode.
# ---------------------------------------------------------------------------
def _hash_df(rows):
    """rows: list of (id, row_hash)"""
    return pd.DataFrame([{"id": r[0], "row_hash": r[1]} for r in rows])


def test_row_hashes_matching_produces_no_mismatch():
    connector = _make_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk: (
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
    connector = _make_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk: (
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


def test_row_hashes_key_missing_from_target():
    connector = _make_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk: (
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
    connector = _make_connector()
    connector.get_row_hashes.side_effect = lambda catalog, schema, table, cols, pk: (
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


def test_row_hashes_skipped_when_no_primary_key_configured():
    connector = _make_connector()
    validator = CatalogValidator(connector)

    result = validator.compare_catalogs(_request())
    table = result.schemas[0].tables[0]

    assert table.data.row_hash_mismatches == []
    assert table.data.row_hash_mismatch_count == 0
    connector.get_row_hashes.assert_not_called()
    # No key configured -> compare_data's own STATISTICS-mode skip is
    # untouched by the row-hash stage.
    assert table.data.status == ValidationStatus.SKIPPED