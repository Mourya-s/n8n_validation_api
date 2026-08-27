"""
Tests for validators/blob_discovery.py: blob-filename-to-table-name
matching (discover_blob_table_matches) and BlobCatalogValidator's row
count / column comparison pipeline. All Azure/Databricks calls are
mocked here so none of these tests require live connections.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from table_validator.models import ValidationStatus
from table_validator.validators.blob_discovery import (
    BlobCatalogValidator,
    _infer_table_name,
    discover_blob_table_matches,
)


# ---------------------------------------------------------------------------
# _infer_table_name(): strip path + extension
# ---------------------------------------------------------------------------
def test_infer_table_name_strips_path_and_extension():
    assert _infer_table_name("validation/2024/Customers.csv") == "Customers"


def test_infer_table_name_no_path():
    assert _infer_table_name("orders.parquet") == "orders"


def test_infer_table_name_no_extension():
    assert _infer_table_name("no_extension") == "no_extension"


# ---------------------------------------------------------------------------
# discover_blob_table_matches(): matched / blob-only / table-only
# ---------------------------------------------------------------------------
def test_discover_matches_present_in_both():
    blobs = ["validation/customers.csv", "validation/orders.parquet"]
    tables = ["customers", "orders"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == [
        ("validation/customers.csv", "customers"),
        ("validation/orders.parquet", "orders"),
    ]
    assert blob_only == []
    assert table_only == []


def test_discover_matches_are_case_insensitive():
    blobs = ["Customers.csv"]
    tables = ["customers"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == [("Customers.csv", "customers")]
    assert blob_only == []
    assert table_only == []


def test_discover_reports_blob_only():
    blobs = ["customers.csv", "only_in_blob.csv"]
    tables = ["customers"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == [("customers.csv", "customers")]
    assert blob_only == ["only_in_blob.csv"]
    assert table_only == []


def test_discover_reports_table_only():
    blobs = ["customers.csv"]
    tables = ["customers", "only_in_catalog"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == [("customers.csv", "customers")]
    assert blob_only == []
    assert table_only == ["only_in_catalog"]


def test_discover_reports_both_blob_only_and_table_only_simultaneously():
    blobs = ["customers.csv", "extra_blob.parquet"]
    tables = ["customers", "extra_table"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == [("customers.csv", "customers")]
    assert blob_only == ["extra_blob.parquet"]
    assert table_only == ["extra_table"]


def test_discover_returns_empty_when_nothing_in_common():
    blobs = ["a.csv"]
    tables = ["b"]

    matched, blob_only, table_only = discover_blob_table_matches(blobs, tables)

    assert matched == []
    assert blob_only == ["a.csv"]
    assert table_only == ["b"]


# ---------------------------------------------------------------------------
# BlobCatalogValidator.validate(): end-to-end with mocked connectors
# ---------------------------------------------------------------------------
def _mock_azure_connector(blobs, dataframes):
    mock = MagicMock()
    mock.container_name = "n8ncontainer"
    mock.list_blobs.return_value = blobs
    mock.read_csv.side_effect = lambda path: dataframes[path]
    return mock


def _mock_databricks_connector(tables, row_counts, schema_df):
    mock = MagicMock()
    mock.get_tables.return_value = tables
    mock.get_row_count.side_effect = lambda catalog, schema, table: row_counts[table]
    mock.get_table_schema.return_value = schema_df
    return mock


def _schema_df(columns):
    return pd.DataFrame(
        [{"column_name": c[0], "data_type": c[1]} for c in columns]
    )


def test_validate_compares_only_matched_pairs_and_flags_row_count_mismatch():
    df_customers = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    azure = _mock_azure_connector(
        blobs=["validation/customers.csv", "validation/only_in_blob.csv"],
        dataframes={"validation/customers.csv": df_customers},
    )
    databricks = _mock_databricks_connector(
        tables=["customers", "only_in_catalog"],
        row_counts={"customers": 5},  # mismatch: blob has 3 rows, table has 5
        schema_df=_schema_df([("id", "bigint"), ("name", "string")]),
    )

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema="bronze")

    assert len(result.schemas) == 1
    schema_result = result.schemas[0]
    # Blob is the source, table is the target: a blob with no matching
    # table is "missing from target"; a table with no matching blob is
    # "extra in target" - same convention as CatalogValidator/AzureSqlValidator.
    assert schema_result.missing_tables == ["validation/only_in_blob.csv"]
    assert schema_result.extra_tables == ["only_in_catalog"]

    # Only the matched pair (customers) gets a TableValidationResult -
    # only_in_blob.csv and only_in_catalog are surfaced as missing/extra,
    # not silently ignored, but never produce a fabricated comparison result.
    assert [t.table for t in schema_result.tables] == ["customers"]

    table = schema_result.tables[0]
    assert table.row_count_source == 3
    assert table.row_count_target == 5
    assert table.row_count_status == ValidationStatus.FAIL
    assert result.status == ValidationStatus.FAIL


def test_validate_passes_when_row_counts_and_columns_match():
    df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    azure = _mock_azure_connector(
        blobs=["orders.csv"],
        dataframes={"orders.csv": df},
    )
    databricks = _mock_databricks_connector(
        tables=["orders"],
        row_counts={"orders": 2},
        schema_df=_schema_df([("id", "bigint"), ("name", "string")]),
    )

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema="bronze")

    table = result.schemas[0].tables[0]
    assert table.row_count_status == ValidationStatus.PASS
    assert table.data_types_status == ValidationStatus.PASS
    assert table.status == ValidationStatus.PASS
    assert result.status == ValidationStatus.PASS


def test_validate_flags_column_type_mismatch():
    df = pd.DataFrame({"id": [1, 2], "amount": [1.5, 2.5]})
    azure = _mock_azure_connector(
        blobs=["orders.csv"],
        dataframes={"orders.csv": df},
    )
    databricks = _mock_databricks_connector(
        tables=["orders"],
        row_counts={"orders": 2},
        # amount is 'double' in the blob (float) but 'string' in the target.
        schema_df=_schema_df([("id", "bigint"), ("amount", "string")]),
    )

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema="bronze")

    table = result.schemas[0].tables[0]
    amount_col = next(c for c in table.columns if c.column == "amount")
    assert amount_col.data_type_status == ValidationStatus.FAIL
    assert table.status == ValidationStatus.FAIL


def test_validate_passes_folder_prefix_and_file_pattern_through_to_list_blobs():
    azure = _mock_azure_connector(blobs=[], dataframes={})
    databricks = _mock_databricks_connector(tables=[], row_counts={}, schema_df=_schema_df([]))

    validator = BlobCatalogValidator(azure, databricks)
    validator.validate(
        target_catalog="tgt_cat",
        target_schema="bronze",
        folder_prefix="validation/2024/",
        file_pattern="*.csv",
    )

    azure.list_blobs.assert_called_once_with("validation/2024/", "*.csv")


def test_validate_no_matches_still_returns_a_response_not_a_crash():
    azure = _mock_azure_connector(blobs=["a.csv"], dataframes={"a.csv": pd.DataFrame()})
    databricks = _mock_databricks_connector(tables=["b"], row_counts={}, schema_df=_schema_df([]))

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema="bronze")

    assert result.schemas[0].tables == []
    # "a.csv" has no matching table -> missing from target.
    # "b" has no matching blob -> extra in target.
    assert result.schemas[0].missing_tables == ["a.csv"]
    assert result.schemas[0].extra_tables == ["b"]
    assert result.status == ValidationStatus.FAIL  # unmatched blob counts as a failure


# ---------------------------------------------------------------------------
# BlobCatalogValidator.validate(): target_schema=None -> compare every
# schema in the catalog. Reproduces the real bug where the CLI passed ""
# instead of None, causing Databricks to error with SCHEMA_NOT_FOUND on
# the empty-string schema name.
# ---------------------------------------------------------------------------
def test_validate_with_no_schema_lists_and_matches_every_schema():
    df = pd.DataFrame({"id": [1, 2]})
    azure = _mock_azure_connector(
        blobs=["customers.csv", "orders.csv"],
        dataframes={"customers.csv": df, "orders.csv": df},
    )
    databricks = MagicMock()
    databricks.get_schemas.return_value = ["bronze", "silver", "information_schema"]
    databricks.get_tables.side_effect = lambda catalog, schema: (
        ["customers"] if schema == "bronze" else ["orders"]
    )
    databricks.get_row_count.return_value = 2
    databricks.get_table_schema.return_value = _schema_df([("id", "bigint")])

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema=None)

    # information_schema excluded, same convention as CatalogValidator.
    databricks.get_schemas.assert_called_once_with("tgt_cat")
    schema_names = sorted(s.schema_name for s in result.schemas)
    assert schema_names == ["bronze", "silver"]

    bronze = next(s for s in result.schemas if s.schema_name == "bronze")
    silver = next(s for s in result.schemas if s.schema_name == "silver")
    assert [t.table for t in bronze.tables] == ["customers"]
    assert [t.table for t in silver.tables] == ["orders"]

    # Both blobs are checked against EVERY schema independently: bronze
    # only has a "customers" table, so orders.csv is reported missing
    # from bronze (and vice versa for silver) - this is correct given
    # "no schema configured" means "check every schema", not a bug.
    assert bronze.missing_tables == ["orders.csv"]
    assert silver.missing_tables == ["customers.csv"]
    assert result.status == ValidationStatus.FAIL


def test_validate_with_no_schema_never_passes_empty_string_to_get_tables():
    """The actual regression: target_schema=None must never degrade to
    "" being passed to DatabricksConnector.get_tables (which Databricks
    itself rejects with SCHEMA_NOT_FOUND)."""
    azure = _mock_azure_connector(blobs=[], dataframes={})
    databricks = MagicMock()
    databricks.get_schemas.return_value = ["bronze"]
    databricks.get_tables.return_value = []

    validator = BlobCatalogValidator(azure, databricks)
    validator.validate(target_catalog="tgt_cat", target_schema=None)

    for call in databricks.get_tables.call_args_list:
        args, kwargs = call
        schema_arg = args[1] if len(args) > 1 else kwargs.get("schema")
        assert schema_arg == "bronze"
        assert schema_arg != ""


def test_validate_with_no_schema_reports_schema_listing_failure_as_error():
    azure = _mock_azure_connector(blobs=["a.csv"], dataframes={"a.csv": pd.DataFrame()})
    databricks = MagicMock()
    databricks.get_schemas.side_effect = RuntimeError("boom")

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(target_catalog="tgt_cat", target_schema=None)

    assert result.status == ValidationStatus.ERROR
    assert "Unable to list target schemas" in result.error


def test_validate_with_explicit_schema_does_not_list_all_schemas():
    """When a schema IS given, get_schemas should never be called - only
    the named schema's tables are fetched."""
    azure = _mock_azure_connector(blobs=[], dataframes={})
    databricks = MagicMock()
    databricks.get_tables.return_value = []

    validator = BlobCatalogValidator(azure, databricks)
    validator.validate(target_catalog="tgt_cat", target_schema="bronze")

    databricks.get_schemas.assert_not_called()
    databricks.get_tables.assert_called_once_with("tgt_cat", "bronze")


# ---------------------------------------------------------------------------
# BlobCatalogValidator.validate(): explicit blob_path + target_table pair
# bypasses filename-to-table discovery entirely, even when names differ.
# ---------------------------------------------------------------------------
def test_validate_explicit_pair_compares_directly_even_with_different_names():
    df = pd.DataFrame({"id": [1, 2]})
    azure = _mock_azure_connector(
        blobs=["irrelevant_other_blob.csv"],  # would never match by filename
        dataframes={"n8ndirectory/weird_name.csv": df},
    )
    databricks = MagicMock()
    databricks.get_table_schema.return_value = _schema_df([("id", "bigint")])
    databricks.get_row_count.return_value = 2

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(
        target_catalog="tgt_cat",
        target_schema="bronze",
        blob_path="n8ndirectory/weird_name.csv",
        target_table="completely_different_table_name",
    )

    # list_blobs/get_tables (discovery) must never be called in this mode.
    azure.list_blobs.assert_not_called()
    databricks.get_tables.assert_not_called()

    assert len(result.schemas) == 1
    schema_result = result.schemas[0]
    assert schema_result.missing_tables == []
    assert schema_result.extra_tables == []
    assert [t.table for t in schema_result.tables] == ["completely_different_table_name"]

    table = schema_result.tables[0]
    assert table.row_count_source == 2
    assert table.row_count_target == 2
    assert table.status == ValidationStatus.PASS
    assert result.status == ValidationStatus.PASS


def test_validate_explicit_pair_requires_target_schema():
    azure = _mock_azure_connector(blobs=[], dataframes={})
    databricks = MagicMock()

    validator = BlobCatalogValidator(azure, databricks)
    result = validator.validate(
        target_catalog="tgt_cat",
        target_schema=None,
        blob_path="a.csv",
        target_table="some_table",
    )

    assert result.status == ValidationStatus.ERROR
    assert "schema" in result.error.lower()


def test_validate_without_both_blob_path_and_target_table_falls_back_to_discovery():
    """Only one of blob_path/target_table set -> normal discovery still
    runs (both must be present to trigger the explicit-pair bypass)."""
    azure = _mock_azure_connector(blobs=["customers.csv"], dataframes={"customers.csv": pd.DataFrame({"id": [1]})})
    databricks = MagicMock()
    databricks.get_tables.return_value = ["customers"]
    databricks.get_table_schema.return_value = _schema_df([("id", "bigint")])
    databricks.get_row_count.return_value = 1

    validator = BlobCatalogValidator(azure, databricks)
    validator.validate(
        target_catalog="tgt_cat",
        target_schema="bronze",
        blob_path="customers.csv",
        target_table=None,  # only blob_path set, not target_table
    )

    azure.list_blobs.assert_called_once()
    databricks.get_tables.assert_called_once_with("tgt_cat", "bronze")
