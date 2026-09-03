"""
Tests for validators/row_validator.py's AzureSqlValidator schema/table
matching: identical-name matching (existing behavior) and schema_map/
table_map explicit-pair matching (the fix for a real user-reported bug
where an Azure SQL 'dbo' schema and a differently-named Databricks target
schema never matched, leaving every sheet empty).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from table_validator.models import AzureSqlValidationRequest, ValidationStatus
from table_validator.validators.row_validator import AzureSqlValidator


def _make_validator(sql_schemas, sql_tables_by_schema, databricks_schemas, databricks_tables_by_schema):
    azure_sql = MagicMock()
    azure_sql.get_schemas.return_value = sql_schemas
    azure_sql.get_tables.side_effect = lambda schema: sql_tables_by_schema.get(schema, [])

    databricks = MagicMock()
    databricks.get_schemas.return_value = databricks_schemas
    databricks.get_tables.side_effect = (
        lambda catalog, schema: databricks_tables_by_schema.get(schema, [])
    )

    return AzureSqlValidator(azure_sql, databricks)


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


def _make_full_validator(**overrides) -> AzureSqlValidator:
    """A validator whose connectors are mocked enough to run a full
    _validate_table pipeline end-to-end (not just the pure matching
    helpers), for regression tests on validate()'s scoping logic."""
    azure_sql = MagicMock()
    azure_sql.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "varchar", True)]
    )
    azure_sql.get_row_count.return_value = 10
    azure_sql.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 10},
        "name": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
    }
    azure_sql.is_min_max_eligible.side_effect = lambda dt: dt.lower() == "int"
    azure_sql.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    databricks = MagicMock()
    databricks.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "string", True)]
    )
    databricks.get_row_count.return_value = 10
    databricks.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 10},
        "name": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
    }
    databricks.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    for key, value in overrides.items():
        target, _, attr = key.partition("__")
        setattr(azure_sql if target == "azure_sql" else databricks, attr, value)

    return AzureSqlValidator(azure_sql, databricks)


# ---------------------------------------------------------------------------
# validate(): explicit schemas/tables scope must exclude unrelated missing/
# extra findings elsewhere in the database - regression test for the same
# bug already fixed in catalog_validator.py's compare_catalogs/_validate_schema.
# ---------------------------------------------------------------------------
def test_explicit_tables_scope_excludes_unrelated_missing_and_extra_tables():
    validator = _make_full_validator()
    validator.azure_sql.get_schemas.return_value = ["dbo"]
    validator.azure_sql.get_tables.return_value = ["customers", "orders"]
    validator.databricks.get_schemas.return_value = ["dbo"]
    validator.databricks.get_tables.return_value = ["customers", "employees"]

    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        schemas=["dbo"],
        tables=["customers"],
    )

    result = validator.validate(request)
    schema_result = result.schemas[0]

    assert schema_result.missing_tables == []
    assert schema_result.extra_tables == []
    assert [t.table for t in schema_result.tables] == ["customers"]
    assert result.status != ValidationStatus.FAIL or schema_result.tables[0].status == ValidationStatus.FAIL


def test_explicit_schemas_scope_excludes_unrelated_missing_and_extra_schemas():
    validator = _make_full_validator()
    validator.azure_sql.get_schemas.return_value = ["dbo", "sales"]
    validator.azure_sql.get_tables.return_value = ["customers"]
    validator.databricks.get_schemas.return_value = ["dbo", "marketing"]
    validator.databricks.get_tables.return_value = ["customers"]

    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        schemas=["dbo"],
    )

    result = validator.validate(request)

    assert result.missing_schemas == []
    assert result.extra_schemas == []
    assert [s.schema_name for s in result.schemas] == ["dbo"]


# ---------------------------------------------------------------------------
# _compare_schemas(): identical-name matching (existing behavior, unchanged)
# ---------------------------------------------------------------------------
def test_compare_schemas_matches_identical_names_with_no_map():
    validator = _make_validator(
        sql_schemas=["bronze"], sql_tables_by_schema={},
        databricks_schemas=["bronze"], databricks_tables_by_schema={},
    )
    request = AzureSqlValidationRequest(target_catalog="tgt_cat")

    common_pairs, missing, extra = validator._compare_schemas(request)

    assert common_pairs == [("bronze", "bronze")]
    assert missing == []
    assert extra == []


def test_compare_schemas_no_match_without_map_when_names_differ():
    """Reproduces the real bug: 'dbo' (Azure SQL) vs 'for_schema_validation'
    (Databricks) never match without an explicit schema_map."""
    validator = _make_validator(
        sql_schemas=["dbo"], sql_tables_by_schema={},
        databricks_schemas=["for_schema_validation"], databricks_tables_by_schema={},
    )
    request = AzureSqlValidationRequest(target_catalog="tgt_cat")

    common_pairs, missing, extra = validator._compare_schemas(request)

    assert common_pairs == []
    assert missing == ["dbo"]
    assert extra == ["for_schema_validation"]


# ---------------------------------------------------------------------------
# _compare_schemas(): schema_map explicit pair bypasses name matching
# ---------------------------------------------------------------------------
def test_compare_schemas_schema_map_matches_different_names():
    validator = _make_validator(
        sql_schemas=["dbo"], sql_tables_by_schema={},
        databricks_schemas=["for_schema_validation"], databricks_tables_by_schema={},
    )
    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        schema_map={"dbo": "for_schema_validation"},
    )

    common_pairs, missing, extra = validator._compare_schemas(request)

    assert common_pairs == [("dbo", "for_schema_validation")]
    assert missing == []
    assert extra == []


def test_compare_schemas_restriction_filters_by_source_name():
    """request.schemas restricts by the SOURCE (Azure SQL) schema name -
    this is what cli/main.py relies on when building schemas=[sql_source.schema]."""
    validator = _make_validator(
        sql_schemas=["dbo", "sales"], sql_tables_by_schema={},
        databricks_schemas=["for_schema_validation", "sales"], databricks_tables_by_schema={},
    )
    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        schemas=["dbo"],
        schema_map={"dbo": "for_schema_validation"},
    )

    with_restriction = [
        (src, tgt) for src, tgt in
        validator._compare_schemas(request)[0]
        if src.lower() in {s.lower() for s in request.schemas}
    ]
    assert with_restriction == [("dbo", "for_schema_validation")]


# ---------------------------------------------------------------------------
# _compare_tables(): identical-name matching (existing behavior, unchanged)
# ---------------------------------------------------------------------------
def test_compare_tables_matches_identical_names_with_no_map():
    validator = _make_validator(
        sql_schemas=[], sql_tables_by_schema={"dbo": ["customers"]},
        databricks_schemas=[], databricks_tables_by_schema={"bronze": ["customers"]},
    )
    request = AzureSqlValidationRequest(target_catalog="tgt_cat")

    common_pairs, missing, extra = validator._compare_tables(request, "dbo", "bronze")

    assert common_pairs == [("customers", "customers")]
    assert missing == []
    assert extra == []


def test_compare_tables_no_match_without_map_when_names_differ():
    validator = _make_validator(
        sql_schemas=[], sql_tables_by_schema={"dbo": ["Employees"]},
        databricks_schemas=[], databricks_tables_by_schema={"bronze": ["employees_sample"]},
    )
    request = AzureSqlValidationRequest(target_catalog="tgt_cat")

    common_pairs, missing, extra = validator._compare_tables(request, "dbo", "bronze")

    assert common_pairs == []
    assert missing == ["Employees"]
    assert extra == ["employees_sample"]


# ---------------------------------------------------------------------------
# _compare_tables(): table_map explicit pair bypasses name matching
# ---------------------------------------------------------------------------
def test_compare_tables_table_map_matches_different_names():
    validator = _make_validator(
        sql_schemas=[], sql_tables_by_schema={"dbo": ["Employees"]},
        databricks_schemas=[], databricks_tables_by_schema={"bronze": ["employees_sample"]},
    )
    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        table_map={"Employees": "employees_sample"},
    )

    common_pairs, missing, extra = validator._compare_tables(request, "dbo", "bronze")

    assert common_pairs == [("Employees", "employees_sample")]
    assert missing == []
    assert extra == []


def test_compare_tables_table_map_preserves_unmapped_identical_matches():
    """table_map only affects the mapped table - other same-named tables
    in the same schema still match normally."""
    validator = _make_validator(
        sql_schemas=[], sql_tables_by_schema={"dbo": ["Employees", "orders"]},
        databricks_schemas=[],
        databricks_tables_by_schema={"bronze": ["employees_sample", "orders"]},
    )
    request = AzureSqlValidationRequest(
        target_catalog="tgt_cat",
        table_map={"Employees": "employees_sample"},
    )

    common_pairs, missing, extra = validator._compare_tables(request, "dbo", "bronze")

    assert sorted(common_pairs) == [("Employees", "employees_sample"), ("orders", "orders")]
    assert missing == []
    assert extra == []


# ---------------------------------------------------------------------------
# AzureSqlConnector auth modes: SQL login vs Microsoft Entra ID service
# principal. Entra is required by Synapse workspaces that have SQL
# authentication disabled, where any UID/PWD is rejected outright.
# ---------------------------------------------------------------------------
def test_connector_defaults_to_sql_auth_when_username_password_given():
    from table_validator.connectors.azure_connector import AzureSqlConnector

    connector = AzureSqlConnector(
        server="s.database.windows.net", database="db",
        username="u", password="p",
    )

    assert connector._use_entra is False


def test_connector_uses_entra_when_full_service_principal_given():
    from table_validator.connectors.azure_connector import AzureSqlConnector

    connector = AzureSqlConnector(
        server="ws.sql.azuresynapse.net", database="pool",
        tenant_id="t", client_id="c", client_secret="s",
    )

    assert connector._use_entra is True


def test_connector_entra_takes_precedence_over_a_leftover_sql_login():
    """A config carrying both must resolve deterministically to Entra,
    not depend on argument order or silently prefer the stale login."""
    from table_validator.connectors.azure_connector import AzureSqlConnector

    connector = AzureSqlConnector(
        server="ws.sql.azuresynapse.net", database="pool",
        username="u", password="p",
        tenant_id="t", client_id="c", client_secret="s",
    )

    assert connector._use_entra is True


def test_connector_partial_service_principal_falls_back_to_sql_requirement():
    """Two of the three Entra fields is not a usable service principal -
    without a SQL login too, that must be a clear ValueError rather than
    an attempted half-configured Entra connection."""
    import pytest

    from table_validator.connectors.azure_connector import AzureSqlConnector

    with pytest.raises(ValueError, match="Entra service principal"):
        AzureSqlConnector(
            server="ws.sql.azuresynapse.net", database="pool",
            tenant_id="t", client_id="c",  # no client_secret
        )


def test_entra_access_token_struct_uses_length_prefixed_utf16le(monkeypatch):
    """The ODBC access-token attribute expects a 4-byte little-endian
    length prefix followed by the UTF-16-LE token bytes - getting this
    layout wrong is silently rejected by the driver, so pin it."""
    import struct
    import sys
    import types

    from table_validator.connectors.azure_connector import AzureSqlConnector

    fake_credential = MagicMock()
    fake_credential.get_token.return_value = MagicMock(token="abc")

    fake_identity = types.ModuleType("azure.identity")
    fake_identity.ClientSecretCredential = MagicMock(return_value=fake_credential)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_identity)

    connector = AzureSqlConnector(
        server="ws.sql.azuresynapse.net", database="pool",
        tenant_id="t", client_id="c", client_secret="s",
    )
    packed = connector._entra_access_token_struct()

    expected_bytes = "abc".encode("utf-16-le")
    assert packed == struct.pack("<i", len(expected_bytes)) + expected_bytes

    # Azure SQL's resource scope - the correct audience for Synapse SQL too.
    fake_credential.get_token.assert_called_once_with(
        "https://database.windows.net/.default"
    )
