"""
Tests for notebook.py's validate_tables() - the notebook-native entry
point. SparkConnector construction and CatalogValidator.compare_catalogs
are both mocked, so these tests verify only validate_tables()'s own
logic: dotted-string parsing, CatalogValidationRequest construction, and
plain-text rendering - not the underlying comparison engine itself
(already covered by test_catalog_validator.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from table_validator.models import (
    CatalogValidationResponse,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
    ValidationTier,
)
from table_validator.models import DataCompareMode
from table_validator.notebook import validate_tables


def _fake_response(status=ValidationStatus.PASS, table_status=ValidationStatus.PASS):
    table = TableValidationResult(
        schema_name="sch1", table="t2", status=table_status,
        exists_in_source=True, exists_in_target=True,
    )
    schema = SchemaValidationResult(schema_name="sch1", status=status, tables=[table])
    return CatalogValidationResponse(
        source_catalog="cat1", target_catalog="cat1", status=status, schemas=[schema],
    )


# ---------------------------------------------------------------------------
# Dotted-string parsing
# ---------------------------------------------------------------------------
def test_malformed_source_raises_value_error():
    with pytest.raises(ValueError, match="source must be"):
        validate_tables("only.two", "cat.sch.tbl", spark=MagicMock())


def test_malformed_target_raises_value_error():
    with pytest.raises(ValueError, match="target must be"):
        validate_tables("cat.sch.tbl", "way.too.many.parts", spark=MagicMock())


def test_blank_part_raises_value_error():
    with pytest.raises(ValueError, match="source must be"):
        validate_tables("cat..tbl", "cat.sch.tbl", spark=MagicMock())


# ---------------------------------------------------------------------------
# CatalogValidationRequest construction
# ---------------------------------------------------------------------------
def test_request_scoped_to_exactly_one_schema_and_table():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    request = captured["request"]
    assert request.source_catalog == "cat1"
    assert request.target_catalog == "cat1"
    assert request.schemas == ["sch1"]
    assert request.tables == ["t1"]


def test_schema_map_and_table_map_populated_only_when_names_differ():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.bronze.old_name", "cat2.silver.new_name", spark=MagicMock())

    request = captured["request"]
    assert request.schema_map == {"bronze": "silver"}
    assert request.table_map == {"old_name": "new_name"}


def test_schema_map_and_table_map_empty_when_names_identical():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.sch1.t1", "cat1.sch1.t1", spark=MagicMock())

    request = captured["request"]
    assert request.schema_map == {}
    assert request.table_map == {}


def test_primary_key_populated_under_both_lookup_forms():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables(
            "cat1.sch1.t1", "cat1.sch1.t2",
            primary_key=["id"], spark=MagicMock(),
        )

    request = captured["request"]
    assert request.primary_keys["t2"] == ["id"]
    assert request.primary_keys["sch1.t2"] == ["id"]


def test_max_tier_and_enabled_validations_left_at_pydantic_defaults():
    """validate_tables must NOT override max_tier/enabled_validations -
    the bare model defaults already equal the CLI's own 'full' mode."""
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    request = captured["request"]
    assert request.max_tier == ValidationTier.COLUMN_DIFF
    assert request.data_compare_mode == DataCompareMode.default()


def test_ignore_only_columns_and_column_map_pass_through():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables(
            "cat1.sch1.t1", "cat1.sch1.t2",
            ignore_columns=["updated_at"],
            only_columns=["id", "name"],
            column_map={"cust_id": "customer_id"},
            spark=MagicMock(),
        )

    request = captured["request"]
    assert request.ignore_columns == ["updated_at"]
    assert request.only_columns == ["id", "name"]
    assert request.column_map == {"cust_id": "customer_id"}


# ---------------------------------------------------------------------------
# SparkConnector construction
# ---------------------------------------------------------------------------
def test_spark_kwarg_forwarded_to_spark_connector():
    fake_spark = MagicMock()
    spark_connector_calls = []

    def fake_spark_connector(spark=None):
        spark_connector_calls.append(spark)
        return MagicMock()

    with patch("table_validator.notebook.SparkConnector", side_effect=fake_spark_connector), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=fake_spark)

    assert spark_connector_calls == [fake_spark]


# ---------------------------------------------------------------------------
# Plain-text rendering
# ---------------------------------------------------------------------------
def test_returns_plain_string_with_no_box_drawing_characters():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    assert isinstance(result, str)
    box_drawing_chars = "│─┌┐└┘┡┩┏┓┗┛━┃"
    assert not any(ch in result for ch in box_drawing_chars)
    assert "Overall status: PASS" in result
    assert "1 total" in result
    assert "1 passed" in result


def test_error_text_included_when_result_has_error():
    error_response = CatalogValidationResponse(
        source_catalog="cat1", target_catalog="cat1",
        status=ValidationStatus.ERROR, schemas=[], error="boom",
    )
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=error_response,
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    assert "Error: boom" in result
    assert "Overall status: ERROR" in result


def test_per_table_lines_omitted_for_single_table():
    """Only one table pair is ever compared by validate_tables - the
    'Per-table results:' section (which only adds value when there's more
    than one row) must not appear."""
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    assert "Per-table results:" not in result
