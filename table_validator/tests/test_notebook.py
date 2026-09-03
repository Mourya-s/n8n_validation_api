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
# Dotted-string parsing - "catalog.schema.table" (single-table) or
# "catalog.schema" (schema-wide sweep) are both valid; anything else
# (wrong part count, any blank part) is rejected.
# ---------------------------------------------------------------------------
def test_malformed_source_raises_value_error():
    with pytest.raises(ValueError, match="source must be"):
        validate_tables("just_one", "cat.sch.tbl", spark=MagicMock())


def test_malformed_target_raises_value_error():
    with pytest.raises(ValueError, match="target must be"):
        validate_tables("cat.sch.tbl", "way.too.many.parts", spark=MagicMock())


def test_blank_part_raises_value_error():
    with pytest.raises(ValueError, match="source must be"):
        validate_tables("cat..tbl", "cat.sch.tbl", spark=MagicMock())


def test_two_part_source_and_target_is_valid_sweep_syntax():
    """A 2-part 'catalog.schema' string on BOTH sides is now valid syntax
    (schema-wide sweep) - must not raise at the parsing stage."""
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        validate_tables("cat1.sch1", "cat1.sch1", spark=MagicMock())  # must not raise


def test_table_named_on_only_one_side_raises_value_error():
    with pytest.raises(ValueError, match="only one side"):
        validate_tables("cat1.sch1.t1", "cat1.sch1", spark=MagicMock())


def test_table_named_on_only_target_side_raises_value_error():
    with pytest.raises(ValueError, match="only one side"):
        validate_tables("cat1.sch1", "cat1.sch1.t1", spark=MagicMock())


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


# ---------------------------------------------------------------------------
# Schema-wide sweep mode: "catalog.schema" (no table) on BOTH sides.
# ---------------------------------------------------------------------------
def test_sweep_mode_leaves_tables_unset_for_auto_discovery():
    """tables=None is what triggers CatalogValidator's own auto-discovery
    of every identically-named table common to both schemas - a sweep
    call must not pass any table restriction unless table_map says so."""
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.bronze", "cat1.silver", spark=MagicMock())

    request = captured["request"]
    assert request.schemas == ["bronze"]
    assert request.tables is None
    assert request.table_map == {}
    assert request.schema_map == {"bronze": "silver"}


def test_sweep_mode_table_map_passes_through():
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
            "cat1.bronze", "cat1.silver",
            table_map={"cust": "customers", "ord": "orders"},
            spark=MagicMock(),
        )

    request = captured["request"]
    assert request.tables is None
    assert request.table_map == {"cust": "customers", "ord": "orders"}


def test_sweep_mode_schema_map_empty_when_schema_names_identical():
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.sch1", "cat1.sch1", spark=MagicMock())

    request = captured["request"]
    assert request.schema_map == {}
    assert request.tables is None


def test_sweep_mode_primary_key_raises_value_error():
    with pytest.raises(ValueError, match="only meaningful for a single-table"):
        validate_tables(
            "cat1.bronze", "cat1.silver",
            primary_key=["id"], spark=MagicMock(),
        )


def test_sweep_mode_primary_keys_dict_stays_empty():
    """Even without primary_key raising (it does), confirm the request's
    own primary_keys dict is never populated in sweep mode - defensive
    regression guard independent of the ValueError test above."""
    captured = {}

    def fake_compare_catalogs(self, request):
        captured["request"] = request
        return _fake_response()

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             fake_compare_catalogs,
         ):
        validate_tables("cat1.bronze", "cat1.silver", spark=MagicMock())

    assert captured["request"].primary_keys == {}


def test_single_table_mode_table_map_kwarg_ignored_in_favor_of_derived_map():
    """table_map is documented as sweep-mode-only; single-table mode
    already derives its own table_map from the two table names and must
    not be affected by an explicit table_map kwarg passed alongside it."""
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
            table_map={"unrelated": "mapping"},
            spark=MagicMock(),
        )

    request = captured["request"]
    assert request.tables == ["t1"]
    assert request.table_map == {"t1": "t2"}


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


def test_ignore_datatype_columns_passes_through():
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
            ignore_datatype_columns=["salary", "age"],
            spark=MagicMock(),
        )

    assert captured["request"].ignore_datatype_columns == ["salary", "age"]


def test_ignore_datatype_columns_defaults_to_empty_list():
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

    assert captured["request"].ignore_datatype_columns == []


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
# Plain-text rendering (str(result) / print(result) - the summary)
# ---------------------------------------------------------------------------
def test_str_is_plain_text_with_no_box_drawing_characters():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    text = str(result)
    box_drawing_chars = "│─┌┐└┘┡┩┏┓┗┛━┃"
    assert not any(ch in text for ch in box_drawing_chars)
    assert "Overall status: PASS" in text
    assert "1 total" in text
    assert "1 passed" in text


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

    text = str(result)
    assert "Error: boom" in text
    assert "Overall status: ERROR" in text


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

    assert "Per-table results:" not in str(result)


# ---------------------------------------------------------------------------
# ValidationResult sheet accessors (.table_validation, .column_validation,
# .data_mismatches, .row_hash_mismatches, .mismatch_categories, .suggestions)
# ---------------------------------------------------------------------------
def test_response_attribute_exposes_raw_catalog_validation_response():
    fake_response = _fake_response()
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=fake_response,
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    assert result.response is fake_response


def test_table_validation_returns_one_row_per_table():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    sheet = result.table_validation
    assert len(sheet) == 1
    assert sheet.headers[0] == "Source Schema"
    text = str(sheet)
    box_drawing_chars = "│─┌┐└┘┡┩┏┓┗┛━┃"
    assert not any(ch in text for ch in box_drawing_chars)
    assert "PASS" in text


def test_table_validation_to_dataframe_returns_real_dataframe():
    import pandas as pd

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    df = result.table_validation.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == result.table_validation.headers


def test_empty_sheet_renders_as_no_rows_placeholder():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    # This fixture's table has no data/mismatch detail at all.
    assert str(result.data_mismatches) == "(no rows)"
    assert len(result.data_mismatches) == 0


def test_all_sheet_properties_are_accessible_and_return_result_table():
    from table_validator.notebook import ResultTable

    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_response(),
         ):
        result = validate_tables("cat1.sch1.t1", "cat1.sch1.t2", spark=MagicMock())

    for attr in (
        "table_validation", "column_validation", "data_mismatches",
        "row_hash_mismatches", "mismatch_categories", "suggestions",
    ):
        sheet = getattr(result, attr)
        assert isinstance(sheet, ResultTable)
        # Must not raise regardless of whether the sheet has rows.
        str(sheet)


# ---------------------------------------------------------------------------
# ResultTable layout: narrow sheets render as one aligned table; sheets
# whose rendered row would be too wide switch to a vertical, one-field-
# per-line block per record instead (never truncating/hiding columns).
# ---------------------------------------------------------------------------
def test_narrow_sheet_renders_as_single_row_table():
    from table_validator.notebook import ResultTable

    headers = ["Source Schema", "Source Table", "Overall Status"]
    rows = [["bronze", "customers", "PASS"], ["bronze", "orders", "FAIL"]]
    text = str(ResultTable(headers, rows))

    # One line per record (plus the header line) - never split across
    # multiple lines for a narrow sheet like this.
    assert text.count("\n") == 2
    assert "bronze" in text and "customers" in text and "PASS" in text
    assert "--- Row" not in text


def test_wide_sheet_renders_as_vertical_blocks():
    from table_validator.notebook import ResultTable

    # Mirrors table_validation's real shape: many columns, some holding
    # long values (an ISO timestamp) - together this exceeds the width
    # threshold even though most individual values are short.
    headers = [f"Column {i}" for i in range(24)]
    row = [f"value_{i}" for i in range(24)]
    row[-2] = "2026-09-03T09:02:26.853264+00:00"
    text = str(ResultTable(headers, [row]))

    assert "--- Row 1 of 1 ---" in text
    for header in headers:
        assert f"{header}" in text
    assert "value_0" in text
    assert "2026-09-03T09:02:26.853264+00:00" in text


def test_vertical_blocks_separated_and_numbered_for_multiple_rows():
    from table_validator.notebook import ResultTable

    headers = [f"Column {i}" for i in range(24)]
    rows = [[f"value_{i}_{r}" for i in range(24)] for r in range(2)]
    text = str(ResultTable(headers, rows))

    assert "--- Row 1 of 2 ---" in text
    assert "--- Row 2 of 2 ---" in text
    assert "value_0_0" in text
    assert "value_0_1" in text


def test_table_validation_sheet_uses_vertical_layout_for_a_real_row():
    """Regression test for the real user-reported readability problem:
    table_validation's 24 columns, once a Validation Timestamp is
    present, must render as vertical blocks - not one unreadable,
    sideways-wrapping line."""
    from table_validator.reports.excel_report import TABLE_HEADERS
    from table_validator.notebook import ResultTable

    row = [
        "for_schema_validation", "file_example_xlsx_100_1",
        "for_schema_validation", "file_example_xlsx_100_1",
        "FAIL", "FAIL", "SKIPPED", None, None, None,
        "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED", "SKIPPED",
        0, "0%", 0, "0%", "SCHEMA_BLOCKED", "",
        "2026-09-03T09:02:26.853264+00:00", 8.77,
    ]
    text = str(ResultTable(TABLE_HEADERS, [row]))

    assert "--- Row 1 of 1 ---" in text
    assert "Source Schema" in text
    assert "Overall Status          : FAIL" in text


# ---------------------------------------------------------------------------
# End-to-end: a schema-wide sweep matching multiple tables must render
# correctly through ValidationResult's summary and every sheet property -
# these already loop every schema/table, so no notebook.py rendering
# changes were needed for sweep support, only request-building; this test
# is the regression guard proving that's actually true.
# ---------------------------------------------------------------------------
def _fake_sweep_response():
    table_a = TableValidationResult(
        schema_name="silver", table="customers", status=ValidationStatus.PASS,
        exists_in_source=True, exists_in_target=True,
    )
    table_b = TableValidationResult(
        schema_name="silver", table="orders", status=ValidationStatus.FAIL,
        exists_in_source=True, exists_in_target=True,
    )
    schema = SchemaValidationResult(
        schema_name="silver", status=ValidationStatus.FAIL, tables=[table_a, table_b],
    )
    return CatalogValidationResponse(
        source_catalog="cat1", target_catalog="cat1", status=ValidationStatus.FAIL,
        schemas=[schema],
    )


def test_sweep_result_summary_lists_every_matched_table():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_sweep_response(),
         ):
        result = validate_tables("cat1.bronze", "cat1.silver", spark=MagicMock())

    text = str(result)
    assert "Per-table results:" in text
    assert "silver.customers: PASS" in text
    assert "silver.orders: FAIL" in text
    assert "2 total" in text
    assert "1 passed" in text
    assert "1 failed" in text


def test_sweep_result_table_validation_sheet_has_one_row_per_matched_table():
    with patch("table_validator.notebook.SparkConnector", return_value=MagicMock()), \
         patch(
             "table_validator.notebook.CatalogValidator.compare_catalogs",
             return_value=_fake_sweep_response(),
         ):
        result = validate_tables("cat1.bronze", "cat1.silver", spark=MagicMock())

    sheet = result.table_validation
    assert len(sheet) == 2
    df = sheet.to_dataframe()
    assert set(df["Source Table"]) == {"customers", "orders"}
