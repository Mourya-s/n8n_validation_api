"""
Tests for reports/excel_report.py's row-building logic. Focused on the
missing_tables/extra_tables labeling bug: a table present in target but
absent from source must be labeled EXTRA_IN_TARGET, not lumped in with
MISSING_FROM_TARGET (which means the opposite - present in source,
absent from target).
"""

from __future__ import annotations

from table_validator.models import (
    CatalogValidationResponse,
    DataValidationResult,
    RowHashMismatch,
    RowMismatchDetail,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
    ValidationTier,
)
from table_validator.reports.excel_report import (
    MISMATCH_HEADERS,
    ROW_HASH_HEADERS,
    TABLE_HEADERS,
    _build_mismatch_rows,
    _build_row_hash_rows,
    _build_suggestion_rows,
    _build_table_rows,
)

_SOURCE_TABLE_COL = 1
_STATUS_COL = 4


def _response(schema_result: SchemaValidationResult) -> CatalogValidationResponse:
    return CatalogValidationResponse(
        source_catalog="src",
        target_catalog="tgt",
        status=ValidationStatus.FAIL,
        schemas=[schema_result],
    )


def test_missing_tables_rendered_as_missing_from_target():
    schema = SchemaValidationResult(
        schema_name="bronze",
        status=ValidationStatus.FAIL,
        missing_tables=["only_in_source"],
        extra_tables=[],
        tables=[],
    )
    rows = _build_table_rows(_response(schema))

    assert len(rows) == 1
    assert rows[0][_SOURCE_TABLE_COL] == "only_in_source"
    assert rows[0][_STATUS_COL] == "MISSING_FROM_TARGET"


def test_extra_tables_rendered_as_extra_in_target_not_missing():
    """Regression test: a table present in target but with no matching
    source (e.g. a Databricks table with no matching blob) must be
    labeled EXTRA_IN_TARGET - previously extra_tables was never rendered
    at all, and blob_discovery.py separately mislabeled these as
    MISSING_FROM_TARGET by putting them in the wrong field."""
    schema = SchemaValidationResult(
        schema_name="bronze",
        status=ValidationStatus.FAIL,
        missing_tables=[],
        extra_tables=["only_in_target"],
        tables=[],
    )
    rows = _build_table_rows(_response(schema))

    assert len(rows) == 1
    assert rows[0][_SOURCE_TABLE_COL] == "only_in_target"
    assert rows[0][_STATUS_COL] == "EXTRA_IN_TARGET"


def test_both_missing_and_extra_tables_rendered_distinctly():
    schema = SchemaValidationResult(
        schema_name="bronze",
        status=ValidationStatus.FAIL,
        missing_tables=["only_in_source"],
        extra_tables=["only_in_target"],
        tables=[],
    )
    rows = _build_table_rows(_response(schema))

    statuses_by_table = {r[_SOURCE_TABLE_COL]: r[_STATUS_COL] for r in rows}
    assert statuses_by_table == {
        "only_in_source": "MISSING_FROM_TARGET",
        "only_in_target": "EXTRA_IN_TARGET",
    }


def test_matched_table_and_extra_table_both_appear():
    matched = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.PASS,
    )
    schema = SchemaValidationResult(
        schema_name="bronze",
        status=ValidationStatus.FAIL,
        missing_tables=[],
        extra_tables=["orphan_table"],
        tables=[matched],
    )
    rows = _build_table_rows(_response(schema))

    statuses_by_table = {r[_SOURCE_TABLE_COL]: r[_STATUS_COL] for r in rows}
    assert statuses_by_table["customers"] == "PASS"
    assert statuses_by_table["orphan_table"] == "EXTRA_IN_TARGET"


def test_all_rows_match_header_width_including_missing_and_extra():
    """Missing/extra table rows are built as a fixed-length filler list
    ("" * N) rather than field-by-field - a new TABLE_HEADERS column
    (e.g. Tier Reached) must have that filler count updated in lockstep,
    or every missing/extra row silently misaligns against the header."""
    matched = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.PASS,
        tier_reached=ValidationTier.FINGERPRINT,
    )
    schema = SchemaValidationResult(
        schema_name="bronze",
        status=ValidationStatus.FAIL,
        missing_tables=["gone"],
        extra_tables=["orphan"],
        tables=[matched],
    )
    rows = _build_table_rows(_response(schema))

    for row in rows:
        assert len(row) == len(TABLE_HEADERS)


def test_tier_reached_populated_for_matched_table():
    matched = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.PASS,
        tier_reached=ValidationTier.FINGERPRINT,
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.PASS, tables=[matched],
    )
    rows = _build_table_rows(_response(schema))

    tier_col = TABLE_HEADERS.index("Tier Reached")
    assert rows[0][tier_col] == "FINGERPRINT"


def test_schema_blocking_gets_top_priority_suggestion():
    blocked = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        schema_blocking=True, tier_reached=ValidationTier.SCHEMA_BLOCKED,
        missing_columns=["age"], columns_status=ValidationStatus.FAIL,
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[blocked],
    )
    rows = _build_suggestion_rows(_response(schema))

    issue_types = [r[3] for r in rows]
    assert "Blocked at Schema Check" in issue_types
    assert issue_types[0] == "Blocked at Schema Check"


def test_partition_column_shows_bucket_summary_when_partitioned():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        tier_reached=ValidationTier.ROW_HASH,
        partitioned=True, partition_column="region",
        partition_buckets_total=10, partition_buckets_culprit=2,
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    partition_col = TABLE_HEADERS.index("Partition")
    assert rows[0][partition_col] == "region (2/10 buckets differed)"


def test_partition_column_shows_skip_reason_when_not_partitioned():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        tier_reached=ValidationTier.ROW_HASH,
        partitioned=False, partition_skip_reason="user declined or non-interactive run",
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    partition_col = TABLE_HEADERS.index("Partition")
    assert rows[0][partition_col] == "not partitioned (user declined or non-interactive run)"


def test_partition_column_blank_when_not_offered():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.PASS,
        tier_reached=ValidationTier.FINGERPRINT,
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.PASS, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    partition_col = TABLE_HEADERS.index("Partition")
    assert rows[0][partition_col] == ""


def test_row_hash_mismatches_sheet_includes_partition_bucket_column():
    mismatch = RowHashMismatch(
        primary_key="42", source_hash="aaa", target_hash="bbb",
        status="MISMATCH", partition_bucket="west",
    )
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            row_hash_mismatches=[mismatch],
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_row_hash_rows(_response(schema))

    bucket_col = ROW_HASH_HEADERS.index("Partition Bucket")
    assert rows[0][bucket_col] == "west"


def test_row_hash_mismatches_sheet_blank_bucket_when_unpartitioned():
    mismatch = RowHashMismatch(
        primary_key="42", source_hash="aaa", target_hash="bbb", status="MISMATCH",
    )
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            row_hash_mismatches=[mismatch],
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_row_hash_rows(_response(schema))

    bucket_col = ROW_HASH_HEADERS.index("Partition Bucket")
    assert rows[0][bucket_col] == ""


def test_mismatch_headers_include_verified_column():
    assert MISMATCH_HEADERS[-1] == "Verified"
    assert len(MISMATCH_HEADERS) == 9


def test_mismatch_rows_show_yes_for_verified_detail():
    detail = RowMismatchDetail(
        schema_name="bronze", table="customers",
        primary_key={"id": 2}, mismatch_column="name",
        source_value="old", target_value="new",
        verified=True,
    )
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            sample_changed_detail=[detail],
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_mismatch_rows(_response(schema))

    verified_col = MISMATCH_HEADERS.index("Verified")
    assert rows[0][verified_col] == "Yes"


def test_mismatch_rows_label_row_number_fallback_detail_as_unverified():
    detail = RowMismatchDetail(
        schema_name="bronze", table="customers",
        primary_key={"row_number": 2}, mismatch_column="name",
        source_value="old", target_value="new",
        verified=False,
    )
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            sample_changed_detail=[detail],
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_mismatch_rows(_response(schema))

    verified_col = MISMATCH_HEADERS.index("Verified")
    assert rows[0][verified_col] == "No (row-number, unverified)"


def test_suggestion_wording_uses_row_position_for_row_number_fallback():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            key_columns=["row_number"],
            row_hash_mismatch_count=2,
            row_hash_mismatch_percentage=0.5,
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_suggestion_rows(_response(schema))

    row_mismatch_rows = [r for r in rows if r[3] == "Row Data Mismatch"]
    assert len(row_mismatch_rows) == 1
    suggestion_text = row_mismatch_rows[0][4]
    assert "row position" in suggestion_text
    assert "best-effort" in suggestion_text
    assert "unverified" in suggestion_text


def test_suggestion_wording_uses_primary_key_for_real_key_path():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            key_columns=["id"],
            row_hash_mismatch_count=2,
            row_hash_mismatch_percentage=0.5,
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_suggestion_rows(_response(schema))

    row_mismatch_rows = [r for r in rows if r[3] == "Row Data Mismatch"]
    assert len(row_mismatch_rows) == 1
    suggestion_text = row_mismatch_rows[0][4]
    assert "by primary key" in suggestion_text
    assert "row position" not in suggestion_text


# ---------------------------------------------------------------------------
# Mismatch Count/% (Table Validation sheet): the tiered fail-fast funnel
# never populates source_only_rows/target_only_rows/changed_rows (those
# only ever came from the legacy FULL-mode EXCEPT/hash-join path) - these
# two columns must fall back to the row-hash comparison's own count/%
# instead of staying permanently blank for every tiered-pipeline table.
# ---------------------------------------------------------------------------
def test_mismatch_count_falls_back_to_row_hash_count_for_tiered_pipeline():
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        row_count_source=100, row_count_target=100,
        data=DataValidationResult(
            mode="STATISTICS", status=ValidationStatus.FAIL,
            row_hash_mismatch_count=3,
            row_hash_mismatch_percentage=3.0,
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    mismatch_count_col = TABLE_HEADERS.index("Mismatch Count")
    mismatch_pct_col = TABLE_HEADERS.index("Mismatch %")
    assert rows[0][mismatch_count_col] == 3
    assert rows[0][mismatch_pct_col] == "3.00%"


def test_mismatch_count_prefers_legacy_fields_when_present():
    """A legacy FULL-mode result (source_only_rows/target_only_rows/
    changed_rows populated) must still take priority over the row-hash
    fallback - the fallback only applies when those are all None."""
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.FAIL,
        row_count_source=100, row_count_target=100,
        data=DataValidationResult(
            mode="FULL", status=ValidationStatus.FAIL,
            source_only_rows=1, target_only_rows=0, changed_rows=2,
            row_hash_mismatch_count=99,  # must be ignored here
            row_hash_mismatch_percentage=99.0,
        ),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.FAIL, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    mismatch_count_col = TABLE_HEADERS.index("Mismatch Count")
    mismatch_pct_col = TABLE_HEADERS.index("Mismatch %")
    assert rows[0][mismatch_count_col] == 3
    assert rows[0][mismatch_pct_col] == "3.00%"


def test_mismatch_count_blank_when_no_mismatches_found_at_all():
    """A clean PASS table (row_hash_mismatch_count=0, the pydantic
    default) must not show a misleading fallback count - stay blank."""
    table = TableValidationResult(
        schema_name="bronze", table="customers", status=ValidationStatus.PASS,
        row_count_source=100, row_count_target=100,
        data=DataValidationResult(mode="STATISTICS", status=ValidationStatus.PASS),
    )
    schema = SchemaValidationResult(
        schema_name="bronze", status=ValidationStatus.PASS, tables=[table],
    )
    rows = _build_table_rows(_response(schema))

    mismatch_count_col = TABLE_HEADERS.index("Mismatch Count")
    mismatch_pct_col = TABLE_HEADERS.index("Mismatch %")
    assert rows[0][mismatch_count_col] == ""
    assert rows[0][mismatch_pct_col] == ""
