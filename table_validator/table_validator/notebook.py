"""
Notebook-native entry point: validate_tables().

A second, independent way to use this package's validation engine,
alongside (not instead of) the `tablevalidator` CLI - pick whichever fits
your context:

- CLI (`tablevalidator configure` + `tablevalidator validate`): a workspace
  URL, personal access token, and SQL Warehouse HTTP path, stored in
  ~/.table_validator/.env/config.yaml - the right shape for a scheduled or
  scripted run from outside Databricks.
- validate_tables() (this module): called directly from a Databricks
  notebook cell, with zero separate auth to configure - it reuses the
  notebook's own already-authenticated ambient SparkSession (see
  connectors/spark_connector.py) instead of opening a new SQL Warehouse
  connection.

Both drive the exact same CatalogValidator comparison engine
(validators/catalog_validator.py), so the depth of checking and the
result shape are consistent between the two - this module adds no new
comparison logic of its own, only request-building and plain-text
rendering around the existing engine.

Typical usage, from a Databricks notebook cell:

    %pip install table-validator

    from table_validator import validate_tables

    result = validate_tables(
        "catalog1.schema1.table1",
        "catalog1.schema1.table2",
    )
    print(result)                     # compact summary
    print(result.table_validation)    # one row per table (like the Excel
                                       # report's "Table Validation" sheet)
    print(result.column_validation)   # one row per column
    print(result.data_mismatches)     # one row per mismatched cell
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from table_validator.cli.summary_table import SummaryData, summary_from_response
from table_validator.connectors.spark_connector import SparkConnector
from table_validator.models import CatalogValidationRequest, CatalogValidationResponse
from table_validator.reports.excel_report import (
    CATEGORY_SUMMARY_HEADERS,
    COLUMN_HEADERS,
    MISMATCH_HEADERS,
    ROW_HASH_HEADERS,
    SUGGESTION_HEADERS,
    TABLE_HEADERS,
    _build_column_rows,
    _build_mismatch_category_summary,
    _build_mismatch_rows,
    _build_row_hash_rows,
    _build_suggestion_rows,
    _build_table_rows,
)
from table_validator.validators.catalog_validator import CatalogValidator

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _parse_fqtn(name: str, label: str) -> Tuple[str, str, str]:
    """
    Parse a fully-qualified 'catalog.schema.table' string into its three
    parts. Raises ValueError with a clear message if `name` isn't exactly
    three non-empty, dot-separated parts - there is no existing parser to
    reuse for this shape (the CLI always keeps catalog/schema/table as
    separate ValidatorConfig fields, never a single combined string).
    """
    parts = name.split(".")
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise ValueError(
            f"{label} must be a fully-qualified 'catalog.schema.table' "
            f"name, got: {name!r}"
        )
    catalog, schema, table = (p.strip() for p in parts)
    return catalog, schema, table


def _format_summary_text(data: SummaryData, result: CatalogValidationResponse) -> str:
    """
    Render a CatalogValidationResponse as plain text - mirroring the
    CONTENT of cli/main.py's own console summary (_print_summary: per-
    table status lines, overall status, aggregate table counts, error
    text), but with zero rich/box-drawing/ANSI content, so it prints and
    copies cleanly from a Databricks notebook cell's plain stdout.

    print_summary_table (cli/summary_table.py) is deliberately NOT reused
    here even via a captured string buffer - rich's Table renders Unicode
    box-drawing characters regardless of destination unless the caller
    passes box=None (which print_summary_table does not do, and changing
    it would alter the CLI's own real terminal output) - so a small,
    independent plain-text formatter is used instead.
    """
    lines: List[str] = []

    per_table_lines: List[str] = []
    for schema in result.schemas:
        for table in schema.tables:
            per_table_lines.append(f"  {schema.schema_name}.{table.table}: {table.status.value}")

    if len(per_table_lines) > 1:
        lines.append("Per-table results:")
        lines.extend(per_table_lines)
        lines.append("")

    lines.append(f"Overall status: {data.overall_status}")
    lines.append(
        f"Tables: {data.total_tables} total, {data.passed_tables} passed, "
        f"{data.failed_tables} failed, {data.error_tables} error, "
        f"{data.skipped_tables} skipped"
    )
    if result.error:
        lines.append(f"Error: {result.error}")

    return "\n".join(lines)


# Past this rendered line width, a one-row-per-record table wraps onto an
# unreadably wide, sideways-scrolling line in a notebook cell (e.g.
# table_validation's 24 mostly-short-value columns easily clears 200
# characters per row) - ResultTable switches to a vertical, one-field-
# per-line block per record instead once a row would exceed this width,
# rather than ever truncating/hiding columns (every column stays
# visible, just laid out top-to-bottom instead of left-to-right). Judged
# by rendered width rather than a raw column count, since a handful of
# columns holding long values (e.g. Data Mismatches' Source/Target Value)
# can be just as unreadable as many columns of short ones, and
# conversely a wider column count of short values (e.g. Data Mismatches'
# own 9 columns) can still fit comfortably on one line.
_MAX_ROW_WIDTH_FOR_ROW_LAYOUT = 180


class ResultTable:
    """
    One sheet's worth of rows (mirrors an Excel report sheet - Table
    Validation, Column Validation, Data Mismatches, etc.), printable
    directly as plain text.

    A narrow sheet (few columns, e.g. Data Mismatches/Suggestions) renders
    as one aligned table via pandas' to_string() - no rich/box-drawing
    characters, so it copies cleanly from a notebook cell. A wide sheet
    (e.g. Table Validation's 24 columns) would otherwise wrap onto one
    unreadable, sideways-scrolling line - past
    _MAX_COLUMNS_FOR_ROW_LAYOUT, it instead renders as one labeled block
    per record, one "Header: value" line per field, blank line between
    records - every column stays visible, just top-to-bottom instead of
    left-to-right.

    Also exposes .rows/.headers for programmatic access, and
    .to_dataframe() for anyone who wants the real pandas DataFrame (e.g.
    to filter/sort/export it themselves) regardless of which layout
    str()/print() chooses.
    """

    def __init__(self, headers: List[str], rows: List[List[Any]]) -> None:
        self.headers = headers
        self.rows = rows

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.headers)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return str(self)

    def _as_row_table(self) -> str:
        # index=False: a notebook reader has no use for pandas' own
        # synthetic 0..N row index here, only the sheet's real columns.
        return self.to_dataframe().to_string(index=False)

    def _as_vertical_blocks(self) -> str:
        label_width = max(len(h) for h in self.headers)
        blocks: List[str] = []
        for i, row in enumerate(self.rows, start=1):
            lines = [f"--- Row {i} of {len(self.rows)} ---"]
            for header, value in zip(self.headers, row):
                lines.append(f"{header.ljust(label_width)} : {value}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def __str__(self) -> str:
        if not self.rows:
            return "(no rows)"
        row_table = self._as_row_table()
        widest_line = max(len(line) for line in row_table.splitlines())
        if widest_line > _MAX_ROW_WIDTH_FOR_ROW_LAYOUT:
            return self._as_vertical_blocks()
        return row_table


class ValidationResult:
    """
    Wraps a CatalogValidationResponse from validate_tables() with the same
    sheet breakdown the Excel report offers, so a caller can choose which
    one to look at - `print(result)` alone shows a compact summary;
    `print(result.table_validation)` / `.column_validation` / etc. show
    one specific sheet's data as a plain-text table, exactly mirroring
    the Excel report's own sheets (built from the very same row-building
    functions, reports/excel_report.py - never a second implementation of
    "what a row looks like").

    `.response` is the raw CatalogValidationResponse, for anyone who wants
    full programmatic access beyond the sheet breakdown.
    """

    def __init__(self, response: CatalogValidationResponse) -> None:
        self.response = response
        self._summary_data = summary_from_response(response)

    @property
    def table_validation(self) -> ResultTable:
        """One row per table - mirrors the Excel report's "Table
        Validation" sheet: schema match, row counts, mismatch counts,
        overall status, etc."""
        return ResultTable(TABLE_HEADERS, _build_table_rows(self.response))

    @property
    def column_validation(self) -> ResultTable:
        """One row per column - mirrors the Excel report's "Column
        Validation" sheet: per-column type/nullable/null-count/distinct-
        count/min-max status."""
        return ResultTable(COLUMN_HEADERS, _build_column_rows(self.response))

    @property
    def data_mismatches(self) -> ResultTable:
        """One row per mismatched cell - mirrors the Excel report's "Data
        Mismatches" sheet. Only populated in FULL mode with a real
        primary key or the row-number fallback; empty otherwise."""
        return ResultTable(MISMATCH_HEADERS, _build_mismatch_rows(self.response))

    @property
    def row_hash_mismatches(self) -> ResultTable:
        """One row per mismatched primary key - mirrors the Excel
        report's "Row Hash Mismatches" sheet."""
        return ResultTable(ROW_HASH_HEADERS, _build_row_hash_rows(self.response))

    @property
    def mismatch_categories(self) -> ResultTable:
        """One row per root-cause category (NULL_MISMATCH,
        STRING_TRUNCATION, CASE_DIFFERENCE, ...) - mirrors the Excel
        report's "Mismatch Categories" sheet's summary table."""
        summaries = _build_mismatch_category_summary(self.response)
        rows = [[s.category, s.count, f"{s.pct:.2f}%", s.top_table, s.top_column] for s in summaries]
        return ResultTable(CATEGORY_SUMMARY_HEADERS, rows)

    @property
    def suggestions(self) -> ResultTable:
        """One plain-English suggestion per issue found on a FAILed/
        ERRORed table - mirrors the Excel report's "Suggestions" sheet."""
        return ResultTable(SUGGESTION_HEADERS, _build_suggestion_rows(self.response))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return _format_summary_text(self._summary_data, self.response)


def validate_tables(
    source: str,
    target: str,
    *,
    primary_key: Optional[List[str]] = None,
    spark: Optional["SparkSession"] = None,
    ignore_columns: Optional[List[str]] = None,
    only_columns: Optional[List[str]] = None,
    column_map: Optional[Dict[str, str]] = None,
) -> ValidationResult:
    """
    Validate one source table against one target table, using the
    notebook's own ambient SparkSession - no workspace URL, personal
    access token, or SQL Warehouse HTTP path needed.

    `source`/`target` are fully-qualified `"catalog.schema.table"` names.
    Runs the same full-depth tiered comparison the CLI's own `--mode full`
    (the CLI's default) does - schema/column checks, row counts, null/
    distinct/min-max statistics, whole-table fingerprint, and (if a
    difference is found) row-hash diff plus column-level mismatch detail
    - since CatalogValidationRequest's own field defaults (max_tier=
    ValidationTier.COLUMN_DIFF, enabled_validations=all four types) already
    equal the CLI's "full" mode; this function does not override either.

    `primary_key`, if given, is used for row-level comparison instead of
    the synthetic ROW_NUMBER() fallback - same tradeoff as the CLI's own
    `primary_key` config field. `ignore_columns`/`only_columns`/
    `column_map` mirror the identically-named CatalogValidationRequest
    fields.

    Returns a ValidationResult: `print(result)` alone shows a compact
    plain-text summary (no rich/box-drawing formatting, safe to print
    directly in a notebook cell); `print(result.table_validation)`,
    `.column_validation`, `.data_mismatches`, `.row_hash_mismatches`,
    `.mismatch_categories`, `.suggestions` each print one specific
    sheet's data as a plain-text table, mirroring the Excel report's own
    sheets one-for-one.
    """
    src_catalog, src_schema, src_table = _parse_fqtn(source, "source")
    tgt_catalog, tgt_schema, tgt_table = _parse_fqtn(target, "target")

    connector = SparkConnector(spark=spark)

    schema_map: Dict[str, str] = (
        {src_schema: tgt_schema} if src_schema.lower() != tgt_schema.lower() else {}
    )
    table_map: Dict[str, str] = (
        {src_table: tgt_table} if src_table.lower() != tgt_table.lower() else {}
    )

    primary_keys: Dict[str, list] = {}
    if primary_key:
        primary_keys[tgt_table] = primary_key
        primary_keys[f"{tgt_schema}.{tgt_table}"] = primary_key

    request = CatalogValidationRequest(
        source_catalog=src_catalog,
        target_catalog=tgt_catalog,
        schemas=[src_schema],
        schema_map=schema_map,
        tables=[src_table],
        table_map=table_map,
        primary_keys=primary_keys,
        ignore_columns=ignore_columns or [],
        only_columns=only_columns,
        column_map=column_map or {},
        # max_tier and enabled_validations deliberately left unset - the
        # Pydantic model defaults already equal the CLI's own "full" mode
        # (see this function's docstring above).
    )

    response = CatalogValidator(connector).compare_catalogs(request)

    return ValidationResult(response)
