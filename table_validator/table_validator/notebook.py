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


def _parse_source_target(name: str, label: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse either "catalog.schema.table" (single-table comparison - the
    returned table is never None) or "catalog.schema" (schema-wide sweep
    - every identically-named table in the schema, optionally renamed via
    validate_tables()'s table_map param; the returned table is None) into
    its parts. Raises ValueError with a clear message for anything else
    (wrong part count, or any blank part) - there is no existing parser
    to reuse for this shape (the CLI always keeps catalog/schema/table as
    separate ValidatorConfig fields, never a single combined string).
    """
    parts = [p.strip() for p in name.split(".")]
    if len(parts) == 3 and all(parts):
        return parts[0], parts[1], parts[2]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1], None
    raise ValueError(
        f"{label} must be a fully-qualified 'catalog.schema.table' name "
        f"(or 'catalog.schema' for a schema-wide sweep), got: {name!r}"
    )


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
    table_map: Optional[Dict[str, str]] = None,
    spark: Optional["SparkSession"] = None,
    ignore_columns: Optional[List[str]] = None,
    only_columns: Optional[List[str]] = None,
    column_map: Optional[Dict[str, str]] = None,
    ignore_datatype_columns: Optional[List[str]] = None,
    row_filter: Optional[str] = None,
    source_row_filter: Optional[str] = None,
    target_row_filter: Optional[str] = None,
) -> ValidationResult:
    """
    Validate a source against a target, using the notebook's own ambient
    SparkSession - no workspace URL, personal access token, or SQL
    Warehouse HTTP path needed. Two modes, selected by whether `source`/
    `target` name a table:

    - Single-table: `"catalog.schema.table"` on both sides - compares
      exactly that one source table against that one target table (the
      two table names don't need to match; a differing name is handled
      automatically).
    - Schema-wide sweep: `"catalog.schema"` (no table) on BOTH sides -
      compares every table with an identical name common to both
      schemas, auto-discovered with zero further configuration (the same
      mechanism the CLI's own blank-table config triggers). Pass
      `table_map` to additionally compare specific tables that were
      renamed between source and target (e.g. source has 'cust', target
      has 'customers') - unmapped tables are still matched by identical
      name as usual. `primary_key` is rejected in this mode (a single key
      can't apply to every table in the sweep - compare one table at a
      time if you need row-level detail via a real key).

    Mixing the two (a table named on only one side) is rejected with a
    clear error, since it's always ambiguous.

    Runs the same full-depth tiered comparison the CLI's own `--mode full`
    (the CLI's default) does - schema/column checks, row counts, null/
    distinct/min-max statistics, whole-table fingerprint, and (if a
    difference is found) row-hash diff plus column-level mismatch detail
    - since CatalogValidationRequest's own field defaults (max_tier=
    ValidationTier.COLUMN_DIFF, enabled_validations=all four types) already
    equal the CLI's "full" mode; this function does not override either.

    `primary_key`, if given (single-table mode only), is used for row-
    level comparison instead of the synthetic ROW_NUMBER() fallback -
    same tradeoff as the CLI's own `primary_key` config field.
    `table_map` (sweep mode only), `ignore_columns`, `only_columns`,
    `column_map`, `ignore_datatype_columns` mirror the identically-named
    CatalogValidationRequest fields.

    `ignore_datatype_columns`, if given, names columns (case-insensitive,
    checked against both the source and target spelling) whose data type
    should never fail the comparison - a genuine type difference on one
    of these columns is reported as SKIPPED rather than PASS/FAIL, and
    (critically) is excluded from Tier 0's BLOCKING cross-family-type-
    change check, so a type change alone on one of these columns never
    aborts the table before row-level comparison even runs. The column's
    other checks (nullable, null/distinct/min-max statistics, row-hash)
    still run normally - only its data type is ignored. Useful when a
    migration is known to have changed a column's type on purpose (e.g.
    STRING -> INT) and you only want to check the row/value-level data,
    not re-litigate the type change every run.

    `row_filter`/`source_row_filter`/`target_row_filter`, if given,
    restrict comparison to only the rows matching a SQL WHERE-clause
    fragment - e.g. `row_filter="id > 20 and id < 100"` or
    `row_filter="gender = 'male'"` - instead of the whole table. Row
    count, statistics, whole-table fingerprint, row-hash diff, and
    column-level mismatch detail are ALL scoped to just the matching
    rows on both sides. `row_filter` applies to both sides equally;
    `source_row_filter`/`target_row_filter` additionally AND onto just
    that one side (all three can combine - e.g. a common status filter
    plus a source-only id range). Each fragment is used as-is (wrapped
    in parentheses for safe AND-combination), not parsed or validated -
    a malformed fragment surfaces as a normal SQL error the first time a
    query actually runs, the same as any other SQL text mistake would.
    Valid in both single-table and schema-wide sweep mode (a sweep
    filtering every matched table by the same condition, e.g.
    `gender = 'male'`, is a legitimate use case).

    Returns a ValidationResult: `print(result)` alone shows a compact
    plain-text summary (no rich/box-drawing formatting, safe to print
    directly in a notebook cell, and correctly listing every table when
    a sweep matched more than one); `print(result.table_validation)`,
    `.column_validation`, `.data_mismatches`, `.row_hash_mismatches`,
    `.mismatch_categories`, `.suggestions` each print one specific
    sheet's data as a plain-text table, mirroring the Excel report's own
    sheets one-for-one.
    """
    src_catalog, src_schema, src_table = _parse_source_target(source, "source")
    tgt_catalog, tgt_schema, tgt_table = _parse_source_target(target, "target")

    connector = SparkConnector(spark=spark)
    if row_filter or source_row_filter or target_row_filter:
        connector.set_row_filters(
            common=row_filter, source=source_row_filter, target=target_row_filter,
        )

    schema_map: Dict[str, str] = (
        {src_schema: tgt_schema} if src_schema.lower() != tgt_schema.lower() else {}
    )

    if src_table and tgt_table:
        # Single-table mode.
        tables_restriction: Optional[List[str]] = [src_table]
        resolved_table_map: Dict[str, str] = (
            {src_table: tgt_table} if src_table.lower() != tgt_table.lower() else {}
        )
        primary_keys: Dict[str, list] = {}
        if primary_key:
            primary_keys[tgt_table] = primary_key
            primary_keys[f"{tgt_schema}.{tgt_table}"] = primary_key
    elif not src_table and not tgt_table:
        # Schema-wide sweep - leaving `tables` unset (None) is what
        # triggers CatalogValidator's own auto-discovery of every table
        # common to both schemas by identical name; table_map (if given)
        # additionally pairs up specific renamed tables, mirroring
        # cli/main.py's own schema-sweep branch exactly.
        tables_restriction = None
        resolved_table_map = dict(table_map or {})
        if primary_key:
            raise ValueError(
                "primary_key is only meaningful for a single-table "
                "comparison (both source and target naming a table) - "
                "for a schema-wide sweep, each table would need its own "
                "key. Compare one table at a time if you need a primary "
                "key for row-level detail."
            )
        primary_keys = {}
    else:
        raise ValueError(
            "source and target must either BOTH name a table "
            "('catalog.schema.table') or BOTH omit it ('catalog.schema' "
            "for a schema-wide sweep) - got a table on only one side."
        )

    request = CatalogValidationRequest(
        source_catalog=src_catalog,
        target_catalog=tgt_catalog,
        schemas=[src_schema],
        schema_map=schema_map,
        tables=tables_restriction,
        table_map=resolved_table_map,
        primary_keys=primary_keys,
        ignore_columns=ignore_columns or [],
        only_columns=only_columns,
        column_map=column_map or {},
        ignore_datatype_columns=ignore_datatype_columns or [],
        # max_tier and enabled_validations deliberately left unset - the
        # Pydantic model defaults already equal the CLI's own "full" mode
        # (see this function's docstring above).
    )

    response = CatalogValidator(connector).compare_catalogs(request)

    return ValidationResult(response)


# ---------------------------------------------------------------------------
# validate_tables.help() - a code/notebook-usage reference, callable
# directly on the function (`from table_validator import validate_tables;
# validate_tables.help()`), separate from `tablevalidator info` (the CLI's
# own usage reference, cli/main.py's info() command) - each covers only the
# approach it belongs to, since a CLI-only user has no use for keyword-
# argument syntax and a notebook-only user never touches config.yaml.
# ---------------------------------------------------------------------------
_HELP_TEXT = """
validate_tables() - notebook-native table validation
------------------------------------------------------
Compares a source table against a target Databricks table (or every
identically-named table across two schemas), from inside a Databricks
notebook cell - no workspace URL, personal access token, or SQL Warehouse
HTTP path to set up, since it reuses the notebook's own already-
authenticated Spark session. Runs the exact same full-depth comparison
engine as the `tablevalidator` CLI.

Install and import:
    %pip install table-validator
    from table_validator import validate_tables

Single-table comparison ("catalog.schema.table" on both sides):
    result = validate_tables(
        "catalog1.schema1.table1",
        "catalog1.schema1.table2",
    )

Schema-wide sweep (no table named - "catalog.schema" on both sides):
compares every identically-named table in that schema in one call:
    result = validate_tables("catalog1.bronze", "catalog2.silver")

    If some tables were renamed between source and target, pass table_map
    (source name -> target name) - unmapped tables are still matched by
    identical name as usual:
        result = validate_tables(
            "catalog1.bronze", "catalog2.silver",
            table_map={"cust": "customers", "ord": "orders"},
        )

Keyword arguments (all optional):
    primary_key            Real key column(s) for row-level comparison
                            instead of the synthetic ROW_NUMBER() fallback.
                            Single-table mode only.
                                primary_key=["id"]
    table_map               Renamed-table pairs, schema-wide sweep mode only.
                                table_map={"cust": "customers"}
    spark                    An explicit SparkSession, if you're not
                             running inside an actual Databricks notebook
                             (e.g. local development). Auto-detected
                             otherwise.
    ignore_columns           Columns excluded entirely from every check.
                                ignore_columns=["updated_at"]
    only_columns             If set, ONLY these columns (plus the primary
                             key, if any) are compared.
                                only_columns=["id", "name"]
    column_map               Renamed-column pairs (source name -> target
                             name), for a column that doesn't share a name
                             between source and target.
                                column_map={"cust_id": "customer_id"}
    ignore_datatype_columns  Columns whose data-type mismatch is ignored -
                             their other checks (nullable, statistics,
                             row-hash) still run normally.
                                ignore_datatype_columns=["legacy_flag"]
    row_filter               Restrict comparison to rows matching a SQL
                             WHERE-fragment, on BOTH sides - row count,
                             statistics, fingerprint, and row-hash/column
                             diff are all scoped to just the matching rows.
                                row_filter="id > 20 and id < 100"
                                row_filter="gender = 'male'"
    source_row_filter        Like row_filter, but applied only to the
    target_row_filter        source/target side respectively - combine
                             with row_filter (ANDed) or use instead of it
                             when the condition must differ per side.
                                validate_tables(
                                    "cat.sch.orders", "cat2.sch.orders",
                                    source_row_filter="id > 20",
                                    target_row_filter="id > 15",
                                )
                             Works in both single-table and schema-wide
                             sweep mode. Each fragment is used as-is, not
                             parsed or validated - a typo surfaces as a
                             normal SQL error when the query runs.

Reading the result:
    print(result)                     Compact plain-text summary.
    print(result.table_validation)    One row per table (Excel report's
                                       "Table Validation" sheet).
    print(result.column_validation)   One row per column.
    print(result.data_mismatches)     One row per mismatched cell (needs a
                                       real primary key or the row-number
                                       fallback, FULL-depth detail only).
    print(result.row_hash_mismatches) One row per mismatched primary key.
    print(result.mismatch_categories) Root-cause breakdown (NULL_MISMATCH,
                                       STRING_TRUNCATION, ...).
    print(result.suggestions)         Plain-English fix suggestions.

    Each of these is a small table object - .headers/.rows for
    programmatic access, or .to_dataframe() for a real pandas DataFrame.
    result.response is the raw CatalogValidationResponse for full
    programmatic access beyond the sheet breakdown.

Call validate_tables.help() any time to print this again.
""".strip("\n")


def _print_validate_tables_help() -> None:
    print(_HELP_TEXT)


validate_tables.help = _print_validate_tables_help
