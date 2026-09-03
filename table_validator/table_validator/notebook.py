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
    print(result)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from table_validator.cli.summary_table import SummaryData, summary_from_response
from table_validator.connectors.spark_connector import SparkConnector
from table_validator.models import CatalogValidationRequest, CatalogValidationResponse
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


def _format_plain_text(data: SummaryData, result: CatalogValidationResponse) -> str:
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


def validate_tables(
    source: str,
    target: str,
    *,
    primary_key: Optional[List[str]] = None,
    spark: Optional["SparkSession"] = None,
    ignore_columns: Optional[List[str]] = None,
    only_columns: Optional[List[str]] = None,
    column_map: Optional[Dict[str, str]] = None,
) -> str:
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

    Returns a plain-text summary (no rich/box-drawing formatting) meant to
    be printed directly in a notebook cell, e.g. `print(result)`.
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

    result = CatalogValidator(connector).compare_catalogs(request)

    summary = summary_from_response(result)
    return _format_plain_text(summary, result)
