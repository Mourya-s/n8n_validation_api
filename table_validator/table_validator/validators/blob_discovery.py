"""
Azure Blob Storage -> Databricks catalog: multi-blob discovery and
lightweight comparison.

Unlike AzureCsvValidator (which validates ONE named blob against ONE named
Databricks table, with full row-hash/row-level comparison), this module
answers "which blobs in a container correspond to which tables in a
Databricks catalog?" - list blobs matching folder_prefix/file_pattern,
infer a candidate table name from each blob's filename (strip path and
extension), and intersect those inferred names against the catalog's
actual table names, using the same list-and-intersect-by-name approach
CatalogValidator already uses for schema/table discovery.

BlobCatalogValidator.validate()'s target_schema is optional, same
"blank means compare everything" convention as
CatalogValidator/AzureSqlValidator: if left unset, every schema in
target_catalog (excluding information_schema) is listed and blobs are
matched against each in turn, aggregating every schema's results onto
one CatalogValidationResponse.

Comparison for each matched (blob, table) pair is intentionally lighter
than AzureCsvValidator's: row count, plus column name/type comparison
only. Row-hash / row-level data-mismatch comparison is deliberately out
of scope here - AzureCsvValidator's hash-formatting rules
(_format_value_for_hash) were built and empirically verified against
CSV-sourced pandas dtypes specifically; extending them to every blob
format matched by a wildcard pattern (Parquet's native int64 vs CSV's
string-then-inferred columns, for example) without the same empirical
verification risks the exact class of silent-wrong-mismatch bug fixed
earlier for decimal-vs-bigint and NVARCHAR-vs-VARCHAR encoding. A user
who wants full row-hash comparison against one specific file can still
use the existing single-blob CsvTableValidationRequest/AzureCsvValidator
path directly.

Column comparison reuses AzureCsvValidator's static helpers
(_infer_databricks_type, _types_compatible) rather than reimplementing
them, since those are already correct and don't depend on any per-file
state.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import List, Optional, Tuple

import pandas as pd

from table_validator.connectors.azure_connector import AzureConnector
from table_validator.connectors.databricks_connector import DatabricksConnector
from table_validator.models import (
    CatalogValidationResponse,
    ColumnValidationResult,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
    ValidationSummary,
)
from table_validator.validators.row_validator import AzureCsvValidator

logger = logging.getLogger(__name__)


def _infer_table_name(blob_path: str) -> str:
    """Strip directory path and extension from a blob path to get a
    candidate table name, e.g. 'validation/2024/Customers.csv' -> 'Customers'."""
    base_name = blob_path.rsplit("/", 1)[-1]
    return base_name.rsplit(".", 1)[0] if "." in base_name else base_name


def discover_blob_table_matches(
    blob_names: List[str],
    catalog_tables: List[str],
) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """
    Match blobs to catalog tables by inferred name (case-insensitive),
    mirroring CatalogValidator.compare_tables' intersection-by-name
    approach.

    Returns (matched_pairs, blob_only, table_only) where matched_pairs is
    a list of (blob_path, table_name) using each side's own real name/
    casing, blob_only is blob paths with no matching table, and
    table_only is catalog table names with no matching blob.
    """
    blob_by_inferred_name = {_infer_table_name(b).lower(): b for b in blob_names}
    table_by_name = {t.lower(): t for t in catalog_tables}

    common_keys = set(blob_by_inferred_name) & set(table_by_name)
    blob_only_keys = set(blob_by_inferred_name) - set(table_by_name)
    table_only_keys = set(table_by_name) - set(blob_by_inferred_name)

    matched_pairs = sorted(
        ((blob_by_inferred_name[k], table_by_name[k]) for k in common_keys),
        key=lambda pair: pair[1],
    )
    blob_only = sorted(blob_by_inferred_name[k] for k in blob_only_keys)
    table_only = sorted(table_by_name[k] for k in table_only_keys)

    return matched_pairs, blob_only, table_only


class BlobCatalogValidator:
    """
    Validates every blob in a container (optionally scoped by
    folder_prefix/file_pattern) that matches a same-named table in a
    Databricks catalog schema, by inferred filename. Row count + column
    name/type comparison only (see module docstring for why row-hash
    comparison is out of scope here).
    """

    # Databricks-managed system schema, present in every catalog - never a
    # real migration target. Same convention as CatalogValidator.
    _EXCLUDED_SCHEMAS = {"information_schema"}

    def __init__(
        self,
        azure_connector: AzureConnector,
        databricks_connector: DatabricksConnector,
    ) -> None:
        self.azure = azure_connector
        self.databricks = databricks_connector
        logger.debug("BlobCatalogValidator initialised")

    def validate(
        self,
        target_catalog: str,
        target_schema: Optional[str] = None,
        folder_prefix: Optional[str] = None,
        file_pattern: Optional[str] = None,
        blob_path: Optional[str] = None,
        target_table: Optional[str] = None,
    ) -> CatalogValidationResponse:
        """
        target_schema=None means "match blobs against every schema in
        target_catalog" - lists all schemas (excluding
        information_schema), matches blobs against each in turn, and
        aggregates every schema's results onto one response, same
        "blank means compare everything" convention as
        CatalogValidator/AzureSqlValidator.

        If both blob_path and target_table are given, filename-to-table
        discovery is bypassed entirely: that exact blob is compared
        directly against that exact table, even if their names don't
        match (same "explicit pair skips name matching" idea as
        AzureSqlValidator's schema_map/table_map). target_schema is
        still required in this mode, since a bare table name alone
        doesn't identify a Databricks schema.
        """
        start = time.perf_counter()
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if blob_path and target_table:
            if not target_schema:
                return self._error_response(
                    target_catalog, run_timestamp, start,
                    "target_table.schema is required when both an exact "
                    "source blob and target table are given.",
                )
            table_result = self._validate_pair(
                blob_path, target_catalog, target_schema, target_table
            )
            schema_result = SchemaValidationResult(
                schema_name=target_schema,
                status=table_result.status,
                missing_tables=[],
                extra_tables=[],
                tables=[table_result],
            )
            execution_time = round(time.perf_counter() - start, 3)
            logger.info(
                "Blob validation finished (explicit pair) | status=%s | duration=%.3fs",
                table_result.status, execution_time,
            )
            return CatalogValidationResponse(
                source_catalog=f"blob:{self.azure.container_name}",
                target_catalog=target_catalog,
                status=table_result.status,
                validation_timestamp=run_timestamp,
                execution_time_seconds=execution_time,
                summary=ValidationSummary(
                    total_schemas=1,
                    passed_schemas=1 if table_result.status == ValidationStatus.PASS else 0,
                    failed_schemas=0 if table_result.status == ValidationStatus.PASS else 1,
                    total_tables=1,
                    passed_tables=1 if table_result.status == ValidationStatus.PASS else 0,
                    failed_tables=0 if table_result.status == ValidationStatus.PASS else 1,
                    error_tables=1 if table_result.status == ValidationStatus.ERROR else 0,
                ),
                schemas=[schema_result],
            )

        try:
            blob_names = self.azure.list_blobs(folder_prefix, file_pattern)
        except Exception as exc:
            logger.exception("Failed to list blobs")
            return self._error_response(
                target_catalog, run_timestamp, start,
                f"Unable to list blobs: {exc}",
            )

        if target_schema:
            schemas_to_check = [target_schema]
        else:
            try:
                all_schemas = self.databricks.get_schemas(target_catalog)
            except Exception as exc:
                logger.exception("Failed to list schemas for '%s'", target_catalog)
                return self._error_response(
                    target_catalog, run_timestamp, start,
                    f"Unable to list target schemas: {exc}",
                )
            schemas_to_check = sorted(
                s for s in all_schemas if s.lower() not in self._EXCLUDED_SCHEMAS
            )
            logger.info(
                "No schema configured - matching blobs against all %d schema(s) "
                "in '%s': %s",
                len(schemas_to_check), target_catalog, schemas_to_check,
            )

        schema_results: List[SchemaValidationResult] = [
            self._validate_schema(blob_names, target_catalog, schema_name)
            for schema_name in schemas_to_check
        ]

        overall_status = _calculate_overall_status([s.status for s in schema_results])

        summary = ValidationSummary(
            total_schemas=len(schema_results),
        )
        for schema_result in schema_results:
            if schema_result.status == ValidationStatus.PASS:
                summary.passed_schemas += 1
            else:
                summary.failed_schemas += 1
            summary.total_tables += len(schema_result.tables)
            summary.missing_tables += len(schema_result.missing_tables)
            summary.extra_tables += len(schema_result.extra_tables)
            for table in schema_result.tables:
                if table.status == ValidationStatus.PASS:
                    summary.passed_tables += 1
                elif table.status == ValidationStatus.ERROR:
                    summary.error_tables += 1
                    summary.failed_tables += 1
                elif table.status == ValidationStatus.FAIL:
                    summary.failed_tables += 1

        execution_time = round(time.perf_counter() - start, 3)
        logger.info(
            "Blob validation finished | status=%s | duration=%.3fs",
            overall_status, execution_time,
        )

        return CatalogValidationResponse(
            source_catalog=f"blob:{self.azure.container_name}",
            target_catalog=target_catalog,
            status=overall_status,
            validation_timestamp=run_timestamp,
            execution_time_seconds=execution_time,
            missing_schemas=[],
            extra_schemas=[],
            summary=summary,
            schemas=schema_results,
        )

    def _validate_schema(
        self,
        blob_names: List[str],
        target_catalog: str,
        target_schema: str,
    ) -> SchemaValidationResult:
        try:
            catalog_tables = self.databricks.get_tables(target_catalog, target_schema)
        except Exception as exc:
            logger.exception(
                "Failed to list tables for '%s.%s'", target_catalog, target_schema
            )
            return SchemaValidationResult(
                schema_name=target_schema,
                status=ValidationStatus.ERROR,
                error=f"Unable to list target tables: {exc}",
            )

        matched_pairs, blob_only, table_only = discover_blob_table_matches(
            blob_names, catalog_tables
        )

        # Same "missing" / "extra" convention as CatalogValidator/
        # AzureSqlValidator: missing_tables = present in source, absent in
        # target; extra_tables = present in target, absent in source. Here
        # the blob is the source and the Databricks table is the target,
        # so a blob with no matching table is "missing from target"
        # (blob_only), and a table with no matching blob is "extra in
        # target" (table_only) - NOT the other way around.
        if blob_only:
            logger.warning(
                "Blobs with no matching table in '%s.%s' (missing from target): %s",
                target_catalog, target_schema, blob_only,
            )
        if table_only:
            logger.warning(
                "Tables in '%s.%s' with no matching blob (extra in target): %s",
                target_catalog, target_schema, table_only,
            )
        logger.info(
            "Found %d matching blob/table pair(s) in '%s.%s'.",
            len(matched_pairs), target_catalog, target_schema,
        )

        table_results: List[TableValidationResult] = [
            self._validate_pair(blob_path, target_catalog, target_schema, table_name)
            for blob_path, table_name in matched_pairs
        ]

        statuses = [t.status for t in table_results]
        if blob_only:
            statuses.append(ValidationStatus.FAIL)
        status = _calculate_overall_status(statuses)

        return SchemaValidationResult(
            schema_name=target_schema,
            status=status,
            missing_tables=blob_only,
            extra_tables=table_only,
            tables=table_results,
        )

    def _error_response(
        self,
        target_catalog: str,
        run_timestamp: str,
        start: float,
        error: str,
    ) -> CatalogValidationResponse:
        return CatalogValidationResponse(
            source_catalog=f"blob:{self.azure.container_name}",
            target_catalog=target_catalog,
            status=ValidationStatus.ERROR,
            validation_timestamp=run_timestamp,
            execution_time_seconds=round(time.perf_counter() - start, 3),
            error=error,
        )

    def _validate_pair(
        self,
        blob_path: str,
        target_catalog: str,
        target_schema: str,
        target_table: str,
    ) -> TableValidationResult:
        result = TableValidationResult(schema_name=target_schema, table=target_table)

        try:
            source_df = self.azure.read_csv(blob_path)
        except Exception as exc:
            logger.exception("Failed to read blob '%s'", blob_path)
            result.status = ValidationStatus.ERROR
            result.error = f"Unable to read blob '{blob_path}': {exc}"
            return result

        try:
            target_schema_df = self.databricks.get_table_schema(
                target_catalog, target_schema, target_table
            )
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s.%s'",
                target_catalog, target_schema, target_table,
            )
            result.status = ValidationStatus.ERROR
            result.error = f"Unable to retrieve column metadata: {exc}"
            return result

        # Column names (common / missing / extra), reusing the same
        # normalization convention as CatalogValidator/AzureCsvValidator.
        src_cols = {str(c).lower(): str(c) for c in source_df.columns}
        tgt_cols = {
            str(c).lower(): str(c) for c in target_schema_df["column_name"]
        }
        missing_cols = sorted(set(src_cols) - set(tgt_cols))
        extra_cols = sorted(set(tgt_cols) - set(src_cols))
        common_cols = sorted(src_cols[k] for k in (set(src_cols) & set(tgt_cols)))

        result.missing_columns = missing_cols
        result.extra_columns = extra_cols
        result.columns_status = (
            ValidationStatus.FAIL if (missing_cols or extra_cols) else ValidationStatus.PASS
        )

        if not common_cols:
            result.status = ValidationStatus.FAIL
            result.error = "No common columns between blob and target table"
            return result

        tgt_type_by_col = {
            str(r["column_name"]).lower(): str(r["data_type"])
            for _, r in target_schema_df.iterrows()
        }

        column_results: List[ColumnValidationResult] = []
        dtype_statuses = []
        for col in common_cols:
            src_type = AzureCsvValidator._infer_databricks_type(source_df[col])
            tgt_type = tgt_type_by_col.get(col.lower(), "")
            status = (
                ValidationStatus.PASS
                if AzureCsvValidator._types_compatible(src_type, tgt_type)
                else ValidationStatus.FAIL
            )
            dtype_statuses.append(status)
            column_results.append(
                ColumnValidationResult(
                    column=col,
                    status=status,
                    source_data_type=src_type,
                    target_data_type=tgt_type,
                    data_type_status=status,
                )
            )

        result.columns = column_results
        result.data_types_status = _calculate_overall_status(dtype_statuses)

        # Row count only - no row-hash / row-level comparison (see module
        # docstring).
        try:
            src_count = len(source_df)
            tgt_count = self.databricks.get_row_count(
                target_catalog, target_schema, target_table
            )
            result.row_count_source = src_count
            result.row_count_target = tgt_count
            result.row_count_difference = tgt_count - src_count
            result.row_count_status = (
                ValidationStatus.PASS if src_count == tgt_count else ValidationStatus.FAIL
            )
        except Exception as exc:
            logger.exception("Failed to compute row count for '%s'", target_table)
            result.row_count_status = ValidationStatus.ERROR
            result.error = f"Row count failed: {exc}"

        result.status = _calculate_overall_status(
            [
                result.columns_status,
                result.data_types_status,
                result.row_count_status,
            ]
        )

        return result


def _calculate_overall_status(
    statuses: List[Optional[ValidationStatus]],
) -> ValidationStatus:
    """Same precedence rule as CatalogValidator.calculate_overall_status."""
    clean = [s for s in statuses if s is not None]
    if not clean:
        return ValidationStatus.SKIPPED
    if any(s == ValidationStatus.ERROR for s in clean):
        return ValidationStatus.ERROR
    if any(s == ValidationStatus.FAIL for s in clean):
        return ValidationStatus.FAIL
    if all(s == ValidationStatus.SKIPPED for s in clean):
        return ValidationStatus.SKIPPED
    return ValidationStatus.PASS
