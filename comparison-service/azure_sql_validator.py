"""
Azure SQL Database -> Databricks catalog validator.

Multi-table version of the Azure Blob CSV validator: every table in the
Azure SQL database (optionally restricted via request.schemas/tables) is
matched by name against a like-named table in the target Databricks
catalog, mirroring CatalogValidator's schema/table matching. Runs the
same validation stages - column names/order, data types, row counts,
null/distinct/min-max statistics, row-hash comparison, row-level data
mismatch detail - against every matched pair.

Unlike the CSV path, both sides here are real SQL engines, so row-hash
comparison is pushed down on BOTH sides instead of loading anything into
pandas: AzureSqlConnector.get_row_hashes (T-SQL HASHBYTES) for the source,
DatabricksConnector.get_row_hashes (sha2/concat_ws) for the target. The
formatting used to build each side's hash-input string is chosen to
produce byte-identical digests for equal logical values - verified
empirically against real data before being trusted (see the "verified"
notes on AzureSqlConnector._hash_string_expr).

Returns the same CatalogValidationResponse shape as CatalogValidator (a
list of SchemaValidationResult, each wrapping its TableValidationResult
list), so report_generator.py needs no changes.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from azure_sql_connector import AzureSqlConnector
from databricks_connector import DatabricksConnector, values_differ
from models import (
    AzureSqlValidationRequest,
    CatalogValidationResponse,
    ColumnValidationResult,
    DataCompareMode,
    DataValidationResult,
    RowHashMismatch,
    RowMismatchDetail,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
    ValidationSummary,
)

logger = logging.getLogger(__name__)


def _calculate_overall_status(statuses: List[Optional[ValidationStatus]]) -> ValidationStatus:
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


class AzureSqlValidator:
    """
    Validates every common table between an Azure SQL Database and a
    Databricks catalog. Responsible for comparison/decision logic only;
    AzureSqlConnector and DatabricksConnector only do I/O.
    """

    def __init__(
        self,
        azure_sql_connector: AzureSqlConnector,
        databricks_connector: DatabricksConnector,
    ) -> None:
        self.azure_sql = azure_sql_connector
        self.databricks = databricks_connector
        logger.debug("AzureSqlValidator initialised")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def validate(self, request: AzureSqlValidationRequest) -> CatalogValidationResponse:
        start = time.perf_counter()
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logger.info(
            "Starting Azure SQL validation | target_catalog=%s", request.target_catalog,
        )

        try:
            common_pairs, missing_schemas, extra_schemas = self._compare_schemas(request)
        except Exception as exc:
            logger.exception("Failed to compare schemas")
            return CatalogValidationResponse(
                source_catalog="azure_sql",
                target_catalog=request.target_catalog,
                status=ValidationStatus.ERROR,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Unable to compare schemas: {exc}",
            )

        if request.schemas:
            wanted = {s.lower() for s in request.schemas}
            common_pairs = [
                (src, tgt) for src, tgt in common_pairs if src.lower() in wanted
            ]

        schema_results: List[SchemaValidationResult] = []
        for source_schema, target_schema in common_pairs:
            schema_results.append(self._validate_schema(request, source_schema, target_schema))

        summary = self._build_summary(schema_results, missing_schemas, extra_schemas)
        overall_status = _calculate_overall_status(
            [s.status for s in schema_results]
            + ([ValidationStatus.FAIL] if missing_schemas else [])
        )

        execution_time = round(time.perf_counter() - start, 3)

        logger.info(
            "Azure SQL validation finished | status=%s | duration=%.3fs",
            overall_status, execution_time,
        )

        return CatalogValidationResponse(
            source_catalog="azure_sql",
            target_catalog=request.target_catalog,
            status=overall_status,
            validation_timestamp=run_timestamp,
            execution_time_seconds=execution_time,
            missing_schemas=missing_schemas,
            extra_schemas=extra_schemas,
            summary=summary,
            schemas=schema_results,
        )

    # ------------------------------------------------------------------
    # Schema / table matching (mirrors CatalogValidator.compare_schemas/tables)
    # ------------------------------------------------------------------
    def _compare_schemas(
        self, request: AzureSqlValidationRequest,
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        """
        Returns (common_pairs, missing, extra) where common_pairs is a list
        of (source_schema, target_schema) name pairs - either identical
        names, or resolved via request.schema_map when the Azure SQL and
        Databricks sides use different schema names for the same logical
        target. `missing` (present in Azure SQL, no match in Databricks)
        and `extra` (present in Databricks, unmatched) are still reported
        by their own-side name for display.
        """
        source_schemas = set(self.azure_sql.get_schemas())
        target_schemas = set(self.databricks.get_schemas(request.target_catalog))

        schema_map_lower = {k.lower(): v for k, v in request.schema_map.items()}

        common_pairs: List[Tuple[str, str]] = []
        matched_source: Set[str] = set()
        matched_target: Set[str] = set()

        for src in source_schemas:
            mapped_target = schema_map_lower.get(src.lower())
            if mapped_target is not None:
                candidate = next(
                    (t for t in target_schemas if t.lower() == mapped_target.lower()), None
                )
            else:
                candidate = next(
                    (t for t in target_schemas if t.lower() == src.lower()), None
                )
            if candidate is not None:
                common_pairs.append((src, candidate))
                matched_source.add(src)
                matched_target.add(candidate)

        missing = sorted(source_schemas - matched_source)
        extra = sorted(target_schemas - matched_target)
        common_pairs.sort(key=lambda pair: pair[0])
        return common_pairs, missing, extra

    def _compare_tables(
        self, request: AzureSqlValidationRequest, source_schema: str, target_schema: str,
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
        """
        Returns (common_pairs, missing, extra) where common_pairs is a list
        of (source_table_name, target_table_name) pairs preserving each
        side's own casing - table names are matched case-insensitively,
        but the two sides' real casing must both be preserved and used
        for their own subsequent lookups (Databricks' information_schema
        table-name match is case-sensitive, and Azure SQL's default
        collation usually is too).
        """
        source_tables = {t.lower(): t for t in self.azure_sql.get_tables(source_schema)}
        target_tables = {
            t.lower(): t for t in self.databricks.get_tables(request.target_catalog, target_schema)
        }

        common_keys = set(source_tables) & set(target_tables)
        missing_keys = set(source_tables) - set(target_tables)
        extra_keys = set(target_tables) - set(source_tables)

        common_pairs = sorted(
            ((source_tables[k], target_tables[k]) for k in common_keys),
            key=lambda pair: pair[0],
        )
        missing = sorted(source_tables[k] for k in missing_keys)
        extra = sorted(target_tables[k] for k in extra_keys)
        return common_pairs, missing, extra

    def _validate_schema(
        self, request: AzureSqlValidationRequest, source_schema: str, target_schema: str,
    ) -> SchemaValidationResult:
        try:
            common_table_pairs, missing_tables, extra_tables = self._compare_tables(
                request, source_schema, target_schema,
            )
        except Exception as exc:
            logger.exception("Failed to compare tables for schema '%s'", source_schema)
            return SchemaValidationResult(
                schema_name=target_schema,
                status=ValidationStatus.ERROR,
                error=f"Unable to compare tables: {exc}",
            )

        if request.tables:
            wanted = {t.lower() for t in request.tables}
            common_table_pairs = [
                (src, tgt) for src, tgt in common_table_pairs if src.lower() in wanted
            ]

        table_results: List[TableValidationResult] = []
        for source_table, target_table in common_table_pairs:
            table_results.append(
                self._validate_table(request, source_schema, target_schema, source_table, target_table)
            )

        statuses = [t.status for t in table_results]
        if missing_tables:
            statuses.append(ValidationStatus.FAIL)

        status = _calculate_overall_status(statuses)

        return SchemaValidationResult(
            schema_name=target_schema,
            status=status,
            missing_tables=missing_tables,
            extra_tables=extra_tables,
            tables=table_results,
        )

    # ------------------------------------------------------------------
    # Per-table pipeline
    # ------------------------------------------------------------------
    def _validate_table(
        self,
        request: AzureSqlValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
    ) -> TableValidationResult:
        result = TableValidationResult(schema_name=target_schema, table=target_table)

        try:
            source_schema_df = self.azure_sql.get_table_schema(source_schema, source_table)
            target_schema_df = self.databricks.get_table_schema(
                request.target_catalog, target_schema, target_table
            )
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s'", source_schema, source_table
            )
            result.status = ValidationStatus.ERROR
            result.error = f"Unable to retrieve column metadata: {exc}"
            return result

        ignore = {c.lower() for c in (request.ignore_columns or [])}

        missing_cols, extra_cols, common_cols = self._compare_columns(
            source_schema_df, target_schema_df, request.case_sensitive_columns, ignore,
        )
        result.missing_columns = missing_cols
        result.extra_columns = extra_cols
        result.columns_status = (
            ValidationStatus.FAIL if (missing_cols or extra_cols) else ValidationStatus.PASS
        )

        if not common_cols:
            result.status = ValidationStatus.FAIL
            result.error = "No common columns between source and target"
            return result

        source_order = [
            c for c in source_schema_df["column_name"].tolist()
            if c.lower() in {x.lower() for x in common_cols}
        ]
        target_order = [
            c for c in target_schema_df["column_name"].tolist()
            if c.lower() in {x.lower() for x in common_cols}
        ]
        result.source_column_order = source_order
        result.target_column_order = target_order

        if request.validate_column_order:
            order_matches = [c.lower() for c in source_order] == [c.lower() for c in target_order]
            result.column_order_status = (
                ValidationStatus.PASS if order_matches else ValidationStatus.FAIL
            )
        else:
            result.column_order_status = ValidationStatus.SKIPPED

        src_by_col = {
            str(r["column_name"]).lower(): r for _, r in source_schema_df.iterrows()
        }
        tgt_by_col = {
            str(r["column_name"]).lower(): r for _, r in target_schema_df.iterrows()
        }

        min_max_columns = [
            c for c in common_cols
            if self.azure_sql.is_min_max_eligible(str(src_by_col.get(c.lower(), {}).get("data_type", "")))
        ]

        try:
            source_stats = self.azure_sql.get_column_statistics(
                source_schema, source_table, common_cols, min_max_columns,
            )
            target_stats = self.databricks.get_column_statistics(
                request.target_catalog, target_schema, target_table, common_cols, min_max_columns,
            )
            stats_error = None
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s.%s'", source_schema, source_table
            )
            source_stats, target_stats = {}, {}
            stats_error = str(exc)

        column_results: List[ColumnValidationResult] = []
        dtype_statuses, nullable_statuses = [], []
        null_statuses, distinct_statuses, minmax_statuses = [], [], []

        for col in common_cols:
            key = col.lower()
            src_row = src_by_col.get(key, {})
            tgt_row = tgt_by_col.get(key, {})

            col_result = ColumnValidationResult(column=col, status=ValidationStatus.PASS)

            src_type = str(src_row.get("data_type"))
            tgt_type = str(tgt_row.get("data_type"))
            col_result.source_data_type = src_type
            col_result.target_data_type = tgt_type
            col_result.data_type_status = (
                ValidationStatus.PASS if self._types_compatible(src_type, tgt_type)
                else ValidationStatus.FAIL
            )
            dtype_statuses.append(col_result.data_type_status)

            src_null = bool(src_row.get("is_nullable"))
            tgt_null = bool(tgt_row.get("is_nullable"))
            col_result.source_nullable = src_null
            col_result.target_nullable = tgt_null
            col_result.nullable_status = (
                ValidationStatus.PASS if src_null == tgt_null else ValidationStatus.FAIL
            )
            nullable_statuses.append(col_result.nullable_status)

            if stats_error:
                col_result.null_count_status = ValidationStatus.ERROR
                col_result.distinct_count_status = ValidationStatus.ERROR
                col_result.error = stats_error
            else:
                s_stat = source_stats.get(col, {})
                t_stat = target_stats.get(col, {})

                col_result.source_null_count = s_stat.get("null_count")
                col_result.target_null_count = t_stat.get("null_count")
                col_result.null_count_status = (
                    ValidationStatus.PASS
                    if col_result.source_null_count == col_result.target_null_count
                    else ValidationStatus.FAIL
                )

                col_result.source_distinct_count = s_stat.get("distinct_count")
                col_result.target_distinct_count = t_stat.get("distinct_count")
                col_result.distinct_count_status = (
                    ValidationStatus.PASS
                    if col_result.source_distinct_count == col_result.target_distinct_count
                    else ValidationStatus.FAIL
                )

                if col in min_max_columns:
                    col_result.source_min = s_stat.get("min")
                    col_result.source_max = s_stat.get("max")
                    col_result.target_min = t_stat.get("min")
                    col_result.target_max = t_stat.get("max")
                    col_result.min_max_status = (
                        ValidationStatus.FAIL
                        if (values_differ(col_result.source_min, col_result.target_min)
                            or values_differ(col_result.source_max, col_result.target_max))
                        else ValidationStatus.PASS
                    )
                else:
                    col_result.min_max_status = ValidationStatus.SKIPPED

            null_statuses.append(col_result.null_count_status)
            distinct_statuses.append(col_result.distinct_count_status)
            minmax_statuses.append(col_result.min_max_status)

            col_result.status = _calculate_overall_status(
                [
                    col_result.data_type_status,
                    col_result.nullable_status,
                    col_result.null_count_status,
                    col_result.distinct_count_status,
                    col_result.min_max_status,
                ]
            )
            column_results.append(col_result)

        result.columns = column_results
        result.data_types_status = _calculate_overall_status(dtype_statuses)
        result.nullable_status = _calculate_overall_status(nullable_statuses)
        result.null_counts_status = _calculate_overall_status(null_statuses)
        result.distinct_counts_status = _calculate_overall_status(distinct_statuses)
        result.min_max_status = _calculate_overall_status(minmax_statuses)

        try:
            src_count = self.azure_sql.get_row_count(source_schema, source_table)
            tgt_count = self.databricks.get_row_count(request.target_catalog, target_schema, target_table)
            result.row_count_source = src_count
            result.row_count_target = tgt_count
            result.row_count_difference = tgt_count - src_count
            result.row_count_status = (
                ValidationStatus.PASS if src_count == tgt_count else ValidationStatus.FAIL
            )
        except Exception as exc:
            logger.exception("Failed to compute row counts for '%s.%s'", source_schema, source_table)
            result.row_count_status = ValidationStatus.ERROR
            result.error = f"Row count failed: {exc}"

        result.data = self._compare_data(
            request, source_schema, target_schema, source_table, target_table,
            common_cols, src_by_col, tgt_by_col,
        )

        result.status = _calculate_overall_status(
            [
                result.columns_status,
                result.column_order_status,
                result.row_count_status,
                result.data_types_status,
                result.nullable_status,
                result.null_counts_status,
                result.distinct_counts_status,
                result.min_max_status,
                result.data.status if result.data else ValidationStatus.SKIPPED,
            ]
        )

        return result

    # ------------------------------------------------------------------
    # Column comparison (mirrors CatalogValidator.compare_columns)
    # ------------------------------------------------------------------
    @staticmethod
    def _compare_columns(
        source_schema_df: pd.DataFrame,
        target_schema_df: pd.DataFrame,
        case_sensitive: bool,
        ignore: Set[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        def norm(name: str) -> str:
            return name if case_sensitive else name.lower()

        src_cols = {
            norm(str(c)): str(c) for c in source_schema_df["column_name"]
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
        }
        tgt_cols = {
            norm(str(c)): str(c) for c in target_schema_df["column_name"]
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
        }

        missing = sorted(set(src_cols) - set(tgt_cols))
        extra = sorted(set(tgt_cols) - set(src_cols))
        common = sorted(src_cols[k] for k in (set(src_cols) & set(tgt_cols)))
        return missing, extra, common

    @staticmethod
    def _types_compatible(source_type: str, target_type: str) -> bool:
        """
        SQL Server and Databricks use different type-name vocabularies for
        the same underlying kind of value (e.g. SQL Server 'varchar' vs
        Databricks 'string', 'int' vs 'int'/'bigint') - this is
        informational, not authoritative; the actual values are still
        compared via row-hash regardless of what this reports.
        """
        s = source_type.strip().lower()
        t = target_type.split("(")[0].strip().lower()

        if s == t:
            return True

        numeric = {
            "tinyint", "smallint", "int", "bigint",
            "float", "real", "decimal", "numeric", "money", "smallmoney",
            "double",
        }
        stringy = {"char", "varchar", "nchar", "nvarchar", "text", "ntext", "string"}
        datey = {"date", "datetime", "datetime2", "smalldatetime", "timestamp"}
        booly = {"bit", "boolean"}

        for group in (numeric, stringy, datey, booly):
            if s in group and t in group:
                return True
        return False

    _SQL_DECIMAL_TYPES = {"decimal", "numeric", "money", "smallmoney"}
    _SQL_INTEGER_TARGET_TYPES = {"tinyint", "smallint", "int", "bigint"}

    @classmethod
    def _effective_source_types(
        cls,
        value_columns: List[str],
        src_by_col: Dict[str, Any],
        tgt_by_col: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Build the {column: sql_server_type} map passed to
        AzureSqlConnector.get_row_hashes/get_row_hashes_by_row_number,
        substituting the synthetic type "decimal_as_integer" whenever the
        source is decimal/numeric/money but the target's real Databricks
        type is a plain integer (bigint/int/smallint/tinyint) - otherwise
        every row would hash as "changed" purely from Azure SQL keeping
        a fractional format (e.g. '45000.00') that Databricks' integer
        column never produces ('45000'), even when the numeric value is
        identical (found empirically comparing HASHBYTES vs sha2() output
        for a decimal-vs-bigint Salary column on real data).
        """
        result: Dict[str, str] = {}
        for col in value_columns:
            src_type = str(src_by_col.get(col.lower(), {}).get("data_type", "")).strip().lower()
            tgt_type = str(tgt_by_col.get(col.lower(), {}).get("data_type", "")).split("(")[0].strip().lower()

            if src_type in cls._SQL_DECIMAL_TYPES and tgt_type in cls._SQL_INTEGER_TARGET_TYPES:
                result[col] = "decimal_as_integer"
            else:
                result[col] = src_type

        return result

    # ------------------------------------------------------------------
    # Row-hash comparison + row-level data mismatch detail
    # ------------------------------------------------------------------
    def _compare_data(
        self,
        request: AzureSqlValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_columns: List[str],
        src_by_col: Dict[str, Any],
        tgt_by_col: Dict[str, Any],
    ) -> DataValidationResult:
        mode = request.data_compare_mode
        key = f"{target_schema}.{target_table}"
        primary_key = request.primary_keys.get(key) or request.primary_keys.get(target_table)

        logger.info(
            "[azuresql-row-hash] table=%s.%s | resolved_key_columns=%s",
            target_schema, target_table, primary_key,
        )

        using_row_number_fallback = False

        if not primary_key:
            using_row_number_fallback = True
            key_columns_for_result: List[str] = ["row_number"]
            value_columns = sorted(common_columns)

            logger.info(
                "[azuresql-row-hash] no key configured for '%s' - falling back to "
                "ROW_NUMBER()-based comparison (ORDER BY every common column). This "
                "is a best-effort fallback, not a substitute for a real key: it can "
                "only meaningfully compare tables whose row SETS are otherwise "
                "identical.", key,
            )

            source_types = self._effective_source_types(value_columns, src_by_col, tgt_by_col)

            try:
                source_hashes = self.azure_sql.get_row_hashes_by_row_number(
                    source_schema, source_table, value_columns, source_types,
                )
            except Exception as exc:
                logger.exception("Failed to compute source row-number hashes for '%s'", key)
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR, key_columns=key_columns_for_result,
                    error=f"Source row-number hashing failed: {exc}",
                )

            try:
                target_hashes = self.databricks.get_row_hashes_by_row_number(
                    request.target_catalog, target_schema, target_table, value_columns,
                )
            except Exception as exc:
                logger.exception("Failed to fetch target row-number hashes for '%s'", key)
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR, key_columns=key_columns_for_result,
                    error=f"Target row-number hashing failed: {exc}",
                )

            primary_key = ["row_number"]
        else:
            key_columns_for_result = primary_key

            missing_keys = [
                k for k in primary_key if k.lower() not in {c.lower() for c in common_columns}
            ]
            if missing_keys:
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR, key_columns=primary_key,
                    error=f"Configured key column(s) not found as common columns: {missing_keys}",
                )

            value_columns = sorted(
                c for c in common_columns if c.lower() not in {k.lower() for k in primary_key}
            )

            source_types = self._effective_source_types(value_columns, src_by_col, tgt_by_col)

            try:
                source_hashes = self.azure_sql.get_row_hashes(
                    source_schema, source_table, value_columns, primary_key, source_types,
                )
            except Exception as exc:
                logger.exception("Failed to compute source row hashes for '%s'", key)
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR, key_columns=primary_key,
                    error=f"Source row hashing failed: {exc}",
                )

            try:
                target_hashes = self.databricks.get_row_hashes(
                    request.target_catalog, target_schema, target_table, value_columns, primary_key,
                )
            except Exception as exc:
                logger.exception("Failed to fetch target row hashes for '%s'", key)
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR, key_columns=primary_key,
                    error=f"Target row hashing failed: {exc}",
                )

        logger.info(
            "[azuresql-row-hash] fetched | table=%s.%s | source_rows=%d | target_rows=%d | "
            "row_number_fallback=%s",
            target_schema, target_table, len(source_hashes), len(target_hashes),
            using_row_number_fallback,
        )

        mismatches, mismatch_count, mismatch_pct = self._compare_row_hashes(
            source_hashes, target_hashes, primary_key,
        )

        logger.info(
            "[azuresql-row-hash] table=%s.%s | mismatch_count=%d | mismatch_pct=%.2f%%",
            target_schema, target_table, mismatch_count, mismatch_pct,
        )

        data_result = DataValidationResult(
            mode=mode,
            status=ValidationStatus.FAIL if mismatch_count > 0 else ValidationStatus.PASS,
            key_columns=key_columns_for_result,
            row_hash_mismatches=mismatches,
            row_hash_mismatch_count=mismatch_count,
            row_hash_mismatch_percentage=mismatch_pct,
            note=(
                "No primary key configured - row-level comparison used a synthetic "
                "ROW_NUMBER() (ORDER BY every common column) instead of a real key. "
                "This only detects differences reliably when both sides contain the "
                "same set of rows; it cannot pinpoint which specific record changed "
                "the way a real key can, and row numbers are not stable identifiers "
                "across runs."
                if using_row_number_fallback else None
            ),
        )

        if mode == DataCompareMode.FULL and mismatch_count > 0 and not using_row_number_fallback:
            data_result.sample_changed_detail = self._changed_row_detail(
                request, source_schema, target_schema, source_table, target_table,
                primary_key, value_columns, source_hashes, target_hashes,
            )

        return data_result

    @staticmethod
    def _compare_row_hashes(
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
        primary_key_cols: List[str],
    ) -> Tuple[List[RowHashMismatch], int, float]:
        # Identical join/classify logic to CatalogValidator.compare_row_hashes.
        def _display_key(row: pd.Series) -> str:
            return "|".join(str(row[k]) for k in primary_key_cols)

        def _key_tuple(row: pd.Series) -> tuple:
            return tuple(row[k] for k in primary_key_cols)

        source_by_key = {_key_tuple(r): r for _, r in source_hashes.iterrows()}
        target_by_key = {_key_tuple(r): r for _, r in target_hashes.iterrows()}

        all_keys = set(source_by_key) | set(target_by_key)
        mismatches: List[RowHashMismatch] = []

        for key_tuple in all_keys:
            src_row = source_by_key.get(key_tuple)
            tgt_row = target_by_key.get(key_tuple)

            if src_row is not None and tgt_row is None:
                mismatches.append(RowHashMismatch(
                    primary_key=_display_key(src_row),
                    source_hash=str(src_row["row_hash"]), target_hash="",
                    status="MISSING_IN_TARGET",
                ))
            elif src_row is None and tgt_row is not None:
                mismatches.append(RowHashMismatch(
                    primary_key=_display_key(tgt_row),
                    source_hash="", target_hash=str(tgt_row["row_hash"]),
                    status="MISSING_IN_SOURCE",
                ))
            elif src_row is not None and tgt_row is not None:
                if src_row["row_hash"] != tgt_row["row_hash"]:
                    mismatches.append(RowHashMismatch(
                        primary_key=_display_key(src_row),
                        source_hash=str(src_row["row_hash"]),
                        target_hash=str(tgt_row["row_hash"]),
                        status="MISMATCH",
                    ))

        total = len(all_keys)
        count = len(mismatches)
        pct = (count / total) * 100 if total else 0.0
        return mismatches, count, pct

    def _changed_row_detail(
        self,
        request: AzureSqlValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        primary_key: List[str],
        value_columns: List[str],
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
    ) -> List[RowMismatchDetail]:
        """
        For MISMATCH keys, fetch the actual source and target row values
        for a bounded sample and diff column-by-column, same approach as
        DatabricksConnector._changed_row_detail.
        """
        limit_samples = request.max_sample_rows

        src_hash_by_key = {
            tuple(r[k] for k in primary_key): r["row_hash"] for _, r in source_hashes.iterrows()
        }
        tgt_hash_by_key = {
            tuple(r[k] for k in primary_key): r["row_hash"] for _, r in target_hashes.iterrows()
        }

        mismatch_keys: List[tuple] = []
        for key_tuple, src_hash in src_hash_by_key.items():
            tgt_hash = tgt_hash_by_key.get(key_tuple)
            if tgt_hash is not None and tgt_hash != src_hash:
                mismatch_keys.append(key_tuple)
            if len(mismatch_keys) >= limit_samples:
                break

        if not mismatch_keys:
            return []

        source_rows = self._fetch_rows_for_keys(
            self.azure_sql, source_schema, source_table, primary_key, value_columns, mismatch_keys,
        )
        target_rows = self._fetch_rows_for_keys(
            self.databricks, target_schema, target_table, primary_key, value_columns, mismatch_keys,
            catalog=request.target_catalog,
        )

        source_by_key = {tuple(r[k] for k in primary_key): r for r in source_rows}
        target_by_key = {tuple(r[k] for k in primary_key): r for r in target_rows}

        detail: List[RowMismatchDetail] = []
        for key_tuple in mismatch_keys:
            src_row = source_by_key.get(key_tuple)
            tgt_row = target_by_key.get(key_tuple)
            if src_row is None or tgt_row is None:
                continue

            mismatched_columns = [
                col for col in value_columns
                if values_differ(src_row.get(col), tgt_row.get(col))
            ]
            if not mismatched_columns:
                mismatched_columns = list(value_columns)

            key_dict = {k: src_row.get(k) for k in primary_key}

            for col in mismatched_columns:
                detail.append(
                    RowMismatchDetail(
                        schema_name=target_schema,
                        table=target_table,
                        primary_key=key_dict,
                        mismatch_column=col,
                        source_value=src_row.get(col),
                        target_value=tgt_row.get(col),
                        source_row_hash=src_hash_by_key.get(key_tuple),
                        target_row_hash=tgt_hash_by_key.get(key_tuple),
                    )
                )

        return detail

    @staticmethod
    def _fetch_rows_for_keys(
        connector,
        schema_name: str,
        table_name: str,
        primary_key: List[str],
        value_columns: List[str],
        keys: List[tuple],
        catalog: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched fetch of rows for a bounded set of primary keys, from
        either connector (AzureSqlConnector or DatabricksConnector - both
        expose _quote_ident/_qualify/execute_query with compatible
        signatures, just different quoting/qualification conventions).
        """
        if not keys:
            return []

        key_idents = [connector._quote_ident(k) for k in primary_key]
        key_list = ", ".join(key_idents)
        value_list = ", ".join(connector._quote_ident(c) for c in value_columns)

        if catalog is not None:
            table_fqtn = connector._qualify(catalog, schema_name, table_name)
        else:
            table_fqtn = connector._qualify(schema_name, table_name)

        def _sql_literal(value: Any) -> str:
            if value is None:
                return "NULL"
            if isinstance(value, (int, float)):
                return str(value)
            return "'" + str(value).replace("'", "''") + "'"

        if len(primary_key) == 1:
            values_sql = ", ".join(_sql_literal(k[0]) for k in keys)
            where_clause = f"{key_idents[0]} IN ({values_sql})"
        else:
            tuples_sql = ", ".join(
                "(" + ", ".join(_sql_literal(v) for v in k) + ")" for k in keys
            )
            where_clause = f"({key_list}) IN ({tuples_sql})"

        query = f"SELECT {key_list}, {value_list} FROM {table_fqtn} WHERE {where_clause}"
        return connector.execute_query(query).to_dict(orient="records")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        schema_results: List[SchemaValidationResult],
        missing_schemas: List[str],
        extra_schemas: List[str],
    ) -> ValidationSummary:
        summary = ValidationSummary()

        summary.total_schemas = len(schema_results) + len(missing_schemas)
        summary.failed_schemas = sum(
            1 for s in schema_results if s.status in (ValidationStatus.FAIL, ValidationStatus.ERROR)
        ) + len(missing_schemas)
        summary.passed_schemas = summary.total_schemas - summary.failed_schemas

        for schema_result in schema_results:
            summary.total_tables += len(schema_result.tables)
            summary.missing_tables += len(schema_result.missing_tables)
            summary.extra_tables += len(schema_result.extra_tables)

            for table in schema_result.tables:
                if table.status == ValidationStatus.PASS:
                    summary.passed_tables += 1
                elif table.status == ValidationStatus.ERROR:
                    summary.error_tables += 1
                    summary.failed_tables += 1
                else:
                    summary.failed_tables += 1

        return summary
