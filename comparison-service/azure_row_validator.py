"""
Azure Blob CSV -> single Databricks table validator.

Runs the same validation stages as CatalogValidator (comparison_engine.py)
- column names/order, data types, row counts, null/distinct/min-max
statistics, row-hash comparison, and row-level data mismatch detail - but
for one CSV file in Azure Blob Storage against one Databricks table,
instead of two whole catalogs.

Returns the exact same CatalogValidationResponse shape as CatalogValidator
(one SchemaValidationResult wrapping one TableValidationResult), so
report_generator.py's CSV/Excel output needs no changes at all - it just
sees "one schema, one table".

Row-hash comparison is the one stage that genuinely differs: the
Databricks side is hashed via pushed-down SQL exactly as before
(DatabricksConnector.get_row_hashes), but the CSV side has no SQL engine
behind it, so its rows are hashed in Python. To make the two hashes
directly comparable, every CSV cell is formatted to match Databricks'
`CAST(col AS STRING)` semantics for that column's *target* data type
(e.g. a whole-number `double` column always gets a trailing ".0", but a
`bigint` column never does) before being fed through the identical
sha2-equivalent digest (hashlib.sha256 producing the same hex output as
Databricks' sha2(x, 256) for the same input string - verified empirically).
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from azure_connector import AzureConnector
from databricks_connector import DatabricksConnector, values_differ
from models import (
    CatalogValidationResponse,
    ColumnValidationResult,
    CsvTableValidationRequest,
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

_NULL_SENTINEL = "\x01NULL\x01"


class AzureCsvValidator:
    """
    Validates one Azure Blob CSV file against one Databricks table.

    Column/type/nullable/row-count/statistics/row-hash comparison logic
    lives here (mirrors CatalogValidator); AzureConnector and
    DatabricksConnector only do I/O.
    """

    def __init__(
        self,
        azure_connector: AzureConnector,
        databricks_connector: DatabricksConnector,
    ) -> None:
        self.azure = azure_connector
        self.databricks = databricks_connector
        logger.debug("AzureCsvValidator initialised")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def validate(self, request: CsvTableValidationRequest) -> CatalogValidationResponse:
        start = time.perf_counter()
        run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logger.info(
            "Starting CSV validation | source=%s | target=%s.%s.%s",
            request.source_blob_path,
            request.target_catalog, request.target_schema, request.target_table,
        )

        try:
            source_df = self.azure.read_csv(request.source_blob_path)
        except Exception as exc:
            logger.exception("Failed to read source CSV '%s'", request.source_blob_path)
            return self._error_response(request, run_timestamp, start,
                                         f"Unable to read source CSV: {exc}")

        try:
            target_schema_df = self.databricks.get_table_schema(
                request.target_catalog, request.target_schema, request.target_table
            )
        except Exception as exc:
            logger.exception(
                "Failed to retrieve target table schema for '%s.%s.%s'",
                request.target_catalog, request.target_schema, request.target_table,
            )
            return self._error_response(request, run_timestamp, start,
                                         f"Unable to retrieve target table schema: {exc}")

        table_result = self._validate_table(request, source_df, target_schema_df)

        summary = ValidationSummary(
            total_schemas=1,
            passed_schemas=1 if table_result.status == ValidationStatus.PASS else 0,
            failed_schemas=0 if table_result.status == ValidationStatus.PASS else 1,
            total_tables=1,
            passed_tables=1 if table_result.status == ValidationStatus.PASS else 0,
            failed_tables=0 if table_result.status == ValidationStatus.PASS else 1,
            error_tables=1 if table_result.status == ValidationStatus.ERROR else 0,
        )

        schema_result = SchemaValidationResult(
            schema_name=request.target_schema,
            status=table_result.status,
            tables=[table_result],
        )

        execution_time = round(time.perf_counter() - start, 3)

        logger.info(
            "CSV validation finished | status=%s | duration=%.3fs",
            table_result.status, execution_time,
        )

        return CatalogValidationResponse(
            source_catalog=request.source_blob_path,
            target_catalog=f"{request.target_catalog}.{request.target_schema}.{request.target_table}",
            status=table_result.status,
            validation_timestamp=run_timestamp,
            execution_time_seconds=execution_time,
            summary=summary,
            schemas=[schema_result],
        )

    def _error_response(
        self,
        request: CsvTableValidationRequest,
        run_timestamp: str,
        start: float,
        error: str,
    ) -> CatalogValidationResponse:
        return CatalogValidationResponse(
            source_catalog=request.source_blob_path,
            target_catalog=f"{request.target_catalog}.{request.target_schema}.{request.target_table}",
            status=ValidationStatus.ERROR,
            validation_timestamp=run_timestamp,
            execution_time_seconds=round(time.perf_counter() - start, 3),
            error=error,
        )

    # ------------------------------------------------------------------
    # Per-table pipeline (mirrors CatalogValidator._validate_table)
    # ------------------------------------------------------------------
    def _validate_table(
        self,
        request: CsvTableValidationRequest,
        source_df: pd.DataFrame,
        target_schema_df: pd.DataFrame,
    ) -> TableValidationResult:

        result = TableValidationResult(
            schema_name=request.target_schema, table=request.target_table,
        )

        ignore = {c.lower() for c in (request.ignore_columns or [])}

        def norm(name: str) -> str:
            return name if request.case_sensitive_columns else name.lower()

        src_cols = {
            norm(str(c)): str(c) for c in source_df.columns
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
        }
        tgt_cols = {
            norm(str(c)): str(c) for c in target_schema_df["column_name"]
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
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
            result.error = "No common columns between source CSV and target table"
            return result

        # Column order (source CSV order vs target ordinal order)
        source_order = [c for c in source_df.columns if c in common_cols]
        target_order = [
            c for c in target_schema_df["column_name"].tolist() if c in common_cols
        ]
        result.source_column_order = source_order
        result.target_column_order = target_order

        if request.validate_column_order:
            order_matches = [c.lower() for c in source_order] == [
                c.lower() for c in target_order
            ]
            result.column_order_status = (
                ValidationStatus.PASS if order_matches else ValidationStatus.FAIL
            )
        else:
            result.column_order_status = ValidationStatus.SKIPPED

        primary_key = request.primary_key
        missing_keys = [k for k in primary_key if k.lower() not in {c.lower() for c in common_cols}]
        if missing_keys:
            result.status = ValidationStatus.ERROR
            result.error = f"Configured primary key column(s) not found as common columns: {missing_keys}"
            return result

        value_columns = sorted(
            c for c in common_cols if c.lower() not in {k.lower() for k in primary_key}
        )

        tgt_by_col = {
            str(r["column_name"]).lower(): r for _, r in target_schema_df.iterrows()
        }

        min_max_columns = [
            c for c in common_cols
            if self.databricks.is_min_max_eligible(
                str(tgt_by_col.get(c.lower(), {}).get("data_type", ""))
            )
        ]

        # Per-column: data type + null/distinct/min-max stats
        try:
            source_stats = self._csv_column_statistics(source_df, common_cols, min_max_columns)
            target_stats = self.databricks.get_column_statistics(
                request.target_catalog, request.target_schema, request.target_table,
                common_cols, min_max_columns,
            )
            stats_error = None
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s'", request.target_table
            )
            source_stats, target_stats = {}, {}
            stats_error = str(exc)

        column_results: List[ColumnValidationResult] = []
        dtype_statuses, null_statuses, distinct_statuses, minmax_statuses = [], [], [], []

        for col in common_cols:
            tgt_row = tgt_by_col.get(col.lower(), {})
            col_result = ColumnValidationResult(column=col, status=ValidationStatus.PASS)

            # Data type: compare the CSV column's inferred pandas dtype
            # (mapped to a Databricks-equivalent name) against the target's
            # real Databricks type. Informational only - not pushed down,
            # since the CSV has no declared schema of its own.
            src_type = self._infer_databricks_type(source_df[col])
            tgt_type = str(tgt_row.get("data_type"))
            col_result.source_data_type = src_type
            col_result.target_data_type = tgt_type
            col_result.data_type_status = (
                ValidationStatus.PASS if self._types_compatible(src_type, tgt_type)
                else ValidationStatus.FAIL
            )
            dtype_statuses.append(col_result.data_type_status)
            col_result.nullable_status = ValidationStatus.SKIPPED

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
                        ValidationStatus.PASS
                        if (col_result.source_min == col_result.target_min
                            and col_result.source_max == col_result.target_max)
                        else ValidationStatus.FAIL
                    )
                else:
                    col_result.min_max_status = ValidationStatus.SKIPPED

            null_statuses.append(col_result.null_count_status)
            distinct_statuses.append(col_result.distinct_count_status)
            minmax_statuses.append(col_result.min_max_status)

            col_result.status = CatalogValidatorLikeStatus.calculate_overall_status(
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
        result.data_types_status = CatalogValidatorLikeStatus.calculate_overall_status(dtype_statuses)
        result.nullable_status = ValidationStatus.SKIPPED
        result.null_counts_status = CatalogValidatorLikeStatus.calculate_overall_status(null_statuses)
        result.distinct_counts_status = CatalogValidatorLikeStatus.calculate_overall_status(distinct_statuses)
        result.min_max_status = CatalogValidatorLikeStatus.calculate_overall_status(minmax_statuses)

        # Row counts
        try:
            src_count = len(source_df)
            tgt_count = self.databricks.get_row_count(
                request.target_catalog, request.target_schema, request.target_table
            )
            result.row_count_source = src_count
            result.row_count_target = tgt_count
            result.row_count_difference = tgt_count - src_count
            result.row_count_status = (
                ValidationStatus.PASS if src_count == tgt_count else ValidationStatus.FAIL
            )
        except Exception as exc:
            logger.exception("Failed to compute row counts for '%s'", request.target_table)
            result.row_count_status = ValidationStatus.ERROR
            result.error = f"Row count failed: {exc}"

        # Row-hash comparison + data mismatches
        result.data = self._compare_data(request, source_df, target_schema_df, common_cols, value_columns)

        result.status = CatalogValidatorLikeStatus.calculate_overall_status(
            [
                result.columns_status,
                result.column_order_status,
                result.row_count_status,
                result.data_types_status,
                result.null_counts_status,
                result.distinct_counts_status,
                result.min_max_status,
                result.data.status if result.data else ValidationStatus.SKIPPED,
            ]
        )

        return result

    # ------------------------------------------------------------------
    # CSV-side column statistics (mirrors DatabricksConnector.get_column_statistics)
    # ------------------------------------------------------------------
    @staticmethod
    def _csv_column_statistics(
        df: pd.DataFrame,
        columns: Sequence[str],
        min_max_columns: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        min_max_set = {c.lower() for c in min_max_columns}
        result: Dict[str, Dict[str, Any]] = {}

        for col in columns:
            series = df[col]
            entry: Dict[str, Any] = {
                "null_count": int(series.isna().sum()),
                "distinct_count": int(series.nunique(dropna=True)),
                "min": None,
                "max": None,
            }
            if col.lower() in min_max_set:
                non_null = series.dropna()
                if not non_null.empty:
                    entry["min"] = non_null.min()
                    entry["max"] = non_null.max()
            result[col] = entry

        return result

    @staticmethod
    def _infer_databricks_type(series: pd.Series) -> str:
        dtype = str(series.dtype)
        if dtype.startswith("int"):
            return "bigint"
        if dtype.startswith("float"):
            return "double"
        if dtype == "bool":
            return "boolean"
        if dtype.startswith("datetime"):
            return "timestamp"
        return "string"

    @staticmethod
    def _types_compatible(source_type: str, target_type: str) -> bool:
        if source_type == target_type:
            return True
        numeric = {"bigint", "int", "smallint", "tinyint", "double", "float", "decimal"}
        target_base = target_type.split("(")[0].lower()
        if source_type in numeric and target_base in numeric:
            return True
        if source_type == "string" and target_base in ("string", "date", "timestamp", "varchar", "char"):
            # CSV columns are read as plain strings/objects for date-like
            # target columns (no schema to infer from) - don't fail solely
            # on that; the actual values still get compared via row-hash.
            return True
        return False

    # ------------------------------------------------------------------
    # Row-hash comparison + row-level data mismatch detail
    # ------------------------------------------------------------------
    def _compare_data(
        self,
        request: CsvTableValidationRequest,
        source_df: pd.DataFrame,
        target_schema_df: pd.DataFrame,
        common_columns: List[str],
        value_columns: List[str],
    ) -> DataValidationResult:

        mode = request.data_compare_mode
        primary_key = request.primary_key
        using_row_number_fallback = not primary_key

        tgt_by_col = {
            str(r["column_name"]).lower(): str(r["data_type"])
            for _, r in target_schema_df.iterrows()
        }

        logger.info(
            "[csv-row-hash] table=%s.%s.%s | key_columns=%s | value_columns=%s",
            request.target_catalog, request.target_schema, request.target_table,
            primary_key or "<none - row-number fallback>", value_columns,
        )

        if using_row_number_fallback:
            hash_value_columns = sorted(value_columns)
            try:
                source_hashes = self._hash_csv_rows_by_row_number(
                    source_df, hash_value_columns, tgt_by_col,
                )
            except Exception as exc:
                logger.exception("Failed to hash source CSV rows by row number")
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR,
                    key_columns=["row_number"], error=f"CSV row-number hashing failed: {exc}",
                )

            try:
                target_hashes = self.databricks.get_row_hashes_by_row_number(
                    request.target_catalog, request.target_schema, request.target_table,
                    hash_value_columns,
                )
            except Exception as exc:
                logger.exception("Failed to fetch target row-number hashes")
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR,
                    key_columns=["row_number"], error=f"Target row-number hashing failed: {exc}",
                )

            effective_key = ["row_number"]
        else:
            try:
                source_hashes = self._hash_csv_rows(source_df, value_columns, primary_key, tgt_by_col)
            except Exception as exc:
                logger.exception("Failed to hash source CSV rows")
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR,
                    key_columns=primary_key, error=f"CSV row hashing failed: {exc}",
                )

            try:
                target_hashes = self.databricks.get_row_hashes(
                    request.target_catalog, request.target_schema, request.target_table,
                    value_columns, primary_key,
                )
            except Exception as exc:
                logger.exception("Failed to fetch target row hashes")
                return DataValidationResult(
                    mode=mode, status=ValidationStatus.ERROR,
                    key_columns=primary_key, error=f"Target row hashing failed: {exc}",
                )

            effective_key = primary_key

        logger.info(
            "[csv-row-hash] fetched | source_rows=%d | target_rows=%d | row_number_fallback=%s",
            len(source_hashes), len(target_hashes), using_row_number_fallback,
        )

        mismatches, mismatch_count, mismatch_pct = self._compare_row_hashes(
            source_hashes, target_hashes, effective_key,
        )

        logger.info(
            "[csv-row-hash] mismatch_count=%d | mismatch_pct=%.2f%%",
            mismatch_count, mismatch_pct,
        )

        data_result = DataValidationResult(
            mode=mode,
            status=ValidationStatus.FAIL if mismatch_count > 0 else ValidationStatus.PASS,
            key_columns=effective_key,
            row_hash_mismatches=mismatches,
            row_hash_mismatch_count=mismatch_count,
            row_hash_mismatch_percentage=mismatch_pct,
            note=(
                "No primary key configured - row-level comparison used a synthetic "
                "row-number match (CSV file order vs. Databricks ROW_NUMBER(), both "
                "sorted by every common column) instead of a real key. Only reliable "
                "when both sides contain the same set of rows; cannot pinpoint which "
                "specific record changed the way a real key can."
                if using_row_number_fallback else None
            ),
        )

        if mode == DataCompareMode.FULL and mismatch_count > 0 and not using_row_number_fallback:
            data_result.sample_changed_detail = self._changed_row_detail(
                request, source_df, source_hashes, target_hashes, primary_key, value_columns,
            )

        return data_result

    @staticmethod
    def _format_value_for_hash(value: Any, databricks_type: str) -> str:
        """
        Format one cell to match Databricks' CAST(col AS STRING) output for
        the given target column type, so a Python-computed hash lines up
        with the SQL-computed hash for the same logical value.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return _NULL_SENTINEL

        base_type = databricks_type.split("(")[0].lower()

        if base_type in ("double", "float", "decimal"):
            f = float(value)
            # Databricks always prints a decimal point for floating types,
            # even for whole numbers (158068.0, never 158068).
            if f == int(f) and "e" not in repr(f).lower():
                return f"{f:.1f}" if base_type != "decimal" else str(Decimal(str(value)))
            return repr(f)

        if base_type in ("bigint", "int", "smallint", "tinyint"):
            return str(int(value))

        if base_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower()
            return "true" if bool(value) else "false"

        if base_type == "date":
            if isinstance(value, (datetime.date, datetime.datetime)):
                d = value.date() if isinstance(value, datetime.datetime) else value
                return d.isoformat()
            return str(value).strip()

        if base_type == "timestamp":
            if isinstance(value, (datetime.date, datetime.datetime)):
                return str(value)
            return str(value).strip()

        return str(value)

    def _hash_csv_rows(
        self,
        df: pd.DataFrame,
        value_columns: Sequence[str],
        primary_key: Sequence[str],
        target_types_by_col: Dict[str, str],
    ) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            parts = [
                self._format_value_for_hash(
                    row.get(col), target_types_by_col.get(col.lower(), "string")
                )
                for col in value_columns
            ]
            digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
            entry = {k: row.get(k) for k in primary_key}
            entry["row_hash"] = digest
            rows.append(entry)

        if not rows:
            return pd.DataFrame(columns=list(primary_key) + ["row_hash"])

        return pd.DataFrame(rows)

    def _hash_csv_rows_by_row_number(
        self,
        df: pd.DataFrame,
        value_columns: Sequence[str],
        target_types_by_col: Dict[str, str],
    ) -> pd.DataFrame:
        """
        Fallback for when no primary key is configured: sorts by every
        value column (matching the ORDER BY used by
        DatabricksConnector.get_row_hashes_by_row_number so both sides
        assign the same row a matching number regardless of storage/file
        order), assigns a 1-based row_number, then hashes each row the
        same way _hash_csv_rows does. See that method's caveat: this only
        gives meaningful results when both sides contain the same set of
        rows.
        """
        sorted_df = df.sort_values(by=list(value_columns), kind="stable").reset_index(drop=True)

        rows = []
        for i, row in sorted_df.iterrows():
            parts = [
                self._format_value_for_hash(
                    row.get(col), target_types_by_col.get(col.lower(), "string")
                )
                for col in value_columns
            ]
            digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
            rows.append({"row_number": i + 1, "row_hash": digest})

        if not rows:
            return pd.DataFrame(columns=["row_number", "row_hash"])

        return pd.DataFrame(rows)

    @staticmethod
    def _compare_row_hashes(
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
        primary_key_cols: Sequence[str],
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
        request: CsvTableValidationRequest,
        source_df: pd.DataFrame,
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
        primary_key: Sequence[str],
        value_columns: Sequence[str],
    ) -> List[RowMismatchDetail]:
        """
        For MISMATCH keys (present, differing hash, on both sides), fetch
        the actual target row values for those keys in one batched query
        and diff column-by-column against the source CSV row - same
        approach as DatabricksConnector._changed_row_detail, but the
        source side comes from the already-loaded CSV DataFrame instead
        of a second SQL query.
        """
        limit_samples = request.max_sample_rows

        src_hash_by_key = {
            tuple(r[k] for k in primary_key): r["row_hash"]
            for _, r in source_hashes.iterrows()
        }
        tgt_hash_by_key = {
            tuple(r[k] for k in primary_key): r["row_hash"]
            for _, r in target_hashes.iterrows()
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

        mismatch_key_set = set(mismatch_keys)
        source_by_key: Dict[tuple, pd.Series] = {}
        for _, row in source_df.iterrows():
            key_tuple = tuple(row[k] for k in primary_key)
            if key_tuple in mismatch_key_set:
                source_by_key[key_tuple] = row
                if len(source_by_key) >= len(mismatch_key_set):
                    break

        target_rows = self._fetch_target_rows_for_keys(
            request, primary_key, value_columns, mismatch_keys,
        )
        target_by_key = {
            tuple(r[k] for k in primary_key): r for r in target_rows
        }

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
                        schema_name=request.target_schema,
                        table=request.target_table,
                        primary_key=key_dict,
                        mismatch_column=col,
                        source_value=src_row.get(col),
                        target_value=tgt_row.get(col),
                        source_row_hash=src_hash_by_key.get(key_tuple),
                        target_row_hash=tgt_hash_by_key.get(key_tuple),
                    )
                )

        return detail

    def _fetch_target_rows_for_keys(
        self,
        request: CsvTableValidationRequest,
        primary_key: Sequence[str],
        value_columns: Sequence[str],
        keys: Sequence[tuple],
    ) -> List[Dict[str, Any]]:
        """Batched fetch of target rows for a bounded set of primary keys."""
        if not keys:
            return []

        key_idents = [self.databricks._quote_ident(k) for k in primary_key]
        key_list = ", ".join(key_idents)
        value_list = ", ".join(self.databricks._quote_ident(c) for c in value_columns)
        table_fqtn = self.databricks._qualify(
            request.target_catalog, request.target_schema, request.target_table
        )

        if len(primary_key) == 1:
            values_sql = ", ".join(self._sql_literal(k[0]) for k in keys)
            where_clause = f"{key_idents[0]} IN ({values_sql})"
        else:
            tuples_sql = ", ".join(
                "(" + ", ".join(self._sql_literal(v) for v in k) + ")" for k in keys
            )
            where_clause = f"({key_list}) IN ({tuples_sql})"

        query = f"SELECT {key_list}, {value_list} FROM {table_fqtn} WHERE {where_clause}"
        return self.databricks.execute_query(query).to_dict(orient="records")

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"


class CatalogValidatorLikeStatus:
    """Reuses CatalogValidator's status-aggregation rule without importing
    the whole class (avoids a circular import with comparison_engine)."""

    @staticmethod
    def calculate_overall_status(
        statuses: List[Optional[ValidationStatus]],
    ) -> ValidationStatus:
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
