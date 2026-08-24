"""
Comparison Engine

Contains all data-comparison logic for the Data Migration Comparison Service.

Receives already-authenticated connector instances and never performs
authentication itself.

This module contains two engines:

1. ComparisonEngine (UNCHANGED) - row-level comparison of a source CSV
   (Azure Storage) against a target table (Databricks Delta Lake).

2. CatalogValidator (NEW) - recursive Databricks catalog-to-catalog
   validation: catalog -> schemas -> tables -> columns -> data, per the
   15-stage validation sequence. All comparisons are pushed down to
   Databricks SQL via DatabricksConnector; this class never loads a full
   table into pandas.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from azure_connector import AzureConnector
from databricks_connector import DatabricksConnector
from models import (
    CatalogValidationRequest,
    CatalogValidationResponse,
    ColumnValidationResult,
    ComparisonRequest,
    ComparisonResponse,
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


class ComparisonEngine:
    """
    Orchestrates side-by-side comparison of a source CSV
    (Azure Storage) against a target table (Databricks Delta Lake).
    """

    def __init__(
        self,
        azure_connector: AzureConnector,
        databricks_connector: DatabricksConnector,
    ) -> None:

        self.azure = azure_connector
        self.databricks = databricks_connector

        logger.debug("ComparisonEngine initialised")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def compare(self, request: ComparisonRequest) -> ComparisonResponse:

        start = time.perf_counter()

        logger.info(
            "Starting comparison | source_csv=%s | target=%s | keys=%s",
            request.source_table,
            request.target_table,
            request.primary_keys,
        )

        try:

            source_df = self._load_source(request)
            target_df = self._load_target(request)

            logger.info(
                "Data loaded successfully | source_shape=%s | target_shape=%s",
                source_df.shape,
                target_df.shape,
            )

            source_df, target_df = self._preprocess(
                source_df,
                target_df,
                request,
            )

            row_count_source = len(source_df)
            row_count_target = len(target_df)

            schema_result = self._compare_schema(request)

            duplicate_result = self._detect_duplicates(
                source_df,
                target_df,
                request,
            )

            missing_extra = self._find_missing_and_extra(
                source_df,
                target_df,
                request,
            )

            null_result = self._compare_nulls(
                source_df,
                target_df,
                request,
            )

            dtype_result = self._compare_dtypes(
                source_df,
                target_df,
                request,
            )

            value_result = self._compare_values(
                source_df,
                target_df,
                request,
            )

            status = self._determine_status(
                schema_result=schema_result,
                duplicate_result=duplicate_result,
                missing_extra=missing_extra,
                null_result=null_result,
                dtype_result=dtype_result,
                value_result=value_result,
            )

            execution_time = round(
                time.perf_counter() - start,
                3,
            )

            response = ComparisonResponse(
                status=status,
                execution_time_seconds=execution_time,
                row_count_source=row_count_source,
                row_count_target=row_count_target,
                schema_match=schema_result["match"],
                matched_rows=missing_extra["matched_rows"],
                missing_rows=missing_extra["missing_rows"],
                extra_rows=missing_extra["extra_rows"],
                duplicate_rows=duplicate_result,
                column_differences=(
                    schema_result["differences"]
                    + dtype_result
                    + null_result
                ),
                sample_mismatches=value_result["sample_mismatches"],
            )

            logger.info(
                "Comparison finished | status=%s | duration=%.3fs",
                status,
                execution_time,
            )

            return response

        except Exception as exc:

            logger.exception(
                "Comparison pipeline failed: %s",
                str(exc),
            )

            raise

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_source(
        self,
        request: ComparisonRequest,
    ) -> pd.DataFrame:

        blob_path = request.source_table

        if not blob_path:
            raise ValueError(
                "source_table must contain a valid Azure Storage CSV path"
            )

        logger.debug(
            "Loading source CSV from Azure Storage: %s",
            blob_path,
        )

        return self.azure.read_csv(blob_path)

    def _load_target(
        self,
        request: ComparisonRequest,
    ) -> pd.DataFrame:

        logger.debug(
            "Loading target table: %s",
            request.target_table,
        )

        if request.target_query:
            return self.databricks.read_query(
                request.target_query
            )

        return self.databricks.read_table(
            request.target_table
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        ignore: Set[str] = set(
            request.ignore_columns or []
        )

        def drop_ignored(df: pd.DataFrame) -> pd.DataFrame:

            cols_to_drop = [
                c
                for c in df.columns
                if c.lower() in {i.lower() for i in ignore}
            ]

            if cols_to_drop:
                return df.drop(columns=cols_to_drop)

            return df

        source = drop_ignored(source.copy())
        target = drop_ignored(target.copy())

        if request.trim_strings:

            for df in (source, target):

                str_cols = df.select_dtypes(
                    include=["object", "string"]
                ).columns

                for col in str_cols:

                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.strip()
                        .replace({"nan": np.nan})
                    )

        if not request.case_sensitive:

            for df in (source, target):

                str_cols = df.select_dtypes(
                    include=["object", "string"]
                ).columns

                for col in str_cols:

                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.lower()
                        .replace({"nan": np.nan})
                    )

        return source, target

    # ------------------------------------------------------------------
    # Schema comparison
    # ------------------------------------------------------------------
    def _compare_schema(
        self,
        request: ComparisonRequest,
    ) -> Dict[str, Any]:

        try:

            src_schema = self.azure.get_schema(
                request.source_table
            )

            tgt_schema = self.databricks.get_schema(
                request.target_table
            )

        except Exception as exc:

            return {
                "match": False,
                "differences": [
                    {
                        "type": "schema_error",
                        "detail": str(exc),
                    }
                ],
            }

        src_cols = {
            str(col).lower()
            for col in src_schema["column_name"]
        }

        tgt_cols = {
            str(col).lower()
            for col in tgt_schema["column_name"]
        }

        ignore = {
            c.lower()
            for c in (request.ignore_columns or [])
        }

        src_cols -= ignore
        tgt_cols -= ignore

        differences = []

        missing_in_target = sorted(
            src_cols - tgt_cols
        )

        extra_in_target = sorted(
            tgt_cols - src_cols
        )

        if missing_in_target:

            differences.append(
                {
                    "type": "missing_columns_in_target",
                    "columns": missing_in_target,
                }
            )

        if extra_in_target:

            differences.append(
                {
                    "type": "extra_columns_in_target",
                    "columns": extra_in_target,
                }
            )

        return {
            "match": len(differences) == 0,
            "differences": differences,
        }

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    def _detect_duplicates(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> Dict[str, Any]:

        keys = request.primary_keys

        if not keys:
            return {
                "source_duplicates": 0,
                "target_duplicates": 0,
            }

        def _dup_count(
            df: pd.DataFrame,
        ) -> Tuple[int, List[Any]]:

            if not all(k in df.columns for k in keys):
                return 0, []

            dup_mask = df.duplicated(
                subset=keys,
                keep=False,
            )

            dup_rows = (
                df.loc[dup_mask, keys]
                .drop_duplicates()
            )

            return (
                int(dup_mask.sum()),
                dup_rows.head(20).to_dict(
                    orient="records"
                ),
            )

        src_count, src_samples = _dup_count(source)
        tgt_count, tgt_samples = _dup_count(target)

        return {
            "source_duplicates": src_count,
            "target_duplicates": tgt_count,
            "source_sample": src_samples,
            "target_sample": tgt_samples,
        }

    # ------------------------------------------------------------------
    # Missing / Extra rows
    # ------------------------------------------------------------------
    def _find_missing_and_extra(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> Dict[str, Any]:

        keys = request.primary_keys

        if not keys:

            src_keys = source.apply(
                lambda r: hash(tuple(r)),
                axis=1,
            )

            tgt_keys = target.apply(
                lambda r: hash(tuple(r)),
                axis=1,
            )

            missing = int(
                (~src_keys.isin(tgt_keys)).sum()
            )

            extra = int(
                (~tgt_keys.isin(src_keys)).sum()
            )

            return {
                "matched_rows": len(source)
                - missing,
                "missing_rows": missing,
                "extra_rows": extra,
                "missing_sample": [],
                "extra_sample": [],
            }

        src_idx = source.set_index(keys)
        tgt_idx = target.set_index(keys)

        missing_keys = src_idx.index.difference(
            tgt_idx.index
        )

        extra_keys = tgt_idx.index.difference(
            src_idx.index
        )

        matched = len(
            src_idx.index.intersection(
                tgt_idx.index
            )
        )

        return {
            "matched_rows": int(matched),
            "missing_rows": int(len(missing_keys)),
            "extra_rows": int(len(extra_keys)),
            "missing_sample": [],
            "extra_sample": [],
        }

    # ------------------------------------------------------------------
    # Null comparison
    # ------------------------------------------------------------------
    def _compare_nulls(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> List[Dict[str, Any]]:

        differences = []

        common_cols = sorted(
            set(source.columns)
            & set(target.columns)
        )

        for col in common_cols:

            src_nulls = int(
                source[col].isna().sum()
            )

            tgt_nulls = int(
                target[col].isna().sum()
            )

            if src_nulls != tgt_nulls:

                differences.append(
                    {
                        "type": "null_count_mismatch",
                        "column": col,
                        "source_nulls": src_nulls,
                        "target_nulls": tgt_nulls,
                    }
                )

        return differences

    # ------------------------------------------------------------------
    # Dtype comparison
    # ------------------------------------------------------------------
    def _compare_dtypes(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> List[Dict[str, Any]]:

        differences = []

        common_cols = sorted(
            set(source.columns)
            & set(target.columns)
        )

        for col in common_cols:

            src_dtype = str(source[col].dtype)
            tgt_dtype = str(target[col].dtype)

            if self._dtypes_compatible(
                src_dtype,
                tgt_dtype,
            ):
                continue

            differences.append(
                {
                    "type": "dtype_mismatch",
                    "column": col,
                    "source_dtype": src_dtype,
                    "target_dtype": tgt_dtype,
                }
            )

        return differences

    @staticmethod
    def _dtypes_compatible(
        a: str,
        b: str,
    ) -> bool:

        if a == b:
            return True

        numeric = {
            "int64",
            "int32",
            "float64",
            "float32",
            "Int64",
            "Float64",
        }

        stringy = {
            "object",
            "string",
            "str",
        }

        if a in numeric and b in numeric:
            return True

        if a in stringy and b in stringy:
            return True

        return False

    # ------------------------------------------------------------------
    # Value comparison
    # ------------------------------------------------------------------
    def _compare_values(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        request: ComparisonRequest,
    ) -> Dict[str, Any]:

        return {
            "mismatch_count": 0,
            "sample_mismatches": [],
        }

    # ------------------------------------------------------------------
    # Status aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_status(
        schema_result: Dict[str, Any],
        duplicate_result: Dict[str, Any],
        missing_extra: Dict[str, Any],
        null_result: List[Dict[str, Any]],
        dtype_result: List[Dict[str, Any]],
        value_result: Dict[str, Any],
    ) -> str:

        if not schema_result["match"]:
            return "FAIL"

        if (
            missing_extra["missing_rows"] > 0
            or missing_extra["extra_rows"] > 0
        ):
            return "FAIL"

        if (
            duplicate_result.get("source_duplicates", 0) > 0
            or duplicate_result.get("target_duplicates", 0) > 0
        ):
            return "FAIL"

        if value_result.get("mismatch_count", 0) > 0:
            return "FAIL"

        if null_result or dtype_result:
            return "WARN"

        return "PASS"


# ----------------------------------------------------------------------
# Helpers (existing - unchanged)
# ----------------------------------------------------------------------
def _safe_serialize(value: Any) -> Any:

    if pd.isna(value):
        return None

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ========================================================================
# NEW: Databricks catalog-to-catalog validator
# ========================================================================
class CatalogValidator:
    """
    Recursive Databricks catalog-to-catalog validator.

    Responsible for the comparison/decision logic only (PASS/FAIL/ERROR/
    SKIPPED). All data retrieval is delegated to DatabricksConnector -
    this class never talks to Databricks directly and never loads a full
    table's rows into memory.

    Usage:
        validator = CatalogValidator(databricks_connector)
        result = validator.compare_catalogs(request)
    """

    def __init__(self, databricks_connector: DatabricksConnector) -> None:
        self.databricks = databricks_connector
        logger.debug("CatalogValidator initialised")

    # ------------------------------------------------------------------
    # Stage 1 + top-level orchestration
    # ------------------------------------------------------------------
    def compare_catalogs(
        self,
        request: CatalogValidationRequest,
    ) -> CatalogValidationResponse:

        start = time.perf_counter()
        run_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Starting catalog validation | source=%s | target=%s",
            request.source_catalog,
            request.target_catalog,
        )

        # Stage 1: catalog exists
        try:
            source_exists = self.databricks.catalog_exists(request.source_catalog)
            target_exists = self.databricks.catalog_exists(request.target_catalog)
        except Exception as exc:
            logger.exception("Failed to verify catalog existence")
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.ERROR,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Unable to verify catalog existence: {exc}",
            )

        if not source_exists or not target_exists:
            missing = []
            if not source_exists:
                missing.append(request.source_catalog)
            if not target_exists:
                missing.append(request.target_catalog)
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.FAIL,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Catalog(s) do not exist: {', '.join(missing)}",
            )

        # Stage 2/3/4: schemas
        try:
            common_schemas, missing_schemas, extra_schemas = self.compare_schemas(
                request.source_catalog, request.target_catalog
            )
        except Exception as exc:
            logger.exception("Failed to compare schemas")
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.ERROR,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Unable to compare schemas: {exc}",
            )

        if request.schemas:
            wanted = {s.lower() for s in request.schemas}
            common_schemas = [s for s in common_schemas if s.lower() in wanted]

        schema_results: List[SchemaValidationResult] = []

        for schema_name in common_schemas:
            schema_results.append(
                self._validate_schema(request, schema_name)
            )

        summary = self._build_summary(schema_results, missing_schemas, extra_schemas)
        overall_status = self.calculate_overall_status(
            [s.status for s in schema_results]
            + ([ValidationStatus.FAIL] if missing_schemas else [])
        )

        execution_time = round(time.perf_counter() - start, 3)

        logger.info(
            "Catalog validation finished | status=%s | duration=%.3fs",
            overall_status,
            execution_time,
        )

        return CatalogValidationResponse(
            source_catalog=request.source_catalog,
            target_catalog=request.target_catalog,
            status=overall_status,
            validation_timestamp=run_timestamp,
            execution_time_seconds=execution_time,
            missing_schemas=missing_schemas,
            extra_schemas=extra_schemas,
            summary=summary,
            schemas=schema_results,
        )

    # Backward/spec-friendly alias
    def validate(self, request: CatalogValidationRequest) -> CatalogValidationResponse:
        return self.compare_catalogs(request)

    # Databricks-managed system schema, present in every catalog - never a
    # real migration target, so it is excluded from validation entirely
    # (not just skipped: never counted as common/missing/extra either).
    _EXCLUDED_SCHEMAS = {"information_schema"}

    # ------------------------------------------------------------------
    # Stage 2/3/4: schema comparison
    # ------------------------------------------------------------------
    def compare_schemas(
        self,
        source_catalog: str,
        target_catalog: str,
    ) -> Tuple[List[str], List[str], List[str]]:

        source_schemas = set(self.databricks.get_schemas(source_catalog))
        target_schemas = set(self.databricks.get_schemas(target_catalog))

        source_schemas -= {
            s for s in source_schemas if s.lower() in self._EXCLUDED_SCHEMAS
        }
        target_schemas -= {
            s for s in target_schemas if s.lower() in self._EXCLUDED_SCHEMAS
        }

        common = sorted(source_schemas & target_schemas)
        missing = sorted(source_schemas - target_schemas)   # in source, not target
        extra = sorted(target_schemas - source_schemas)     # in target, not source

        return common, missing, extra

    def _validate_schema(
        self,
        request: CatalogValidationRequest,
        schema_name: str,
    ) -> SchemaValidationResult:

        try:
            common_tables, missing_tables, extra_tables = self.compare_tables(
                request.source_catalog, request.target_catalog, schema_name
            )
        except Exception as exc:
            logger.exception("Failed to compare tables for schema '%s'", schema_name)
            return SchemaValidationResult(
                schema_name=schema_name,
                status=ValidationStatus.ERROR,
                error=f"Unable to compare tables: {exc}",
            )

        if request.tables:
            wanted = {t.lower() for t in request.tables}
            common_tables = [t for t in common_tables if t.lower() in wanted]

        table_results: List[TableValidationResult] = []

        for table_name in common_tables:
            table_results.append(
                self._validate_table(request, schema_name, table_name)
            )

        statuses = [t.status for t in table_results]
        if missing_tables:
            statuses.append(ValidationStatus.FAIL)

        status = self.calculate_overall_status(statuses)

        return SchemaValidationResult(
            schema_name=schema_name,
            status=status,
            missing_tables=missing_tables,
            extra_tables=extra_tables,
            tables=table_results,
        )

    # ------------------------------------------------------------------
    # Stage 3/4: table comparison
    # ------------------------------------------------------------------
    def compare_tables(
        self,
        source_catalog: str,
        target_catalog: str,
        schema_name: str,
    ) -> Tuple[List[str], List[str], List[str]]:

        source_tables = set(self.databricks.get_tables(source_catalog, schema_name))
        target_tables = set(self.databricks.get_tables(target_catalog, schema_name))

        common = sorted(source_tables & target_tables)
        missing = sorted(source_tables - target_tables)
        extra = sorted(target_tables - source_tables)

        return common, missing, extra

    # ------------------------------------------------------------------
    # Per-table pipeline: stages 5-15
    # ------------------------------------------------------------------
    def _validate_table(
        self,
        request: CatalogValidationRequest,
        schema_name: str,
        table_name: str,
    ) -> TableValidationResult:

        result = TableValidationResult(schema_name=schema_name, table=table_name)

        try:
            source_schema_df = self.databricks.get_table_schema(
                request.source_catalog, schema_name, table_name
            )
            target_schema_df = self.databricks.get_table_schema(
                request.target_catalog, schema_name, table_name
            )
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s'", schema_name, table_name
            )
            result.status = ValidationStatus.ERROR
            result.error = f"Unable to retrieve column metadata: {exc}"
            return result

        ignore = {c.lower() for c in (request.ignore_columns or [])}

        # Stage 5/6: column names
        missing_cols, extra_cols, common_cols = self.compare_columns(
            source_schema_df, target_schema_df, request.case_sensitive_columns, ignore
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

        # Stage 9: column order (on common columns only)
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
            order_matches = [c.lower() for c in source_order] == [
                c.lower() for c in target_order
            ]
            result.column_order_status = (
                ValidationStatus.PASS if order_matches else ValidationStatus.FAIL
            )
        else:
            result.column_order_status = ValidationStatus.SKIPPED

        # Per-column: data type + nullable (stages 7, 8)
        src_by_col = {
            str(r["column_name"]).lower(): r for _, r in source_schema_df.iterrows()
        }
        tgt_by_col = {
            str(r["column_name"]).lower(): r for _, r in target_schema_df.iterrows()
        }

        min_max_columns = [
            c for c in common_cols
            if self.databricks.is_min_max_eligible(
                str(src_by_col.get(c.lower(), {}).get("data_type", ""))
            )
        ]

        # Stage 11-13: null/distinct/min-max, batched per table
        try:
            source_stats = self.databricks.get_column_statistics(
                request.source_catalog, schema_name, table_name,
                common_cols, min_max_columns,
            )
            target_stats = self.databricks.get_column_statistics(
                request.target_catalog, schema_name, table_name,
                common_cols, min_max_columns,
            )
            stats_error = None
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s.%s'", schema_name, table_name
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

            # Stage 7: data type
            src_type = str(src_row.get("data_type"))
            tgt_type = str(tgt_row.get("data_type"))
            col_result.source_data_type = src_type
            col_result.target_data_type = tgt_type
            col_result.data_type_status = (
                ValidationStatus.PASS if src_type == tgt_type else ValidationStatus.FAIL
            )
            dtype_statuses.append(col_result.data_type_status)

            # Stage 8: nullable
            if request.validate_nullable:
                src_null = bool(src_row.get("is_nullable"))
                tgt_null = bool(tgt_row.get("is_nullable"))
                col_result.source_nullable = src_null
                col_result.target_nullable = tgt_null
                col_result.nullable_status = (
                    ValidationStatus.PASS if src_null == tgt_null else ValidationStatus.FAIL
                )
            else:
                col_result.nullable_status = ValidationStatus.SKIPPED
            nullable_statuses.append(col_result.nullable_status)

            if stats_error:
                col_result.null_count_status = ValidationStatus.ERROR
                col_result.distinct_count_status = ValidationStatus.ERROR
                col_result.error = stats_error
            else:
                s_stat = source_stats.get(col, {})
                t_stat = target_stats.get(col, {})

                # Stage 11: null counts
                col_result.source_null_count = s_stat.get("null_count")
                col_result.target_null_count = t_stat.get("null_count")
                col_result.null_count_status = (
                    ValidationStatus.PASS
                    if col_result.source_null_count == col_result.target_null_count
                    else ValidationStatus.FAIL
                )

                # Stage 12: distinct counts
                col_result.source_distinct_count = s_stat.get("distinct_count")
                col_result.target_distinct_count = t_stat.get("distinct_count")
                col_result.distinct_count_status = (
                    ValidationStatus.PASS
                    if col_result.source_distinct_count == col_result.target_distinct_count
                    else ValidationStatus.FAIL
                )

                # Stage 13: min/max
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

            col_result.status = self.calculate_overall_status(
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
        result.data_types_status = self.calculate_overall_status(dtype_statuses)
        result.nullable_status = self.calculate_overall_status(nullable_statuses)
        result.null_counts_status = self.calculate_overall_status(null_statuses)
        result.distinct_counts_status = self.calculate_overall_status(distinct_statuses)
        result.min_max_status = self.calculate_overall_status(minmax_statuses)

        # Stage 10: row counts
        try:
            src_count = self.databricks.get_row_count(
                request.source_catalog, schema_name, table_name
            )
            tgt_count = self.databricks.get_row_count(
                request.target_catalog, schema_name, table_name
            )
            result.row_count_source = src_count
            result.row_count_target = tgt_count
            result.row_count_difference = tgt_count - src_count
            result.row_count_status = (
                ValidationStatus.PASS if src_count == tgt_count else ValidationStatus.FAIL
            )
        except Exception as exc:
            logger.exception(
                "Failed to compute row counts for '%s.%s'", schema_name, table_name
            )
            result.row_count_status = ValidationStatus.ERROR
            result.error = f"Row count failed: {exc}"

        # Stage 15: actual data comparison
        result.data = self.compare_data(request, schema_name, table_name, common_cols)

        # Stage 15b: row-hash comparison - primary mechanism for row-level
        # mismatch detection whenever a key is configured, independent of
        # data_compare_mode (runs even when compare_data above was SKIPPED
        # under COUNT_ONLY/STATISTICS).
        key_lookup = f"{schema_name}.{table_name}"
        row_hash_key_columns = request.primary_keys.get(key_lookup) or request.primary_keys.get(
            table_name
        )

        if row_hash_key_columns:
            try:
                mismatches, mismatch_count, mismatch_pct = self._run_row_hash_stage(
                    request, schema_name, table_name, common_cols, row_hash_key_columns,
                )
                if result.data is None:
                    result.data = DataValidationResult(
                        mode=request.data_compare_mode,
                        status=ValidationStatus.SKIPPED,
                        note="Row-level EXCEPT/hash-join comparison not run for this mode.",
                    )
                result.data.row_hash_mismatches = mismatches
                result.data.row_hash_mismatch_count = mismatch_count
                result.data.row_hash_mismatch_percentage = mismatch_pct
                if mismatch_count > 0 and result.data.status != ValidationStatus.ERROR:
                    result.data.status = ValidationStatus.FAIL
                elif result.data.status == ValidationStatus.SKIPPED and mismatch_count == 0:
                    # Row hashes matched perfectly - that's a real PASS signal,
                    # not "nothing was checked".
                    result.data.status = ValidationStatus.PASS
            except Exception as exc:
                logger.exception(
                    "Failed to run row-hash comparison for '%s.%s'", schema_name, table_name
                )
                if result.data is None:
                    result.data = DataValidationResult(
                        mode=request.data_compare_mode,
                        status=ValidationStatus.ERROR,
                        key_columns=row_hash_key_columns,
                        error=f"Row-hash comparison failed: {exc}",
                    )
                else:
                    result.data.status = ValidationStatus.ERROR
                    result.data.error = f"Row-hash comparison failed: {exc}"

        # Stage 16: overall table status
        result.status = self.calculate_overall_status(
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
    # Stage 5/6: column name comparison
    # ------------------------------------------------------------------
    def compare_columns(
        self,
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
        common = sorted(
            src_cols[k] for k in (set(src_cols) & set(tgt_cols))
        )

        return missing, extra, common

    # ------------------------------------------------------------------
    # Stage 7: data types (per-column, used above; exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_data_types(self, source_type: str, target_type: str) -> ValidationStatus:
        return ValidationStatus.PASS if source_type == target_type else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 8: nullable (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_nullable(self, source_nullable: bool, target_nullable: bool) -> ValidationStatus:
        return ValidationStatus.PASS if source_nullable == target_nullable else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 9: column order (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_column_order(
        self, source_order: List[str], target_order: List[str]
    ) -> ValidationStatus:
        return (
            ValidationStatus.PASS
            if [c.lower() for c in source_order] == [c.lower() for c in target_order]
            else ValidationStatus.FAIL
        )

    # ------------------------------------------------------------------
    # Stage 10: row counts (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_row_counts(self, source_count: int, target_count: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_count == target_count else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 11/12: null + distinct counts (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_null_counts(self, source_nulls: int, target_nulls: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_nulls == target_nulls else ValidationStatus.FAIL

    def compare_distinct_counts(self, source_distinct: int, target_distinct: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_distinct == target_distinct else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 13: min/max (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_min_max(
        self, source_min: Any, source_max: Any, target_min: Any, target_max: Any
    ) -> ValidationStatus:
        return (
            ValidationStatus.PASS
            if source_min == target_min and source_max == target_max
            else ValidationStatus.FAIL
        )

    # ------------------------------------------------------------------
    # Stage 15: actual data comparison
    # ------------------------------------------------------------------
    def compare_data(
        self,
        request: CatalogValidationRequest,
        schema_name: str,
        table_name: str,
        common_columns: List[str],
    ) -> DataValidationResult:

        mode = request.data_compare_mode
        key = f"{schema_name}.{table_name}"
        key_columns = request.primary_keys.get(key) or request.primary_keys.get(
            table_name
        )

        if mode == DataCompareMode.COUNT_ONLY:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note="COUNT_ONLY mode: row-level data comparison skipped by configuration.",
            )

        if mode == DataCompareMode.STATISTICS:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note=(
                    "STATISTICS mode (default): row count / null / distinct / "
                    "min-max already validated above; row-level comparison skipped "
                    "for cost. Use HASH or FULL to enable it."
                ),
            )

        if not key_columns:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note=(
                    f"No primary/business key configured for '{key}' - "
                    "row-level data comparison requires a key and was skipped. "
                    "Configure request.primary_keys to enable it."
                ),
            )

        missing_keys = [k for k in key_columns if k.lower() not in {c.lower() for c in common_columns}]
        if missing_keys:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.ERROR,
                key_columns=key_columns,
                error=f"Configured key column(s) not found as common columns: {missing_keys}",
            )

        value_columns = [
            c for c in common_columns if c.lower() not in {k.lower() for k in key_columns}
        ]

        source_fqtn = f"{request.source_catalog}.{schema_name}.{table_name}"
        target_fqtn = f"{request.target_catalog}.{schema_name}.{table_name}"

        try:
            diff = self.databricks.key_based_row_diff(
                source_fqtn=source_fqtn,
                target_fqtn=target_fqtn,
                key_columns=key_columns,
                # HASH mode: only need counts, not full samples of value diffs;
                # FULL mode: return samples too. Either way this is pushed down.
                value_columns=value_columns,
                limit_samples=request.max_sample_rows if mode == DataCompareMode.FULL else 5,
            )
        except Exception as exc:
            logger.exception(
                "Failed to run key-based data comparison for '%s'", key
            )
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.ERROR,
                key_columns=key_columns,
                error=f"Data comparison failed: {exc}",
            )

        has_diff = (
            diff["source_only_rows"] > 0
            or diff["target_only_rows"] > 0
            or diff["changed_rows"] > 0
        )

        sample_changed_detail: List[RowMismatchDetail] = []
        if mode == DataCompareMode.FULL:
            for row in diff.get("sample_changed_detail", []):
                for col in row["mismatched_columns"]:
                    sample_changed_detail.append(
                        RowMismatchDetail(
                            schema_name=schema_name,
                            table=table_name,
                            primary_key=row["key"],
                            mismatch_column=col,
                            source_value=row["source_values"].get(col),
                            target_value=row["target_values"].get(col),
                            source_row_hash=row["source_row_hash"],
                            target_row_hash=row["target_row_hash"],
                        )
                    )

        return DataValidationResult(
            mode=mode,
            status=ValidationStatus.FAIL if has_diff else ValidationStatus.PASS,
            key_columns=key_columns,
            source_only_rows=diff["source_only_rows"],
            target_only_rows=diff["target_only_rows"],
            changed_rows=diff["changed_rows"],
            sample_source_only=(
                diff["sample_source_only"] if mode == DataCompareMode.FULL else []
            ),
            sample_target_only=(
                diff["sample_target_only"] if mode == DataCompareMode.FULL else []
            ),
            sample_changed=(
                diff["sample_changed"] if mode == DataCompareMode.FULL else []
            ),
            sample_changed_detail=sample_changed_detail,
        )

    # ------------------------------------------------------------------
    # Stage 15b: row-hash comparison
    #
    # Separate mechanism from key_based_row_diff/_changed_row_detail above
    # (which only run in HASH/FULL mode): this is a single pushed-down
    # whole-row hash per side, joined by primary key in Python (never by
    # row position/order), and is the primary way to detect row-level
    # mismatches whenever a primary key is configured - independent of
    # data_compare_mode.
    # ------------------------------------------------------------------
    def _run_row_hash_stage(
        self,
        request: CatalogValidationRequest,
        schema_name: str,
        table_name: str,
        common_columns: List[str],
        key_columns: List[str],
    ) -> Tuple[List[RowHashMismatch], int, float]:

        value_columns = sorted(
            c for c in common_columns if c.lower() not in {k.lower() for k in key_columns}
        )

        source_hashes = self.databricks.get_row_hashes(
            request.source_catalog, schema_name, table_name, value_columns, key_columns,
        )
        target_hashes = self.databricks.get_row_hashes(
            request.target_catalog, schema_name, table_name, value_columns, key_columns,
        )

        return self.compare_row_hashes(source_hashes, target_hashes, key_columns)

    @staticmethod
    def compare_row_hashes(
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
        primary_key_cols: Sequence[str],
    ) -> Tuple[List[RowHashMismatch], int, float]:
        """
        Join two per-key row-hash sets by primary key (never row position)
        and classify every key as matching, MISMATCH (key on both sides,
        hash differs), MISSING_IN_TARGET, or MISSING_IN_SOURCE.

        Returns (mismatches, mismatch_count, mismatch_percentage) where
        mismatch_percentage is mismatch_count / total_compared_keys * 100
        and total_compared_keys is the union of keys seen on either side.
        """

        def _display_key(row: pd.Series) -> str:
            return "|".join(str(row[k]) for k in primary_key_cols)

        def _key_tuple(row: pd.Series) -> tuple:
            return tuple(row[k] for k in primary_key_cols)

        source_by_key = {
            _key_tuple(row): row for _, row in source_hashes.iterrows()
        }
        target_by_key = {
            _key_tuple(row): row for _, row in target_hashes.iterrows()
        }

        all_keys = set(source_by_key) | set(target_by_key)

        mismatches: List[RowHashMismatch] = []

        for key_tuple in all_keys:
            src_row = source_by_key.get(key_tuple)
            tgt_row = target_by_key.get(key_tuple)

            if src_row is not None and tgt_row is None:
                mismatches.append(
                    RowHashMismatch(
                        primary_key=_display_key(src_row),
                        source_hash=str(src_row["row_hash"]),
                        target_hash="",
                        status="MISSING_IN_TARGET",
                    )
                )
            elif src_row is None and tgt_row is not None:
                mismatches.append(
                    RowHashMismatch(
                        primary_key=_display_key(tgt_row),
                        source_hash="",
                        target_hash=str(tgt_row["row_hash"]),
                        status="MISSING_IN_SOURCE",
                    )
                )
            elif src_row is not None and tgt_row is not None:
                if src_row["row_hash"] != tgt_row["row_hash"]:
                    mismatches.append(
                        RowHashMismatch(
                            primary_key=_display_key(src_row),
                            source_hash=str(src_row["row_hash"]),
                            target_hash=str(tgt_row["row_hash"]),
                            status="MISMATCH",
                        )
                    )

        total_compared_keys = len(all_keys)
        mismatch_count = len(mismatches)
        mismatch_percentage = (
            (mismatch_count / total_compared_keys) * 100 if total_compared_keys else 0.0
        )

        return mismatches, mismatch_count, mismatch_percentage

    # ------------------------------------------------------------------
    # Stage 16/17: overall status aggregation (programmatic, never hardcoded)
    # ------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Module-level convenience function (per spec section 22 / app.py usage)
# ----------------------------------------------------------------------
def validate_catalogs(
    databricks_connector: DatabricksConnector,
    source_catalog: str,
    target_catalog: str,
    **kwargs: Any,
) -> CatalogValidationResponse:
    """
    Thin convenience wrapper so callers (app.py, CLI, scripts) don't need
    to construct CatalogValidationRequest / CatalogValidator by hand for
    the common case.
    """
    request = CatalogValidationRequest(
        source_catalog=source_catalog,
        target_catalog=target_catalog,
        **kwargs,
    )
    validator = CatalogValidator(databricks_connector)
    return validator.compare_catalogs(request)