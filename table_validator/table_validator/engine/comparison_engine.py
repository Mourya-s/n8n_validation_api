"""
Comparison Engine

Row-level comparison of a source CSV (Azure Storage) against a target
table (Databricks Delta Lake) - the original/legacy comparison path.

Receives already-authenticated connector instances and never performs
authentication itself.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from table_validator.connectors.azure_connector import AzureConnector
from table_validator.connectors.databricks_connector import DatabricksConnector
from table_validator.models import (
    ComparisonRequest,
    ComparisonResponse,
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
