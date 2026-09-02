"""
Databricks SQL Warehouse Connector

Responsible solely for establishing connectivity to a Databricks SQL Warehouse
and retrieving data / schema information.

Contains no comparison logic. Every method here answers a factual question
("does this catalog exist", "what are the null counts for these columns")
- it never decides PASS/FAIL. That decision lives in validators/catalog_validator.py.
"""

from __future__ import annotations

import datetime
import logging
import numbers
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection

if TYPE_CHECKING:
    from table_validator.models import HashCanonicalizationSpec

logger = logging.getLogger(__name__)

# numbers.Number covers int/float/Decimal AND numpy's numeric scalar types
# (np.int64, np.float64, ...) in one check - values returned by pandas/
# numpy-backed connectors (Databricks) need to compare cleanly against
# plain Decimal/int values from a DB-API driver (e.g. pyodbc), and
# Decimal.__eq__ raises TypeError rather than returning False when handed
# a numpy scalar it doesn't recognize.
_NUMERIC_TYPES = numbers.Number


def values_differ(a: Any, b: Any) -> bool:
    """
    Robust value comparison for two independently-fetched cells (e.g. one
    row read from a source and one from a target, via separate round-trips
    - a SQL query pair, or a CSV row vs a SQL row).

    Raw `!=` is too strict here: two sides can return numerically-equal
    values with different Python representations for the SAME underlying
    type family (e.g. Decimal vs float precision, or date vs datetime),
    which would otherwise register as a false mismatch against a row
    already flagged as changed by a whole-row hash comparison.

    Deliberately NOT applied across a numeric-vs-string type change (e.g.
    a column migrated from double to string) - that IS a real, reportable
    difference even if the string happens to parse to the same number.
    """
    if a is None or b is None:
        return a is not b

    if isinstance(a, _NUMERIC_TYPES) and isinstance(b, _NUMERIC_TYPES):
        return abs(float(a) - float(b)) > 1e-9

    if isinstance(a, (datetime.date, datetime.datetime)) and isinstance(
        b, (datetime.date, datetime.datetime)
    ):
        a_cmp = a.date() if isinstance(a, datetime.datetime) else a
        b_cmp = b.date() if isinstance(b, datetime.datetime) else b
        return a_cmp != b_cmp

    # A type change between the two sides (e.g. one side returned a number,
    # the other a string) is itself a real difference, even if their
    # string forms happen to look identical.
    if type(a) is not type(b):
        return True

    return str(a) != str(b)


# Data types for which MIN/MAX is meaningful. Kept as a prefix match against
# the raw Databricks type string (e.g. "decimal(10,2)" -> "decimal").
_MIN_MAX_ELIGIBLE_TYPE_PREFIXES = (
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "double",
    "decimal",
    "date",
    "timestamp",
)


class DatabricksConnector:
    """
    Lightweight reusable connector for Databricks SQL Warehouse.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        http_path: Optional[str] = None,
        retry_timeout_seconds: Optional[float] = None,
    ) -> None:
        """
        host/token/http_path must be resolved by the caller before
        construction - e.g. via table_validator.auth.databricks_auth.
        get_databricks_token() for the token, and config.databricks.
        workspace_url/http_path for the rest. This connector does not
        read credentials from the environment itself.

        retry_timeout_seconds overrides databricks-sql-connector's
        CloudFetch HTTP retry policy (_retry_stop_after_attempts_duration),
        which otherwise defaults to 300 seconds. On a slow or unstable
        network, downloading a large row-hash result set can legitimately
        take longer than that, causing a hard
        "Retry request would exceed Retry policy max retry duration"
        failure even though the query itself succeeded - raising this
        gives a slow-but-working connection more time instead of giving
        up. Falls back to the DATABRICKS_RETRY_TIMEOUT_SECONDS environment
        variable, then the connector's own default, if not given.
        """

        self._host = host
        self._token = token
        self._http_path = http_path
        self._retry_timeout_seconds = retry_timeout_seconds or (
            float(os.environ["DATABRICKS_RETRY_TIMEOUT_SECONDS"])
            if os.environ.get("DATABRICKS_RETRY_TIMEOUT_SECONDS")
            else None
        )

        if not self._host or not self._token:
            raise ValueError(
                "Databricks host and token are required. "
                "Provide them via constructor arguments."
            )

        if not self._http_path:
            raise ValueError(
                "Databricks HTTP path is required."
            )

        self._connection: Optional[Connection] = None

        logger.debug(
            "DatabricksConnector initialized for host=%s",
            self._host,
        )

    # ------------------------------------------------------------------
    # Connection Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:

        if self._connection is not None:
            return

        try:

            connect_kwargs: Dict[str, Any] = {
                "server_hostname": self._host,
                "http_path": self._http_path,
                "access_token": self._token,
            }
            if self._retry_timeout_seconds is not None:
                connect_kwargs["_retry_stop_after_attempts_duration"] = (
                    self._retry_timeout_seconds
                )

            self._connection = sql.connect(**connect_kwargs)

            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()

            logger.info(
                "Successfully connected to Databricks SQL Warehouse"
            )

        except Exception as exc:

            self._connection = None

            logger.exception(
                "Failed to connect to Databricks SQL Warehouse"
            )

            raise ConnectionError(
                f"Unable to connect to Databricks: {exc}"
            ) from exc

    def disconnect(self) -> None:

        if self._connection is None:
            return

        try:

            self._connection.close()

            logger.info(
                "Disconnected from Databricks SQL Warehouse"
            )

        except Exception as exc:

            logger.warning(
                "Error while closing Databricks connection: %s",
                exc,
            )

        finally:
            self._connection = None

    def test_connection(self) -> bool:

        try:

            self.connect()

            with self._connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchall()

            logger.info(
                "Databricks connection test succeeded"
            )

            return True

        except Exception as exc:

            logger.error(
                "Databricks connection test failed: %s",
                exc,
            )

            return False

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> Connection:

        if self._connection is None:
            self.connect()

        if self._connection is None:
            raise ConnectionError(
                "Databricks connection is not available"
            )

        return self._connection

    def _execute_to_dataframe(
        self,
        query: str,
    ) -> pd.DataFrame:

        connection = self._ensure_connected()

        try:

            with connection.cursor() as cursor:

                cursor.execute(query)

                if cursor.description is None:
                    return pd.DataFrame()

                columns = [
                    desc[0]
                    for desc in cursor.description
                ]

                rows = cursor.fetchall()

                return pd.DataFrame(
                    rows,
                    columns=columns,
                )

        except Exception as exc:

            logger.exception(
                "Failed to execute query against Databricks"
            )

            raise RuntimeError(
                f"Unable to execute query: {exc}"
            ) from exc

    @staticmethod
    def _quote_ident(identifier: str) -> str:
        """Backtick-quote a single identifier part, escaping embedded backticks."""
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"

    @classmethod
    def _qualify(cls, *parts: str) -> str:
        """Build a fully-qualified, backtick-quoted `a`.`b`.`c` identifier."""
        return ".".join(cls._quote_ident(p) for p in parts if p is not None)

    # ------------------------------------------------------------------
    # Data Retrieval (existing - unchanged)
    # ------------------------------------------------------------------
    def read_table(
        self,
        table_name: str,
    ) -> pd.DataFrame:

        if not table_name or not table_name.strip():
            raise ValueError(
                "table_name must be a non-empty string"
            )

        safe_name = ".".join(
            f"`{part}`"
            for part in table_name.split(".")
        )

        query = f"SELECT * FROM {safe_name}"

        logger.info(
            "Reading table '%s' from Databricks",
            table_name,
        )

        df = self._execute_to_dataframe(query)

        logger.info(
            "Successfully read table '%s' - shape=%s",
            table_name,
            df.shape,
        )

        return df

    def read_query(
        self,
        query: str,
    ) -> pd.DataFrame:

        if not query or not query.strip():
            raise ValueError(
                "query must be a non-empty string"
            )

        logger.info(
            "Executing custom query against Databricks"
        )

        df = self._execute_to_dataframe(query)

        logger.info(
            "Query returned shape=%s",
            df.shape,
        )

        return df

    def get_schema(
        self,
        table_name: str,
    ) -> pd.DataFrame:

        if not table_name or not table_name.strip():
            raise ValueError(
                "table_name must be a non-empty string"
            )

        safe_name = ".".join(
            f"`{part}`"
            for part in table_name.split(".")
        )

        describe_query = (
            f"DESCRIBE TABLE {safe_name}"
        )

        try:

            raw_df = self._execute_to_dataframe(
                describe_query
            )

            if raw_df.empty:

                return pd.DataFrame(
                    columns=[
                        "column_name",
                        "data_type",
                        "is_nullable",
                        "character_maximum_length",
                    ]
                )

            raw_df = raw_df[
                raw_df["col_name"].notna()
            ]

            raw_df = raw_df[
                raw_df["col_name"] != ""
            ]

            raw_df = raw_df[
                ~raw_df["col_name"].astype(str).str.startswith("#")
            ]

            schema_df = pd.DataFrame(
                {
                    "column_name": raw_df["col_name"],
                    "data_type": raw_df["data_type"],
                    "is_nullable": "YES",
                    "character_maximum_length": None,
                }
            )

            logger.info(
                "Schema for '%s' retrieved - %d columns",
                table_name,
                len(schema_df),
            )

            return schema_df.reset_index(drop=True)

        except Exception as exc:

            logger.exception(
                "Failed to retrieve schema for '%s'",
                table_name,
            )

            raise RuntimeError(
                f"Unable to retrieve schema for '{table_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # NEW: generic passthrough (public alias, used by CatalogValidator
    # for anything not covered by a dedicated method below)
    # ------------------------------------------------------------------
    def execute_query(self, query: str) -> pd.DataFrame:
        """Public entry point for executing an arbitrary read-only query."""
        return self._execute_to_dataframe(query)

    # ------------------------------------------------------------------
    # NEW: Catalog / schema / table metadata methods
    # ------------------------------------------------------------------
    def catalog_exists(self, catalog: str) -> bool:
        try:
            df = self._execute_to_dataframe("SHOW CATALOGS")
        except Exception as exc:
            logger.exception("Failed to list catalogs")
            raise RuntimeError(f"Unable to list catalogs: {exc}") from exc

        if df.empty:
            return False

        col = df.columns[0]
        return catalog.lower() in {str(v).lower() for v in df[col]}

    def get_schemas(self, catalog: str) -> List[str]:
        try:
            df = self._execute_to_dataframe(
                f"SHOW SCHEMAS IN {self._quote_ident(catalog)}"
            )
        except Exception as exc:
            logger.exception("Failed to list schemas for catalog '%s'", catalog)
            raise RuntimeError(
                f"Unable to list schemas for catalog '{catalog}': {exc}"
            ) from exc

        if df.empty:
            return []

        col = "databaseName" if "databaseName" in df.columns else df.columns[0]
        return sorted(str(v) for v in df[col])

    def schema_exists(self, catalog: str, schema: str) -> bool:
        return schema.lower() in {s.lower() for s in self.get_schemas(catalog)}

    def get_tables(self, catalog: str, schema: str) -> List[str]:
        try:
            df = self._execute_to_dataframe(
                f"SHOW TABLES IN {self._qualify(catalog, schema)}"
            )
        except Exception as exc:
            logger.exception(
                "Failed to list tables for '%s.%s'", catalog, schema
            )
            raise RuntimeError(
                f"Unable to list tables for '{catalog}.{schema}': {exc}"
            ) from exc

        if df.empty:
            return []

        col = "tableName" if "tableName" in df.columns else df.columns[0]
        return sorted(str(v) for v in df[col])

    def table_exists(self, catalog: str, schema: str, table: str) -> bool:
        return table.lower() in {t.lower() for t in self.get_tables(catalog, schema)}

    def get_table_schema(
        self,
        catalog: str,
        schema: str,
        table: str,
    ) -> pd.DataFrame:
        """
        Returns columns: column_name, data_type, is_nullable (bool),
        ordinal_position - sourced from information_schema, which (unlike
        DESCRIBE TABLE) gives real nullability and a reliable column order.
        """
        query = f"""
            SELECT column_name, full_data_type AS data_type,
                   is_nullable, ordinal_position
            FROM {self._quote_ident(catalog)}.information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
            ORDER BY ordinal_position
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s.%s'",
                catalog, schema, table,
            )
            raise RuntimeError(
                f"Unable to retrieve column metadata for "
                f"'{catalog}.{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(
                columns=["column_name", "data_type", "is_nullable", "ordinal_position"]
            )

        df["is_nullable"] = df["is_nullable"].astype(str).str.upper().eq("YES")
        return df.reset_index(drop=True)

    def get_row_count(self, catalog: str, schema: str, table: str) -> int:
        query = f"SELECT COUNT(*) AS row_count FROM {self._qualify(catalog, schema, table)}"
        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to get row count for '%s.%s.%s'", catalog, schema, table
            )
            raise RuntimeError(
                f"Unable to get row count for '{catalog}.{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return 0
        return int(df.iloc[0]["row_count"])

    def get_column_statistics(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        min_max_columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Single aggregate query returning null count, distinct count, and
        (for min_max_columns) MIN/MAX for every requested column - avoids
        one round-trip per column.

        Returns: {column_name: {"null_count": int, "distinct_count": int,
                                 "min": Any | None, "max": Any | None}}
        """
        if not columns:
            return {}

        min_max_set = {c.lower() for c in (min_max_columns or [])}

        select_parts = []
        for col in columns:
            q = self._quote_ident(col)
            select_parts.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS `{col}__nulls`")
            select_parts.append(f"COUNT(DISTINCT {q}) AS `{col}__distinct`")
            if col.lower() in min_max_set:
                select_parts.append(f"MIN({q}) AS `{col}__min`")
                select_parts.append(f"MAX({q}) AS `{col}__max`")

        query = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM {self._qualify(catalog, schema, table)}"
        )

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s.%s.%s'",
                catalog, schema, table,
            )
            raise RuntimeError(
                f"Unable to compute column statistics for "
                f"'{catalog}.{schema}.{table}': {exc}"
            ) from exc

        result: Dict[str, Dict[str, Any]] = {}

        if df.empty:
            return {col: {"null_count": None, "distinct_count": None,
                           "min": None, "max": None} for col in columns}

        row = df.iloc[0]

        for col in columns:
            entry: Dict[str, Any] = {
                "null_count": int(row.get(f"{col}__nulls"))
                if row.get(f"{col}__nulls") is not None else None,
                "distinct_count": int(row.get(f"{col}__distinct"))
                if row.get(f"{col}__distinct") is not None else None,
                "min": None,
                "max": None,
            }
            if col.lower() in min_max_set:
                entry["min"] = row.get(f"{col}__min")
                entry["max"] = row.get(f"{col}__max")
            result[col] = entry

        return result

    @staticmethod
    def is_min_max_eligible(data_type: str) -> bool:
        dt = (data_type or "").strip().lower()
        return any(dt.startswith(prefix) for prefix in _MIN_MAX_ELIGIBLE_TYPE_PREFIXES)

    def _row_hash_expr(
        self,
        columns: Sequence[str],
        spec: Optional["HashCanonicalizationSpec"] = None,
    ) -> str:
        """
        Build the canonical per-row hash SQL expression shared by
        get_row_hashes, get_row_hashes_by_row_number, and
        get_table_fingerprint, so all three tiers hash identically.

        Defaults reproduce the original inline expression byte-for-byte:
        sha2(concat_ws('||', COALESCE(CAST(col AS STRING), sentinel)...), 256).
        `spec` is accepted for forward compatibility with
        HashCanonicalizationSpec but is not yet applied here - see
        models.HashCanonicalizationSpec docstring.
        """
        null_sentinel = (spec.null_sentinel if spec else None) or "\x01NULL\x01"

        hashed_exprs = [
            f"COALESCE(CAST({self._quote_ident(c)} AS STRING), '{null_sentinel}')"
            for c in columns
        ]

        if hashed_exprs:
            return f"sha2(concat_ws('||', {', '.join(hashed_exprs)}), 256)"
        return f"sha2('{null_sentinel}', 256)"

    def get_table_fingerprint(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        spec: Optional["HashCanonicalizationSpec"] = None,
    ) -> Dict[str, Any]:
        """
        Tier 2: single order-independent whole-table fingerprint, computed
        entirely server-side - no row data ever leaves the warehouse.

        Combines three aggregates that are individually weak but strong
        together: COUNT(*) alone misses swapped/altered values; SUM alone
        is collision-prone/overflow-prone; XOR alone is blind to
        duplicated rows. A 15-hex-char (60-bit) prefix of the per-row hash
        is converted to a numeric value via conv(hex, 16, 10) - Databricks
        SQL has no native hex-to-numeric cast - which keeps the XOR
        argument within BIGINT range, while the SUM accumulates as
        DECIMAL(38,0) to stay overflow-safe across an entire table.

        Returns {"row_count": int, "hash_sum": Decimal|None, "hash_xor": int|None}.
        """
        row_hash_expr = self._row_hash_expr(columns, spec)
        hash_prefix = f"substr({row_hash_expr}, 1, 15)"

        query = f"""
            SELECT
                COUNT(*) AS row_count,
                SUM(CAST(conv({hash_prefix}, 16, 10) AS DECIMAL(38,0))) AS hash_sum,
                BIT_XOR(CAST(conv({hash_prefix}, 16, 10) AS BIGINT)) AS hash_xor
            FROM {self._qualify(catalog, schema, table)}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute table fingerprint for '%s.%s.%s'",
                catalog, schema, table,
            )
            raise RuntimeError(
                f"Unable to compute table fingerprint for "
                f"'{catalog}.{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return {"row_count": 0, "hash_sum": None, "hash_xor": None}

        row = df.iloc[0]
        return {
            "row_count": int(row.get("row_count") or 0),
            "hash_sum": row.get("hash_sum"),
            "hash_xor": row.get("hash_xor"),
        }

    def get_table_fingerprint_by_bucket(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        bucket_column: str,
        spec: Optional["HashCanonicalizationSpec"] = None,
    ) -> pd.DataFrame:
        """
        Tier 3: the same triple fingerprint as get_table_fingerprint, but
        GROUP BY a chosen bucket column - one row per distinct bucket
        value, computed entirely server-side. Used to narrow a confirmed
        table-level mismatch down to the specific bucket(s) that actually
        differ, so Tier 4's row-hash diff only needs to scan those buckets
        instead of the whole table.

        Returns a DataFrame with columns: bucket_value, row_count,
        hash_sum, hash_xor - one row per distinct value of bucket_column
        present on this side.
        """
        row_hash_expr = self._row_hash_expr(columns, spec)
        hash_prefix = f"substr({row_hash_expr}, 1, 15)"
        bucket_ident = self._quote_ident(bucket_column)

        query = f"""
            SELECT
                {bucket_ident} AS bucket_value,
                COUNT(*) AS row_count,
                SUM(CAST(conv({hash_prefix}, 16, 10) AS DECIMAL(38,0))) AS hash_sum,
                BIT_XOR(CAST(conv({hash_prefix}, 16, 10) AS BIGINT)) AS hash_xor
            FROM {self._qualify(catalog, schema, table)}
            GROUP BY {bucket_ident}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute bucketed table fingerprint for '%s.%s.%s' "
                "(bucket_column='%s')",
                catalog, schema, table, bucket_column,
            )
            raise RuntimeError(
                f"Unable to compute bucketed table fingerprint for "
                f"'{catalog}.{schema}.{table}' (bucket_column='{bucket_column}'): {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(columns=["bucket_value", "row_count", "hash_sum", "hash_xor"])

        return df

    def key_based_row_diff(
        self,
        source_fqtn: str,
        target_fqtn: str,
        key_columns: Sequence[str],
        value_columns: Sequence[str],
        limit_samples: int = 50,
    ) -> Dict[str, Any]:
        """
        Push-down key-based row comparison between two fully-qualified
        tables (e.g. 'cat_a.schema.table' vs 'cat_b.schema.table').

        Returns counts of source-only / target-only rows (via SQL EXCEPT
        on key columns) and changed-row count (matching key, differing
        row hash), plus a small bounded sample of each - never a full
        collect() of either table.
        """
        key_idents = [self._quote_ident(k) for k in key_columns]
        key_list = ", ".join(key_idents)

        src = ".".join(self._quote_ident(p) for p in source_fqtn.split("."))
        tgt = ".".join(self._quote_ident(p) for p in target_fqtn.split("."))

        # Source-only / target-only keys via EXCEPT (fully pushed down)
        source_only_query = f"""
            SELECT {key_list} FROM {src}
            EXCEPT
            SELECT {key_list} FROM {tgt}
        """
        target_only_query = f"""
            SELECT {key_list} FROM {tgt}
            EXCEPT
            SELECT {key_list} FROM {src}
        """

        source_only_df = self._execute_to_dataframe(
            f"SELECT COUNT(*) AS c FROM ({source_only_query}) x"
        )
        target_only_df = self._execute_to_dataframe(
            f"SELECT COUNT(*) AS c FROM ({target_only_query}) x"
        )

        source_only_count = int(source_only_df.iloc[0]["c"]) if not source_only_df.empty else 0
        target_only_count = int(target_only_df.iloc[0]["c"]) if not target_only_df.empty else 0

        sample_source_only = self._execute_to_dataframe(
            f"{source_only_query} LIMIT {int(limit_samples)}"
        ).to_dict(orient="records")
        sample_target_only = self._execute_to_dataframe(
            f"{target_only_query} LIMIT {int(limit_samples)}"
        ).to_dict(orient="records")

        # Changed rows: matching key, differing hash of value columns
        changed_count = 0
        sample_changed: List[Dict[str, Any]] = []
        sample_changed_detail: List[Dict[str, Any]] = []

        if value_columns:
            value_concat = ", ".join(self._quote_ident(c) for c in value_columns)
            changed_query = f"""
                SELECT {key_list} FROM (
                    SELECT {key_list}, hash({value_concat}) AS __row_hash
                    FROM {src}
                ) s
                JOIN (
                    SELECT {key_list}, hash({value_concat}) AS __row_hash
                    FROM {tgt}
                ) t
                USING ({key_list})
                WHERE s.__row_hash != t.__row_hash
            """
            changed_df = self._execute_to_dataframe(
                f"SELECT COUNT(*) AS c FROM ({changed_query}) x"
            )
            changed_count = int(changed_df.iloc[0]["c"]) if not changed_df.empty else 0

            sample_changed = self._execute_to_dataframe(
                f"{changed_query} LIMIT {int(limit_samples)}"
            ).to_dict(orient="records")

            sample_changed_detail = self._changed_row_detail(
                src=src,
                tgt=tgt,
                key_columns=key_columns,
                key_idents=key_idents,
                key_list=key_list,
                value_columns=value_columns,
                changed_query=changed_query,
                limit_samples=limit_samples,
            )

        return {
            "source_only_rows": source_only_count,
            "target_only_rows": target_only_count,
            "changed_rows": changed_count,
            "sample_source_only": sample_source_only,
            "sample_target_only": sample_target_only,
            "sample_changed": sample_changed,
            "sample_changed_detail": sample_changed_detail,
        }

    def get_row_detail_for_keys(
        self,
        source_catalog: str,
        target_catalog: str,
        schema: str,
        table: str,
        key_column: str,
        key_values: Sequence[str],
        value_columns: Sequence[str],
        limit_samples: int = 500,
        target_value_columns: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tier 5: column-level diff for a bounded, already-known set of
        mismatched keys (single-column key only). Fetches key + value
        columns plus a whole-row hash() from both sides for exactly
        those keys - never a full-table pull - and diffs them
        column-by-column so callers can report exactly which column(s)
        differ per row. `key_values` are treated as opaque string literals
        (matching Tier 4's compare_row_hashes display-key convention).

        `value_columns` is the SOURCE-side spelling; `target_value_columns`
        (when a column_map applies), the positionally-aligned TARGET-side
        spelling for the same columns. `key_column` itself is assumed
        identical on both sides (a column used as a primary key must not
        also be renamed via column_map - enforced at request-resolution
        time, not here).
        """
        if not key_values:
            return []

        src = self._qualify(source_catalog, schema, table)
        tgt = self._qualify(target_catalog, schema, table)
        key_ident = self._quote_ident(key_column)
        quoted_values = ", ".join(f"'{str(v).replace(chr(39), chr(39) * 2)}'" for v in key_values)
        changed_query = (
            f"SELECT {key_ident} FROM {src} "
            f"WHERE CAST({key_ident} AS STRING) IN ({quoted_values})"
        )

        return self._changed_row_detail(
            src=src,
            tgt=tgt,
            key_columns=[key_column],
            key_idents=[key_ident],
            key_list=key_ident,
            value_columns=value_columns,
            changed_query=changed_query,
            limit_samples=limit_samples,
            target_value_columns=target_value_columns,
        )

    def _changed_row_detail(
        self,
        src: str,
        tgt: str,
        key_columns: Sequence[str],
        key_idents: List[str],
        key_list: str,
        value_columns: Sequence[str],
        changed_query: str,
        limit_samples: int,
        target_value_columns: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        For a bounded sample of changed keys (from `changed_query`), fetch
        the full source and target rows (key + value columns) plus a
        whole-row hash for each side, so callers can report exactly which
        column(s) differ per row without ever collecting a full table.

        `value_columns` is always the SOURCE-side column spelling.
        `target_value_columns`, when given, is the positionally-aligned
        TARGET-side spelling for the same columns (a column_map case) -
        the source and target SQL each use their own side's names, and
        the result is reconciled back to ONE canonical label per pair
        (the target name, or the shared name when unmapped) so callers
        never have to know which side a given result dict's keys came
        from. When omitted, target_value_columns defaults to
        value_columns (today's behavior, unchanged).
        """
        target_value_columns = list(target_value_columns or value_columns)
        source_to_target = dict(zip(value_columns, target_value_columns))

        source_value_idents = [self._quote_ident(c) for c in value_columns]
        target_value_idents = [self._quote_ident(c) for c in target_value_columns]
        source_select_list = ", ".join(key_idents + source_value_idents)
        target_select_list = ", ".join(key_idents + target_value_idents)
        source_concat = ", ".join(source_value_idents)
        target_concat = ", ".join(target_value_idents)

        source_rows = self._execute_to_dataframe(f"""
            SELECT {source_select_list}, hash({source_concat}) AS __row_hash
            FROM {src}
            WHERE ({key_list}) IN (SELECT {key_list} FROM ({changed_query} LIMIT {int(limit_samples)}) __k)
        """).to_dict(orient="records")

        target_rows = self._execute_to_dataframe(f"""
            SELECT {target_select_list}, hash({target_concat}) AS __row_hash
            FROM {tgt}
            WHERE ({key_list}) IN (SELECT {key_list} FROM ({changed_query} LIMIT {int(limit_samples)}) __k)
        """).to_dict(orient="records")

        def _key_tuple(row: Dict[str, Any]) -> tuple:
            return tuple(row.get(k) for k in key_columns)

        target_by_key = {_key_tuple(r): r for r in target_rows}

        detail: List[Dict[str, Any]] = []
        for src_row in source_rows:
            tgt_row = target_by_key.get(_key_tuple(src_row))
            if tgt_row is None:
                continue

            # Diff by pair, but report every result keyed by the
            # canonical (target) column name - src_row/tgt_row are read
            # using each side's own real column name.
            mismatched_columns = [
                source_to_target[src_col] for src_col in value_columns
                if values_differ(src_row.get(src_col), tgt_row.get(source_to_target[src_col]))
            ]
            if not mismatched_columns:
                # SQL-side hash() flagged this row as changed, but our
                # tolerant per-column comparison found nothing - the two
                # comparisons disagree (e.g. a difference the hash catches
                # that our value comparison normalizes away). Report the
                # row anyway rather than silently dropping a row the
                # mismatch count already accounts for.
                mismatched_columns = list(target_value_columns)

            detail.append(
                {
                    "key": {k: src_row.get(k) for k in key_columns},
                    "mismatched_columns": mismatched_columns,
                    "source_values": {
                        source_to_target[src_col]: src_row.get(src_col)
                        for src_col in value_columns
                        if source_to_target[src_col] in mismatched_columns
                    },
                    "target_values": {
                        tgt_col: tgt_row.get(tgt_col)
                        for tgt_col in mismatched_columns
                    },
                    "source_row_hash": src_row.get("__row_hash"),
                    "target_row_hash": tgt_row.get("__row_hash"),
                }
            )

        return detail

    def _bucket_where_clause(
        self,
        bucket_predicate: Optional[Tuple[str, Any]],
    ) -> str:
        """
        Build a `WHERE {col} = {value}` clause (or `WHERE {col} IS NULL`
        for a null bucket value) scoping a query to exactly one partition
        bucket, or an empty string when no predicate is given (whole-table
        query, today's default behavior). The value is treated as an
        opaque literal from a bucket-fingerprint query's own result - it
        did not come from user input, but is still quoted defensively.
        """
        if bucket_predicate is None:
            return ""
        column, value = bucket_predicate
        ident = self._quote_ident(column)
        if value is None:
            return f"WHERE {ident} IS NULL"
        escaped = str(value).replace("'", "''")
        return f"WHERE CAST({ident} AS STRING) = '{escaped}'"

    def get_row_hashes(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        primary_key_cols: Sequence[str],
        bucket_predicate: Optional[Tuple[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Single push-down query returning one deterministic row hash per
        primary key value(s). `columns` is the fixed, already-sorted list
        of business columns (PK excluded) to hash - callers must pass the
        SAME order for both source and target so the hashes are directly
        comparable. Never pulls row data into pandas beyond the key(s) and
        the resulting hash column.

        Each column is COALESCE(CAST(col AS STRING), sentinel)'d before
        concatenation so NULLs hash consistently and never collapse into
        an empty-string collision with a genuinely empty string value.

        `bucket_predicate`, when given as (column, value), scopes the
        query to exactly one partition bucket (Tier 3) instead of the
        whole table - this is what makes a partitioned Tier 4 cheaper
        than an unpartitioned one.

        Returns a DataFrame with one row per primary key: the key column(s)
        plus `row_hash`.
        """
        if not primary_key_cols:
            raise ValueError("primary_key_cols must be non-empty")

        key_list = ", ".join(self._quote_ident(k) for k in primary_key_cols)

        row_hash_expr = self._row_hash_expr(columns)
        where_clause = self._bucket_where_clause(bucket_predicate)

        query = f"""
            SELECT {key_list}, {row_hash_expr} AS row_hash
            FROM {self._qualify(catalog, schema, table)}
            {where_clause}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute row hashes for '%s.%s.%s'", catalog, schema, table
            )
            raise RuntimeError(
                f"Unable to compute row hashes for '{catalog}.{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(columns=list(primary_key_cols) + ["row_hash"])

        return df

    def get_row_hashes_by_row_number(
        self,
        catalog: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        bucket_predicate: Optional[Tuple[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Fallback for tables with no configured primary key: assigns a
        synthetic row number via ROW_NUMBER() OVER (ORDER BY <every
        requested column>) on both sides, then hashes each row the same
        way get_row_hashes does. Ordering by every column (not insertion
        order, which SQL never guarantees) means two logically-identical
        rows always sort next to each other and get matching numbers
        regardless of physical storage order - but this is NOT a
        substitute for a real key: if the two sides don't contain the
        same *set* of rows, row N on one side is not necessarily the same
        logical record as row N on the other, and comparisons will be
        misleading. Only use when no real shared key exists.

        `bucket_predicate`, when given as (column, value), scopes both the
        ROW_NUMBER() sort and the hash computation to exactly one
        partition bucket (Tier 3) rather than the whole table - row
        numbers are still only comparable within the same bucket on both
        sides, which is exactly the intended scope here.

        Returns a DataFrame with columns: row_number, row_hash.
        """
        if not columns:
            raise ValueError("columns must be non-empty for row-number based hashing")

        order_by = ", ".join(self._quote_ident(c) for c in columns)
        row_hash_expr = self._row_hash_expr(columns)
        where_clause = self._bucket_where_clause(bucket_predicate)

        query = f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {order_by}) AS row_number,
                {row_hash_expr} AS row_hash
            FROM {self._qualify(catalog, schema, table)}
            {where_clause}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute row-number-based hashes for '%s.%s.%s'", catalog, schema, table
            )
            raise RuntimeError(
                f"Unable to compute row-number-based hashes for '{catalog}.{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(columns=["row_number", "row_hash"])

        return df

    def get_row_detail_for_row_numbers(
        self,
        source_catalog: str,
        target_catalog: str,
        schema: str,
        table: str,
        order_by_columns: Sequence[str],
        row_numbers: Sequence[int],
        value_columns: Sequence[str],
        limit_samples: int = 500,
        bucket_predicate: Optional[Tuple[str, Any]] = None,
        target_order_by_columns: Optional[Sequence[str]] = None,
        target_value_columns: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Best-effort Tier 5 column-level diff for the ROW_NUMBER() fallback
        (no primary key configured). Re-executes the SAME
        ROW_NUMBER() OVER (ORDER BY <order_by_columns>) window per side
        used by get_row_hashes_by_row_number, filtered down to the given
        row numbers, then diffs the fetched rows column-by-column.

        `order_by_columns`/`value_columns` are the SOURCE-side spelling;
        `target_order_by_columns`/`target_value_columns` (when a
        column_map applies), the positionally-aligned TARGET-side
        spelling for the same columns - each MUST be the exact same
        per-side column list (same order) passed to
        get_row_hashes_by_row_number for this table, which is what keeps
        row numbers consistent between the hash-computation pass (Tier 4)
        and this re-fetch (Tier 5). When omitted, the target lists default
        to the source lists (today's behavior, unchanged).

        This is inherently best-effort, not a substitute for a real key:
        "row N" on the source and target are only the same logical record
        if both sides otherwise contain the same row set in the same
        relative order. Callers must mark any result derived from this
        method as unverified (see RowMismatchDetail.verified).

        A window-function output column (ROW_NUMBER() here) can only be
        filtered in a query that reads it as a plain column from a
        subquery - it cannot be filtered in the same SELECT that computes
        it via OVER(). Hence the subquery/CTE shape below, rather than a
        flat SELECT ... WHERE row_number IN (...).

        Returns the same shape as _changed_row_detail: one dict per
        row with "key" (here always {"row_number": N}),
        "mismatched_columns", "source_values", "target_values",
        "source_row_hash", "target_row_hash" - all keyed/labeled by the
        canonical (target) column name.
        """
        if not row_numbers:
            return []

        target_order_by_columns = list(target_order_by_columns or order_by_columns)
        target_value_columns = list(target_value_columns or value_columns)
        source_to_target = dict(zip(value_columns, target_value_columns))

        src = self._qualify(source_catalog, schema, table)
        tgt = self._qualify(target_catalog, schema, table)
        source_order_by = ", ".join(self._quote_ident(c) for c in order_by_columns)
        target_order_by = ", ".join(self._quote_ident(c) for c in target_order_by_columns)
        source_value_idents = [self._quote_ident(c) for c in value_columns]
        target_value_idents = [self._quote_ident(c) for c in target_value_columns]
        source_select_list = ", ".join(source_value_idents)
        target_select_list = ", ".join(target_value_idents)
        where_clause = self._bucket_where_clause(bucket_predicate)
        source_row_hash_expr = self._row_hash_expr(value_columns)
        target_row_hash_expr = self._row_hash_expr(target_value_columns)

        # row_numbers are Python ints derived from our own prior
        # ROW_NUMBER() output (never user input) - safe to inline.
        unique_row_numbers = sorted(set(int(n) for n in row_numbers))[: int(limit_samples)]
        row_numbers_csv = ", ".join(str(n) for n in unique_row_numbers)

        def _numbered_query(fqtn: str, order_by: str, select_list: str, row_hash_expr: str) -> str:
            return f"""
                SELECT row_number, {select_list}, {row_hash_expr} AS __row_hash
                FROM (
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY {order_by}) AS row_number,
                        {select_list}
                    FROM {fqtn}
                    {where_clause}
                ) numbered
                WHERE row_number IN ({row_numbers_csv})
            """

        try:
            source_rows = self._execute_to_dataframe(
                _numbered_query(src, source_order_by, source_select_list, source_row_hash_expr)
            ).to_dict(orient="records")
            target_rows = self._execute_to_dataframe(
                _numbered_query(tgt, target_order_by, target_select_list, target_row_hash_expr)
            ).to_dict(orient="records")
        except Exception as exc:
            logger.exception(
                "Failed to fetch row-number-based row detail for '%s.%s'", schema, table
            )
            raise RuntimeError(
                f"Unable to fetch row-number-based row detail for '{schema}.{table}': {exc}"
            ) from exc

        target_by_row_number = {r["row_number"]: r for r in target_rows}

        detail: List[Dict[str, Any]] = []
        for src_row in source_rows:
            tgt_row = target_by_row_number.get(src_row["row_number"])
            if tgt_row is None:
                continue

            mismatched_columns = [
                source_to_target[src_col] for src_col in value_columns
                if values_differ(src_row.get(src_col), tgt_row.get(source_to_target[src_col]))
            ]
            if not mismatched_columns:
                mismatched_columns = list(target_value_columns)

            detail.append(
                {
                    "key": {"row_number": src_row["row_number"]},
                    "mismatched_columns": mismatched_columns,
                    "source_values": {
                        source_to_target[src_col]: src_row.get(src_col)
                        for src_col in value_columns
                        if source_to_target[src_col] in mismatched_columns
                    },
                    "target_values": {
                        tgt_col: tgt_row.get(tgt_col)
                        for tgt_col in mismatched_columns
                    },
                    "source_row_hash": src_row.get("__row_hash"),
                    "target_row_hash": tgt_row.get("__row_hash"),
                }
            )

        return detail

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> "DatabricksConnector":

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:

        self.disconnect()