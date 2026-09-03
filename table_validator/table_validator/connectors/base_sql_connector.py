"""
BaseSqlConnector: shared SQL-building logic for every "talks Spark SQL to
a Databricks-compatible engine" connector.

Every method here builds a SQL string and calls self._execute_to_dataframe
(the one abstract seam) to run it - none of them touch a connection object
directly. This is what lets DatabricksConnector (a real SQL Warehouse
connection via databricks-sql-connector) and SparkConnector (an ambient
notebook SparkSession, via spark.sql(...).toPandas()) share 100% of this
query-building/result-post-processing code with zero duplication - only
_execute_to_dataframe (and the connection-lifecycle methods connect/
disconnect/test_connection, which stay on each concrete subclass) differs
between the two.

Contains no comparison logic. Every method here answers a factual question
("does this catalog exist", "what are the null counts for these columns")
- it never decides PASS/FAIL. That decision lives in validators/catalog_validator.py.
"""

from __future__ import annotations

import datetime
import logging
import numbers
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

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


class BaseSqlConnector:
    """
    Shared SQL-building logic for Databricks-compatible connectors.
    Concrete subclasses (DatabricksConnector, SparkConnector) implement
    only connection lifecycle (__init__/connect/disconnect/
    test_connection) and _execute_to_dataframe - every method below is
    connection-agnostic and works unchanged regardless of how the query
    text actually gets executed. Subclasses must call
    super().__init__() so the row-filter state below exists.
    """

    def __init__(self) -> None:
        # Row-filter predicates (raw SQL WHERE-fragment strings), set via
        # set_row_filters() - notebook.py's validate_tables() is the only
        # current caller (row_filter/source_row_filter/target_row_filter
        # kwargs). None (the default for every field) means "no filter,
        # whole table" - the CLI path never calls set_row_filters, so
        # every existing caller's behavior is completely unchanged.
        self._common_row_filter: Optional[str] = None
        self._source_row_filter: Optional[str] = None
        self._target_row_filter: Optional[str] = None

    def set_row_filters(
        self,
        *,
        common: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        """
        Restrict every subsequent row-level query on this connector
        (row count, statistics, fingerprint, row-hash, column-level
        detail) to rows matching these SQL WHERE-fragment predicates -
        `common` applies to both sides, `source`/`target` apply
        additionally (ANDed) to just that side, so all three can combine
        (e.g. common="status = 'active'" plus source="id > 20" means the
        source side gets both conditions ANDed, the target side gets
        only the common one).

        Each fragment is used as-is, parenthesized for safe AND-
        combination when more than one applies - not parsed or
        validated in any way; a malformed fragment surfaces as a normal
        SQL error from the underlying engine the first time a query
        actually runs, the same way a typo in any other user-supplied
        SQL text would. This is a notebook-facing convenience for a
        trusted caller filtering their own comparison, not a public API
        boundary accepting untrusted input.
        """
        self._common_row_filter = common
        self._source_row_filter = source
        self._target_row_filter = target

    def _scoped_table(
        self,
        catalog: str,
        schema: str,
        table: str,
        side: Optional[str] = None,
    ) -> str:
        """
        Like _qualify(catalog, schema, table), but wraps the result in a
        filtered subquery when a row filter applies for `side`
        ("source"/"target"/None - None means "no per-side filter, only
        the common one if set"). Every existing row-level query already
        obtains its FROM-clause table reference through exactly one call
        like this per side, so swapping that one call site is enough to
        make the whole tiered comparison filter-aware with zero changes
        to catalog_validator.py's own tier logic.

        Returns _qualify's own output completely unchanged when no
        filter applies for this call (both `_common_row_filter` and the
        relevant per-side filter are None) - this is what keeps every
        existing caller's generated SQL byte-identical to before this
        feature existed.
        """
        qualified = self._qualify(catalog, schema, table)

        conditions = []
        if self._common_row_filter:
            conditions.append(f"({self._common_row_filter})")
        if side == "source" and self._source_row_filter:
            conditions.append(f"({self._source_row_filter})")
        elif side == "target" and self._target_row_filter:
            conditions.append(f"({self._target_row_filter})")

        if not conditions:
            return qualified

        return f"(SELECT * FROM {qualified} WHERE {' AND '.join(conditions)}) AS filtered"

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
    # Generic passthrough (public alias, used by CatalogValidator
    # for anything not covered by a dedicated method below)
    # ------------------------------------------------------------------
    def execute_query(self, query: str) -> pd.DataFrame:
        """Public entry point for executing an arbitrary read-only query."""
        return self._execute_to_dataframe(query)

    # ------------------------------------------------------------------
    # Catalog / schema / table metadata methods
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

    def get_row_count(
        self, catalog: str, schema: str, table: str, side: Optional[str] = None,
    ) -> int:
        query = f"SELECT COUNT(*) AS row_count FROM {self._scoped_table(catalog, schema, table, side)}"
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
        side: Optional[str] = None,
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
            f"FROM {self._scoped_table(catalog, schema, table, side)}"
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

        def _to_int_or_none(value: Any) -> Optional[int]:
            """
            SUM/COUNT DISTINCT over ZERO rows (e.g. a row_filter that
            legitimately matches nothing on this side) still returns
            exactly one row from this aggregate query - but the
            aggregate value itself is SQL NULL, which a pandas/Arrow
            round-trip (spark.sql(...).toPandas()) represents as float
            NaN, not Python None. `value is not None` alone doesn't
            catch that - NaN is its own object, not None - so int(NaN)
            was reaching int() and raising ValueError. isinstance(value,
            float) and value != value is the standard NaN check (NaN is
            the only float that isn't equal to itself); pd.isna() would
            also work but this avoids adding a pandas call to a function
            that otherwise doesn't need one.
            """
            if value is None:
                return None
            if isinstance(value, float) and value != value:
                return None
            return int(value)

        for col in columns:
            entry: Dict[str, Any] = {
                "null_count": _to_int_or_none(row.get(f"{col}__nulls")),
                "distinct_count": _to_int_or_none(row.get(f"{col}__distinct")),
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
        side: Optional[str] = None,
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
            FROM {self._scoped_table(catalog, schema, table, side)}
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
        side: Optional[str] = None,
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
            FROM {self._scoped_table(catalog, schema, table, side)}
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
        source_schema: Optional[str] = None,
        source_table: Optional[str] = None,
        target_schema: Optional[str] = None,
        target_table: Optional[str] = None,
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

        `schema`/`table` are the legacy shared-name params, used for both
        sides when the newer, independent `source_schema`/`source_table`/
        `target_schema`/`target_table` are left unset - this is what kept
        this call working correctly for a schema_map/table_map pair (a
        genuinely differently-named source/target table), which previously
        always queried the source catalog for the TARGET-side name and
        failed with TABLE_OR_VIEW_NOT_FOUND whenever the two names
        actually differed.
        """
        if not key_values:
            return []

        src = self._scoped_table(source_catalog, source_schema or schema, source_table or table, "source")
        tgt = self._scoped_table(target_catalog, target_schema or schema, target_table or table, "target")
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
        side: Optional[str] = None,
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
            FROM {self._scoped_table(catalog, schema, table, side)}
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
        side: Optional[str] = None,
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
            FROM {self._scoped_table(catalog, schema, table, side)}
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
        source_schema: Optional[str] = None,
        source_table: Optional[str] = None,
        target_schema: Optional[str] = None,
        target_table: Optional[str] = None,
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

        `schema`/`table` are the legacy shared-name params, used for both
        sides when the newer, independent `source_schema`/`source_table`/
        `target_schema`/`target_table` are left unset. Real bug fixed
        here: previously this method ALWAYS queried both source_catalog
        and target_catalog using the single `table` name - fine when
        source/target share a name, but for a genuinely renamed table
        pair (schema_map/table_map, or the CLI's own source_table.table
        != target_table.table with no primary key configured) it silently
        queried the SOURCE catalog for the TARGET's table name, which
        doesn't exist there, and raised TABLE_OR_VIEW_NOT_FOUND - Tier 5
        would then fail and sample_changed_detail (the Data Mismatches /
        Mismatch Categories sheets' data source) stayed empty even though
        Tier 4 had already confirmed real row-hash mismatches.

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

        src = self._scoped_table(source_catalog, source_schema or schema, source_table or table, "source")
        tgt = self._scoped_table(target_catalog, target_schema or schema, target_table or table, "target")
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
    # Context Manager Support - relies on each concrete subclass's own
    # connect()/disconnect() (for SparkConnector, both are no-ops).
    # ------------------------------------------------------------------
    def __enter__(self) -> "BaseSqlConnector":

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:

        self.disconnect()
