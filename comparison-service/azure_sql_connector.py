"""
Azure SQL Database Connector

Responsible solely for establishing connectivity to an Azure SQL Database
(via pyodbc / ODBC Driver 17 for SQL Server) and retrieving data / schema
information.

Contains no comparison logic. Every method here answers a factual question
("what tables exist", "what are the null counts for these columns") - it
never decides PASS/FAIL. That decision lives in azure_sql_validator.py,
mirroring the DatabricksConnector / CatalogValidator split.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pyodbc

logger = logging.getLogger(__name__)

# Data types for which MIN/MAX is meaningful. Matched as a prefix against
# SQL Server's INFORMATION_SCHEMA.COLUMNS.DATA_TYPE values.
_MIN_MAX_ELIGIBLE_TYPE_PREFIXES = (
    "tinyint",
    "smallint",
    "int",
    "bigint",
    "float",
    "real",
    "decimal",
    "numeric",
    "money",
    "smallmoney",
    "date",
    "datetime",
    "datetime2",
    "smalldatetime",
)


class AzureSqlConnector:
    """
    Lightweight reusable connector for Azure SQL Database.
    """

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:

        self._server = server or os.getenv("AZURE_SQL_SERVER")
        self._database = database or os.getenv("AZURE_SQL_DATABASE")
        self._username = username or os.getenv("AZURE_SQL_USERNAME")
        self._password = password or os.getenv("AZURE_SQL_PASSWORD")

        if not self._server or not self._database:
            raise ValueError(
                "Azure SQL server and database are required. "
                "Provide them via constructor or environment variables "
                "(AZURE_SQL_SERVER, AZURE_SQL_DATABASE)."
            )

        if not self._username or not self._password:
            raise ValueError(
                "Azure SQL username and password are required. "
                "Provide them via constructor or environment variables "
                "(AZURE_SQL_USERNAME, AZURE_SQL_PASSWORD)."
            )

        self._connection: Optional[pyodbc.Connection] = None

        logger.debug(
            "AzureSqlConnector initialized for server=%s database=%s",
            self._server, self._database,
        )

    # ------------------------------------------------------------------
    # Connection Lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:

        if self._connection is not None:
            return

        try:
            conn_str = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER=tcp:{self._server},1433;"
                f"DATABASE={self._database};"
                f"UID={self._username};"
                f"PWD={self._password};"
                "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
            )

            self._connection = pyodbc.connect(conn_str)

            logger.info(
                "Successfully connected to Azure SQL Database '%s' on '%s'",
                self._database, self._server,
            )

        except Exception as exc:
            self._connection = None
            logger.exception("Failed to connect to Azure SQL Database")
            raise ConnectionError(
                f"Unable to connect to Azure SQL Database: {exc}"
            ) from exc

    def disconnect(self) -> None:

        if self._connection is None:
            return

        try:
            self._connection.close()
            logger.info("Disconnected from Azure SQL Database")
        except Exception as exc:
            logger.warning("Error while closing Azure SQL connection: %s", exc)
        finally:
            self._connection = None

    def test_connection(self) -> bool:
        try:
            self.connect()
            self._execute_to_dataframe("SELECT 1 AS ok")
            logger.info("Azure SQL connection test succeeded")
            return True
        except Exception as exc:
            logger.error("Azure SQL connection test failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _ensure_connected(self) -> pyodbc.Connection:
        if self._connection is None:
            self.connect()
        if self._connection is None:
            raise ConnectionError("Azure SQL connection is not available")
        return self._connection

    def _execute_to_dataframe(self, query: str) -> pd.DataFrame:
        connection = self._ensure_connected()
        try:
            cursor = connection.cursor()
            cursor.execute(query)

            if cursor.description is None:
                cursor.close()
                return pd.DataFrame()

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            return pd.DataFrame((tuple(r) for r in rows), columns=columns)

        except Exception as exc:
            logger.exception("Failed to execute query against Azure SQL Database")
            raise RuntimeError(f"Unable to execute query: {exc}") from exc

    @staticmethod
    def _quote_ident(identifier: str) -> str:
        """Bracket-quote a single identifier part, escaping embedded brackets."""
        escaped = identifier.replace("]", "]]")
        return f"[{escaped}]"

    @classmethod
    def _qualify(cls, schema: str, table: str) -> str:
        """Build a schema-qualified, bracket-quoted [schema].[table] identifier."""
        return f"{cls._quote_ident(schema)}.{cls._quote_ident(table)}"

    # ------------------------------------------------------------------
    # Generic passthrough
    # ------------------------------------------------------------------
    def execute_query(self, query: str) -> pd.DataFrame:
        """Public entry point for executing an arbitrary read-only query."""
        return self._execute_to_dataframe(query)

    # ------------------------------------------------------------------
    # Schema / table metadata
    # ------------------------------------------------------------------
    def get_schemas(self) -> List[str]:
        query = """
            SELECT SCHEMA_NAME
            FROM INFORMATION_SCHEMA.SCHEMATA
            WHERE SCHEMA_NAME NOT IN (
                'sys', 'INFORMATION_SCHEMA', 'guest', 'db_owner',
                'db_accessadmin', 'db_securityadmin', 'db_ddladmin',
                'db_backupoperator', 'db_datareader', 'db_datawriter',
                'db_denydatareader', 'db_denydatawriter'
            )
        """
        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception("Failed to list schemas")
            raise RuntimeError(f"Unable to list schemas: {exc}") from exc

        if df.empty:
            return []
        return sorted(str(v) for v in df["SCHEMA_NAME"])

    def get_tables(self, schema: str) -> List[str]:
        query = f"""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_TYPE = 'BASE TABLE'
        """
        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception("Failed to list tables for schema '%s'", schema)
            raise RuntimeError(f"Unable to list tables for schema '{schema}': {exc}") from exc

        if df.empty:
            return []
        return sorted(str(v) for v in df["TABLE_NAME"])

    def get_table_schema(self, schema: str, table: str) -> pd.DataFrame:
        """
        Returns columns: column_name, data_type, is_nullable (bool),
        ordinal_position.
        """
        query = f"""
            SELECT COLUMN_NAME AS column_name,
                   DATA_TYPE AS data_type,
                   IS_NULLABLE AS is_nullable,
                   ORDINAL_POSITION AS ordinal_position
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
        """
        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s'", schema, table
            )
            raise RuntimeError(
                f"Unable to retrieve column metadata for '{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(
                columns=["column_name", "data_type", "is_nullable", "ordinal_position"]
            )

        df["is_nullable"] = df["is_nullable"].astype(str).str.upper().eq("YES")
        return df.reset_index(drop=True)

    def get_row_count(self, schema: str, table: str) -> int:
        query = f"SELECT COUNT(*) AS row_count FROM {self._qualify(schema, table)}"
        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception("Failed to get row count for '%s.%s'", schema, table)
            raise RuntimeError(f"Unable to get row count for '{schema}.{table}': {exc}") from exc

        if df.empty:
            return 0
        return int(df.iloc[0]["row_count"])

    def get_column_statistics(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        min_max_columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Single aggregate query returning null count, distinct count, and
        (for min_max_columns) MIN/MAX for every requested column.

        Returns: {column_name: {"null_count": int, "distinct_count": int,
                                 "min": Any | None, "max": Any | None}}
        """
        if not columns:
            return {}

        min_max_set = {c.lower() for c in (min_max_columns or [])}

        select_parts = []
        for col in columns:
            q = self._quote_ident(col)
            alias_nulls = self._quote_ident(f"{col}__nulls")
            alias_distinct = self._quote_ident(f"{col}__distinct")
            select_parts.append(f"SUM(CASE WHEN {q} IS NULL THEN 1 ELSE 0 END) AS {alias_nulls}")
            select_parts.append(f"COUNT(DISTINCT {q}) AS {alias_distinct}")
            if col.lower() in min_max_set:
                alias_min = self._quote_ident(f"{col}__min")
                alias_max = self._quote_ident(f"{col}__max")
                select_parts.append(f"MIN({q}) AS {alias_min}")
                select_parts.append(f"MAX({q}) AS {alias_max}")

        query = f"SELECT {', '.join(select_parts)} FROM {self._qualify(schema, table)}"

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s.%s'", schema, table
            )
            raise RuntimeError(
                f"Unable to compute column statistics for '{schema}.{table}': {exc}"
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

    # ------------------------------------------------------------------
    # Row-hash comparison (pushed down via HASHBYTES)
    # ------------------------------------------------------------------
    def get_row_hashes(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        primary_key_cols: Sequence[str],
        column_types: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Single push-down query returning one deterministic row hash per
        primary key value(s), via T-SQL HASHBYTES('SHA2_256', ...).

        `column_types` maps each column name (case-insensitive) to its
        declared SQL Server data type (e.g. from get_table_schema), used
        to format the hash-input string identically to Databricks' own
        CAST(col AS STRING) convention for the equivalent value:
          - float/real -> CAST to FLOAT then formatted so whole numbers
            keep a trailing ".0" (matches Databricks' `double` string
            form; verified empirically, see azure_sql_validator.py).
          - decimal/numeric/money/smallmoney -> CONVERT(VARCHAR, x, 2)
            (fixed, non-scientific decimal notation).
          - the synthetic type "decimal_as_integer" -> caller override for
            when the TARGET column is an integer type even though this
            side is decimal/numeric/money: formats without decimal places
            so equal whole values hash identically instead of every row
            differing purely from the type mismatch (see
            AzureSqlValidator._effective_source_types).
          - date -> CONVERT(VARCHAR, x, 23) ('YYYY-MM-DD').
          - datetime/datetime2/smalldatetime -> CONVERT(VARCHAR, x, 126)
            (ISO 8601).
          - everything else (including when the type is unknown) ->
            CAST(col AS NVARCHAR(MAX)).

        Returns a DataFrame with one row per primary key: the key
        column(s) plus `row_hash` (lowercase hex string, to match
        Databricks' sha2() output format).
        """
        if not primary_key_cols:
            raise ValueError("primary_key_cols must be non-empty")

        key_list = ", ".join(self._quote_ident(k) for k in primary_key_cols)
        null_sentinel = "\x01NULL\x01"
        types_lower = {k.lower(): v for k, v in (column_types or {}).items()}

        hashed_exprs = [
            f"ISNULL({self._hash_string_expr(c, types_lower.get(c.lower(), ''))}, '{null_sentinel}')"
            for c in columns
        ]

        if hashed_exprs:
            concat_expr = " + '||' + ".join(hashed_exprs)
            row_hash_expr = (
                f"LOWER(CONVERT(VARCHAR(64), "
                f"HASHBYTES('SHA2_256', {concat_expr}), 2))"
            )
        else:
            row_hash_expr = (
                f"LOWER(CONVERT(VARCHAR(64), "
                f"HASHBYTES('SHA2_256', '{null_sentinel}'), 2))"
            )

        query = f"""
            SELECT {key_list}, {row_hash_expr} AS row_hash
            FROM {self._qualify(schema, table)}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception("Failed to compute row hashes for '%s.%s'", schema, table)
            raise RuntimeError(
                f"Unable to compute row hashes for '{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(columns=list(primary_key_cols) + ["row_hash"])

        return df

    def get_row_hashes_by_row_number(
        self,
        schema: str,
        table: str,
        columns: Sequence[str],
        column_types: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Fallback for tables with no configured primary key: assigns a
        synthetic row number via ROW_NUMBER() OVER (ORDER BY <every
        requested column>) and hashes each row, mirroring
        DatabricksConnector.get_row_hashes_by_row_number so both sides can
        be compared the same way when no real shared key exists. See that
        method's docstring for the caveat about what row-number matching
        can and cannot detect.

        Returns a DataFrame with columns: row_number, row_hash.
        """
        if not columns:
            raise ValueError("columns must be non-empty for row-number based hashing")

        null_sentinel = "\x01NULL\x01"
        types_lower = {k.lower(): v for k, v in (column_types or {}).items()}

        hashed_exprs = [
            f"ISNULL({self._hash_string_expr(c, types_lower.get(c.lower(), ''))}, '{null_sentinel}')"
            for c in columns
        ]
        order_by = ", ".join(self._quote_ident(c) for c in columns)
        concat_expr = " + '||' + ".join(hashed_exprs)
        row_hash_expr = (
            f"LOWER(CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', {concat_expr}), 2))"
        )

        query = f"""
            SELECT
                ROW_NUMBER() OVER (ORDER BY {order_by}) AS row_number,
                {row_hash_expr} AS row_hash
            FROM {self._qualify(schema, table)}
        """

        try:
            df = self._execute_to_dataframe(query)
        except Exception as exc:
            logger.exception(
                "Failed to compute row-number-based hashes for '%s.%s'", schema, table
            )
            raise RuntimeError(
                f"Unable to compute row-number-based hashes for '{schema}.{table}': {exc}"
            ) from exc

        if df.empty:
            return pd.DataFrame(columns=["row_number", "row_hash"])

        return df

    def _hash_string_expr(self, column: str, data_type: str) -> str:
        """
        Build the SQL expression that converts one column to its hash-input
        string form, keyed off the column's declared SQL Server type so it
        matches Databricks' CAST(col AS STRING) output for the equivalent
        value (see get_row_hashes docstring for the per-type rules).
        """
        q = self._quote_ident(column)
        dt = (data_type or "").strip().lower()

        if dt in ("float", "real"):
            # SQL Server's default CAST(float AS NVARCHAR) can use
            # scientific notation and doesn't guarantee a trailing ".0"
            # for whole numbers the way Databricks' double->string does.
            # STR(x, 30, 10) then trimming trailing zeros (keeping at
            # least one decimal digit) reproduces that convention.
            return (
                f"CASE WHEN {q} IS NULL THEN NULL ELSE "
                f"CASE WHEN {q} = ROUND({q}, 0) THEN "
                f"CONVERT(VARCHAR(30), CAST({q} AS BIGINT)) + '.0' "
                f"ELSE LTRIM(RTRIM(STR({q}, 30, 10))) END END"
            )

        if dt == "decimal_as_integer":
            # Caller-requested override: the target column is an integer
            # type even though this side is decimal/numeric/money - format
            # without decimal places (dropping a fractional remainder, if
            # any) so numerically-equal whole values hash identically
            # instead of every row appearing changed purely due to the
            # type mismatch. A genuinely fractional value here means a
            # real precision loss versus the integer target, which still
            # surfaces correctly since the fraction is truncated on both
            # sides' comparison via values_differ() at the detail stage.
            return f"CONVERT(VARCHAR(50), CAST({q} AS BIGINT))"

        if dt in ("decimal", "numeric", "money", "smallmoney"):
            return f"CONVERT(VARCHAR(50), {q}, 2)"

        if dt == "date":
            return f"CONVERT(VARCHAR(10), {q}, 23)"

        if dt in ("datetime", "datetime2", "smalldatetime"):
            return f"CONVERT(VARCHAR(33), {q}, 126)"

        # VARCHAR, not NVARCHAR: HASHBYTES hashes raw bytes, and
        # NVARCHAR is UTF-16 (2 bytes/char) while Databricks' CAST(col AS
        # STRING) is UTF-8 (1 byte/char for ASCII) - identical text would
        # otherwise hash completely differently between the two sides
        # (verified empirically: same input string, NVARCHAR cast gave a
        # column twice the byte length of the matching VARCHAR/UTF-8 form,
        # and a different hash). ASCII-only: VARCHAR uses a single-byte
        # codepage, so genuine non-ASCII characters (accents, non-Latin
        # scripts) will NOT hash-match Databricks' UTF-8 form under this
        # cast - re-verify before trusting row-hash results on Unicode text.
        return f"CAST({q} AS VARCHAR(MAX))"

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> "AzureSqlConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
