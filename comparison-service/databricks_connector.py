"""
Databricks SQL Warehouse Connector

Responsible solely for establishing connectivity to a Databricks SQL Warehouse
and retrieving data / schema information.

Contains no comparison logic.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection

logger = logging.getLogger(__name__)


class DatabricksConnector:
    """
    Lightweight reusable connector for Databricks SQL Warehouse.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        token: Optional[str] = None,
        http_path: Optional[str] = None,
    ) -> None:

        self._host = host or os.getenv("DATABRICKS_HOST")
        self._token = token or os.getenv("DATABRICKS_TOKEN")
        self._http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")

        if not self._host or not self._token:
            raise ValueError(
                "Databricks host and token are required. "
                "Provide them via constructor or environment variables."
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

            self._connection = sql.connect(
                server_hostname=self._host,
                http_path=self._http_path,
                access_token=self._token,
            )

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

    # ------------------------------------------------------------------
    # Data Retrieval
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