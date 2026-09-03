"""
Databricks SQL Warehouse Connector

Responsible solely for establishing connectivity to a Databricks SQL Warehouse
and retrieving data / schema information.

Contains no comparison logic. Every method here answers a factual question
("does this catalog exist", "what are the null counts for these columns")
- it never decides PASS/FAIL. That decision lives in validators/catalog_validator.py.

All SQL-building/result-post-processing logic lives on the shared
BaseSqlConnector this class subclasses (see connectors/base_sql_connector.py)
- this file itself only owns the Thrift/SQL-Warehouse connection lifecycle
(__init__/connect/disconnect/test_connection/_ensure_connected) and the one
abstract seam BaseSqlConnector declares, _execute_to_dataframe, implemented
here via a databricks-sql-connector cursor. See connectors/spark_connector.py
for the sibling connector that implements the same BaseSqlConnector interface
via an ambient notebook SparkSession instead - both share 100% of the query-
building code with zero duplication.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection

from table_validator.connectors.base_sql_connector import (  # noqa: F401
    BaseSqlConnector,
    values_differ,
)

logger = logging.getLogger(__name__)


class DatabricksConnector(BaseSqlConnector):
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
        super().__init__()

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
