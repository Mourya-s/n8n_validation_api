"""
SparkConnector: notebook-native connector for Databricks, using the
notebook's own ambient SparkSession instead of a Thrift/HTTP SQL Warehouse
connection. Requires zero separate auth config (no workspace URL, no
personal access token, no SQL Warehouse HTTP path) - it inherits ambient
auth from the calling notebook's own cluster session.

Implements the exact same BaseSqlConnector interface DatabricksConnector
does (see connectors/base_sql_connector.py) - every query-building method
is shared, unchanged, between the two; the only difference is how a query
string actually gets executed: DatabricksConnector opens a real SQL
Warehouse connection via databricks-sql-connector, SparkConnector runs
spark.sql(query).toPandas() against a session that already exists.

pyspark is NOT a hard dependency of this package - it is only imported
here, lazily, inside __init__, and only when the caller doesn't already
pass an active SparkSession. This keeps `import table_validator` (and this
module itself) safe for CLI-only users who never install pyspark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from table_validator.connectors.base_sql_connector import BaseSqlConnector

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


class SparkConnector(BaseSqlConnector):
    """
    Connector that runs every query via spark.sql(...).toPandas() against
    an ambient SparkSession - no separate connection or authentication is
    ever established, since the notebook's own cluster session is reused
    as-is.
    """

    def __init__(self, spark: Optional["SparkSession"] = None) -> None:
        """
        spark defaults to SparkSession.getActiveSession() - the standard
        way any Databricks notebook cell's own `spark` global is picked up
        with zero imports/config beyond pyspark itself, which every
        Databricks notebook runtime already provides. Pass spark=
        explicitly to use a different/specific session (e.g. in tests, or
        outside a notebook with pyspark installed as the optional `spark`
        extra).
        """
        if spark is None:
            # Deferred import: pyspark must never be a hard dependency for
            # CLI-only users - only constructing a SparkConnector with no
            # explicit session requires it to be installed.
            from pyspark.sql import SparkSession as _SparkSession

            spark = _SparkSession.getActiveSession()

        if spark is None:
            raise RuntimeError(
                "No active SparkSession found. SparkConnector must be "
                "constructed inside a Databricks notebook (where a "
                "SparkSession is already active), or pass spark= explicitly."
            )

        self._spark = spark

    # ------------------------------------------------------------------
    # Connection Lifecycle - no-ops: the ambient SparkSession's lifecycle
    # is owned by the notebook/cluster, never by this connector.
    # ------------------------------------------------------------------
    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def test_connection(self) -> bool:
        try:
            self._execute_to_dataframe("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # The one abstract seam BaseSqlConnector declares.
    # ------------------------------------------------------------------
    def _execute_to_dataframe(self, query: str) -> pd.DataFrame:
        return self._spark.sql(query).toPandas()
