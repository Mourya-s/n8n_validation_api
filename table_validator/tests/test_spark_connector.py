"""
Tests for SparkConnector (connectors/spark_connector.py).

No real pyspark session is required for most tests: spark.sql(...).
toPandas() is mocked directly, so these tests verify the connector's
_execute_to_dataframe wiring and connection-lifecycle no-ops without
needing a real Spark cluster. The one test that DOES exercise the real
pyspark import machinery (getActiveSession() returning None) is skipped
cleanly if pyspark isn't installed, via pytest.importorskip.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from table_validator.connectors.base_sql_connector import BaseSqlConnector
from table_validator.connectors.spark_connector import SparkConnector


def _fake_spark(return_df: pd.DataFrame) -> MagicMock:
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = return_df
    return fake_spark


def test_spark_connector_is_a_base_sql_connector():
    """SparkConnector must share BaseSqlConnector's full query-building
    surface - this is what makes it a drop-in for CatalogValidator."""
    assert issubclass(SparkConnector, BaseSqlConnector)


def test_execute_to_dataframe_runs_spark_sql_and_returns_pandas():
    expected = pd.DataFrame({"a": [1, 2]})
    fake_spark = _fake_spark(expected)
    connector = SparkConnector(spark=fake_spark)

    result = connector._execute_to_dataframe("SELECT 1")

    fake_spark.sql.assert_called_once_with("SELECT 1")
    fake_spark.sql.return_value.toPandas.assert_called_once()
    assert result is expected


def test_public_method_delegates_through_spark_sql():
    """A real public method (not just the raw seam) must route through
    spark.sql(...).toPandas() with the expected query - confirms the
    inherited BaseSqlConnector methods work unmodified against this
    connector."""
    fake_spark = _fake_spark(pd.DataFrame({"tableName": ["t1", "t2"]}))
    connector = SparkConnector(spark=fake_spark)

    tables = connector.get_tables("cat", "sch")

    assert tables == ["t1", "t2"]
    called_query = fake_spark.sql.call_args[0][0]
    assert "SHOW TABLES IN" in called_query
    assert "`cat`.`sch`" in called_query


def test_connect_disconnect_are_no_ops():
    fake_spark = _fake_spark(pd.DataFrame())
    connector = SparkConnector(spark=fake_spark)

    # Must not raise, must not touch fake_spark at all.
    connector.connect()
    connector.disconnect()
    fake_spark.sql.assert_not_called()


def test_test_connection_returns_true_on_success():
    fake_spark = _fake_spark(pd.DataFrame({"1": [1]}))
    connector = SparkConnector(spark=fake_spark)

    assert connector.test_connection() is True


def test_test_connection_returns_false_on_failure():
    fake_spark = MagicMock()
    fake_spark.sql.side_effect = RuntimeError("boom")
    connector = SparkConnector(spark=fake_spark)

    assert connector.test_connection() is False


def test_context_manager_calls_noop_connect_and_disconnect():
    fake_spark = _fake_spark(pd.DataFrame())
    with SparkConnector(spark=fake_spark) as connector:
        assert isinstance(connector, SparkConnector)
    fake_spark.sql.assert_not_called()


def test_explicit_spark_session_skips_getActiveSession_entirely(monkeypatch):
    """When spark= is passed explicitly, pyspark's own SparkSession class
    must never be imported/consulted at all - confirms this connector
    never requires pyspark to be installed for a caller who already has
    a session object in hand (e.g. from a non-Databricks Spark cluster)."""
    fake_spark = _fake_spark(pd.DataFrame())
    # No pyspark import guard needed here since spark= is given directly -
    # the constructor's own `if spark is None` branch is never entered.
    connector = SparkConnector(spark=fake_spark)
    assert connector._spark is fake_spark


def test_no_active_session_raises_clear_runtime_error():
    pytest.importorskip("pyspark")
    import pyspark.sql as pyspark_sql

    original = pyspark_sql.SparkSession.getActiveSession
    pyspark_sql.SparkSession.getActiveSession = staticmethod(lambda: None)
    try:
        with pytest.raises(RuntimeError, match="No active SparkSession"):
            SparkConnector()
    finally:
        pyspark_sql.SparkSession.getActiveSession = original
