"""
Tests for DatabricksConnector (connectors/databricks_connector.py).

No live Databricks connection is required: _execute_to_dataframe is
monkeypatched to capture the generated SQL and return a canned result,
so these tests verify SQL-generation logic and result parsing only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

import table_validator.connectors.databricks_connector as databricks_connector_module
from table_validator.connectors.databricks_connector import DatabricksConnector


def _connector() -> DatabricksConnector:
    return DatabricksConnector(host="h", token="t", http_path="p")


# ---------------------------------------------------------------------------
# _row_hash_expr extraction: get_row_hashes/get_row_hashes_by_row_number
# must generate byte-identical SQL to the pre-refactor inline expression
# for the default (no-op) canonicalization spec.
# ---------------------------------------------------------------------------
def test_get_row_hashes_sql_unchanged_by_row_hash_expr_extraction(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_hashes("cat", "sch", "tbl", ["name", "age"], ["id"])

    query = captured["query"]
    assert "sha2(concat_ws('||'," in query
    assert "COALESCE(CAST(`name` AS STRING), '\x01NULL\x01')" in query
    assert "COALESCE(CAST(`age` AS STRING), '\x01NULL\x01')" in query
    assert "`id`" in query
    assert "FROM `cat`.`sch`.`tbl`" in query


def test_get_row_hashes_by_row_number_sql_unchanged_by_row_hash_expr_extraction(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_hashes_by_row_number("cat", "sch", "tbl", ["name", "age"])

    query = captured["query"]
    assert "ROW_NUMBER() OVER (ORDER BY `name`, `age`)" in query
    assert "sha2(concat_ws('||'," in query


# ---------------------------------------------------------------------------
# get_table_fingerprint (Tier 2): single aggregate query, no row data ever
# fetched beyond one summary row.
# ---------------------------------------------------------------------------
def test_get_table_fingerprint_sql_shape(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame([{"row_count": 100, "hash_sum": 12345, "hash_xor": 999}])

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    result = connector.get_table_fingerprint("cat", "sch", "tbl", ["id", "name"])

    query = captured["query"]
    assert "COUNT(*) AS row_count" in query
    assert "SUM(CAST(conv(substr(" in query
    assert "AS DECIMAL(38,0))) AS hash_sum" in query
    assert "BIT_XOR(CAST(conv(substr(" in query
    assert "AS BIGINT)) AS hash_xor" in query
    assert "substr(sha2(concat_ws('||'," in query
    assert ", 1, 15)" in query
    assert "FROM `cat`.`sch`.`tbl`" in query

    assert result == {"row_count": 100, "hash_sum": 12345, "hash_xor": 999}


def test_get_table_fingerprint_empty_result_returns_zero_defaults(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(connector, "_execute_to_dataframe", lambda q: pd.DataFrame())

    result = connector.get_table_fingerprint("cat", "sch", "tbl", ["id"])

    assert result == {"row_count": 0, "hash_sum": None, "hash_xor": None}


def test_get_table_fingerprint_wraps_exceptions(monkeypatch):
    connector = _connector()

    def raise_exc(query):
        raise RuntimeError("boom")

    monkeypatch.setattr(connector, "_execute_to_dataframe", raise_exc)

    with pytest.raises(RuntimeError, match="Unable to compute table fingerprint"):
        connector.get_table_fingerprint("cat", "sch", "tbl", ["id"])


# ---------------------------------------------------------------------------
# get_table_fingerprint_by_bucket (Tier 3): same fingerprint, GROUP BY a
# chosen bucket column - one row per bucket, not per table row.
# ---------------------------------------------------------------------------
def test_get_table_fingerprint_by_bucket_sql_shape(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame([
            {"bucket_value": "east", "row_count": 50, "hash_sum": 111, "hash_xor": 222},
            {"bucket_value": "west", "row_count": 50, "hash_sum": 333, "hash_xor": 444},
        ])

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    result = connector.get_table_fingerprint_by_bucket(
        "cat", "sch", "tbl", ["id", "name"], bucket_column="region",
    )

    query = captured["query"]
    assert "`region` AS bucket_value" in query
    assert "GROUP BY `region`" in query
    assert "COUNT(*) AS row_count" in query
    assert "SUM(CAST(conv(substr(" in query
    assert "BIT_XOR(CAST(conv(substr(" in query
    assert "FROM `cat`.`sch`.`tbl`" in query

    assert list(result["bucket_value"]) == ["east", "west"]
    assert list(result["row_count"]) == [50, 50]


def test_get_table_fingerprint_by_bucket_empty_result_returns_empty_frame(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(connector, "_execute_to_dataframe", lambda q: pd.DataFrame())

    result = connector.get_table_fingerprint_by_bucket(
        "cat", "sch", "tbl", ["id"], bucket_column="region",
    )

    assert list(result.columns) == ["bucket_value", "row_count", "hash_sum", "hash_xor"]
    assert len(result) == 0


def test_get_table_fingerprint_by_bucket_wraps_exceptions(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(
        connector, "_execute_to_dataframe",
        lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="Unable to compute bucketed table fingerprint"):
        connector.get_table_fingerprint_by_bucket("cat", "sch", "tbl", ["id"], bucket_column="region")


# ---------------------------------------------------------------------------
# bucket_predicate on get_row_hashes / get_row_hashes_by_row_number (Tier 4
# scoped to a single culprit bucket, instead of the whole table).
# ---------------------------------------------------------------------------
def test_get_row_hashes_with_bucket_predicate_adds_where_clause(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_hashes(
        "cat", "sch", "tbl", ["name"], ["id"], bucket_predicate=("region", "east"),
    )

    query = captured["query"]
    assert "WHERE CAST(`region` AS STRING) = 'east'" in query


def test_get_row_hashes_without_bucket_predicate_has_no_where_clause(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_hashes("cat", "sch", "tbl", ["name"], ["id"])

    assert "WHERE" not in captured["query"]


def test_get_row_hashes_by_row_number_with_bucket_predicate_adds_where_clause(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_hashes_by_row_number(
        "cat", "sch", "tbl", ["name"], bucket_predicate=("region", "east"),
    )

    query = captured["query"]
    assert "WHERE CAST(`region` AS STRING) = 'east'" in query


def test_bucket_predicate_null_value_uses_is_null():
    connector = _connector()
    clause = connector._bucket_where_clause(("region", None))
    assert clause == "WHERE `region` IS NULL"


def test_bucket_predicate_escapes_embedded_quote():
    connector = _connector()
    clause = connector._bucket_where_clause(("region", "o'brien"))
    assert clause == "WHERE CAST(`region` AS STRING) = 'o''brien'"


# ---------------------------------------------------------------------------
# get_row_detail_for_keys (Tier 5): bounded fetch for a known key set.
# ---------------------------------------------------------------------------
def test_get_row_detail_for_keys_builds_in_clause_and_returns_diff(monkeypatch):
    connector = _connector()
    queries = []

    def fake_execute(query):
        queries.append(query)
        # The first line is always "SELECT ... FROM <fqtn>" for the side
        # being queried - later lines (the IN-subquery) may reference the
        # *other* side's catalog too, so match on the first FROM only.
        first_from = query.strip().splitlines()[1].strip()
        if first_from.startswith("FROM `cat_source`"):
            return pd.DataFrame([{"id": 2, "name": "old", "__row_hash": 111}])
        if first_from.startswith("FROM `cat_target`"):
            return pd.DataFrame([{"id": 2, "name": "new", "__row_hash": 222}])
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    detail = connector.get_row_detail_for_keys(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        key_column="id",
        key_values=["2"],
        value_columns=["name"],
    )

    assert len(detail) == 1
    assert detail[0]["mismatched_columns"] == ["name"]
    assert detail[0]["source_values"]["name"] == "old"
    assert detail[0]["target_values"]["name"] == "new"
    assert any("IN ('2')" in q for q in queries)


def test_get_row_detail_for_keys_empty_keys_short_circuits(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(
        connector, "_execute_to_dataframe",
        lambda q: (_ for _ in ()).throw(AssertionError("should not query")),
    )

    detail = connector.get_row_detail_for_keys(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        key_column="id",
        key_values=[],
        value_columns=["name"],
    )

    assert detail == []


# ---------------------------------------------------------------------------
# get_row_detail_for_row_numbers (Tier 5, best-effort for the ROW_NUMBER()
# fallback): a windowed subquery per side, filtered by row_number IN (...),
# never a flat WHERE against the base table (ROW_NUMBER() isn't a real
# column there).
# ---------------------------------------------------------------------------
def test_get_row_detail_for_row_numbers_builds_windowed_query_and_returns_diff(monkeypatch):
    connector = _connector()
    queries = []

    def fake_execute(query):
        queries.append(query)
        # Find the line naming the base table being scanned (not the
        # `FROM (` subquery wrapper line above it).
        from_line = next(
            line.strip() for line in query.strip().splitlines()
            if line.strip().startswith("FROM `")
        )
        if from_line.startswith("FROM `cat_source`"):
            return pd.DataFrame([{"row_number": 2, "id": 2, "name": "old", "__row_hash": 111}])
        if from_line.startswith("FROM `cat_target`"):
            return pd.DataFrame([{"row_number": 2, "id": 2, "name": "new", "__row_hash": 222}])
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    detail = connector.get_row_detail_for_row_numbers(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        order_by_columns=["id", "name"],
        row_numbers=[2],
        value_columns=["id", "name"],
    )

    assert len(detail) == 1
    assert detail[0]["key"] == {"row_number": 2}
    assert detail[0]["mismatched_columns"] == ["name"]
    assert detail[0]["source_values"]["name"] == "old"
    assert detail[0]["target_values"]["name"] == "new"

    for query in queries:
        assert "ROW_NUMBER() OVER (ORDER BY" in query
        assert "WHERE row_number IN (2)" in query
        # Never the physical-column IN-clause shape used by the real-key
        # Tier 5 path - ROW_NUMBER() isn't a base-table column.
        assert "CAST(`row_number` AS STRING) IN" not in query


def test_get_row_detail_for_row_numbers_empty_row_numbers_short_circuits(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(
        connector, "_execute_to_dataframe",
        lambda q: (_ for _ in ()).throw(AssertionError("should not query")),
    )

    detail = connector.get_row_detail_for_row_numbers(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        order_by_columns=["id"],
        row_numbers=[],
        value_columns=["id"],
    )

    assert detail == []


def test_get_row_detail_for_row_numbers_with_bucket_predicate_adds_where_clause(monkeypatch):
    connector = _connector()
    captured = {}

    def fake_execute(query):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    connector.get_row_detail_for_row_numbers(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        order_by_columns=["id"],
        row_numbers=[2],
        value_columns=["id"],
        bucket_predicate=("region", "east"),
    )

    assert "WHERE CAST(`region` AS STRING) = 'east'" in captured["query"]


def test_get_row_detail_for_row_numbers_wraps_exceptions(monkeypatch):
    connector = _connector()
    monkeypatch.setattr(
        connector, "_execute_to_dataframe",
        lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="Unable to fetch row-number-based row detail"):
        connector.get_row_detail_for_row_numbers(
            source_catalog="cat_source",
            target_catalog="cat_target",
            schema="sch",
            table="tbl",
            order_by_columns=["id"],
            row_numbers=[2],
            value_columns=["id"],
        )


# ---------------------------------------------------------------------------
# connect(): retry timeout override for slow/unstable CloudFetch downloads.
# ---------------------------------------------------------------------------
def test_connect_omits_retry_kwarg_by_default(monkeypatch):
    """Without an explicit retry_timeout_seconds or
    DATABRICKS_RETRY_TIMEOUT_SECONDS, connect() must not pass
    _retry_stop_after_attempts_duration at all, preserving
    databricks-sql-connector's own default (300s)."""
    monkeypatch.delenv("DATABRICKS_RETRY_TIMEOUT_SECONDS", raising=False)
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(databricks_connector_module.sql, "connect", fake_connect)
    connector = _connector()

    connector.connect()

    assert "_retry_stop_after_attempts_duration" not in captured


def test_connect_passes_explicit_retry_timeout_seconds(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RETRY_TIMEOUT_SECONDS", raising=False)
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(databricks_connector_module.sql, "connect", fake_connect)
    connector = DatabricksConnector(
        host="h", token="t", http_path="p", retry_timeout_seconds=900.0
    )

    connector.connect()

    assert captured["_retry_stop_after_attempts_duration"] == 900.0


def test_connect_falls_back_to_retry_timeout_env_var(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RETRY_TIMEOUT_SECONDS", "1200")
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(databricks_connector_module.sql, "connect", fake_connect)
    connector = _connector()

    connector.connect()

    assert captured["_retry_stop_after_attempts_duration"] == 1200.0


def test_connect_explicit_arg_takes_priority_over_env_var(monkeypatch):
    monkeypatch.setenv("DATABRICKS_RETRY_TIMEOUT_SECONDS", "1200")
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(databricks_connector_module.sql, "connect", fake_connect)
    connector = DatabricksConnector(
        host="h", token="t", http_path="p", retry_timeout_seconds=600.0
    )

    connector.connect()

    assert captured["_retry_stop_after_attempts_duration"] == 600.0


# ---------------------------------------------------------------------------
# column_map (Phase 5): _changed_row_detail / get_row_detail_for_keys /
# get_row_detail_for_row_numbers each side using its own column spelling,
# reconciled back to the canonical (target) name.
# ---------------------------------------------------------------------------
def test_get_row_detail_for_keys_uses_target_value_columns_for_target_query(monkeypatch):
    connector = _connector()
    queries = []

    def fake_execute(query):
        queries.append(query)
        first_from = query.strip().splitlines()[1].strip()
        if first_from.startswith("FROM `cat_source`"):
            return pd.DataFrame([{"id": 2, "cust_id": "111", "__row_hash": 111}])
        if first_from.startswith("FROM `cat_target`"):
            return pd.DataFrame([{"id": 2, "customer_id": "222", "__row_hash": 222}])
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    detail = connector.get_row_detail_for_keys(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        key_column="id",
        key_values=["2"],
        value_columns=["cust_id"],
        target_value_columns=["customer_id"],
    )

    assert len(detail) == 1
    # Reconciled to the canonical (target) name.
    assert detail[0]["mismatched_columns"] == ["customer_id"]
    assert detail[0]["source_values"]["customer_id"] == "111"
    assert detail[0]["target_values"]["customer_id"] == "222"

    # The source-side query text uses the source spelling; target-side
    # query text uses the target spelling. Match on the first FROM line
    # (a later line's IN-subquery always references the source, by
    # design - that subquery is just resolving which keys changed).
    def _first_from(q):
        return next(
            line.strip() for line in q.strip().splitlines()
            if line.strip().startswith("FROM `")
        )

    source_query = next(q for q in queries if _first_from(q).startswith("FROM `cat_source`"))
    target_query = next(q for q in queries if _first_from(q).startswith("FROM `cat_target`"))
    assert "`cust_id`" in source_query
    assert "`customer_id`" not in source_query
    assert "`customer_id`" in target_query
    assert "`cust_id`" not in target_query


def test_get_row_detail_for_row_numbers_uses_target_columns_for_target_query(monkeypatch):
    connector = _connector()
    queries = []

    def fake_execute(query):
        queries.append(query)
        from_line = next(
            line.strip() for line in query.strip().splitlines()
            if line.strip().startswith("FROM `")
        )
        if from_line.startswith("FROM `cat_source`"):
            return pd.DataFrame([{"row_number": 2, "cust_id": "111", "__row_hash": 111}])
        if from_line.startswith("FROM `cat_target`"):
            return pd.DataFrame([{"row_number": 2, "customer_id": "222", "__row_hash": 222}])
        return pd.DataFrame()

    monkeypatch.setattr(connector, "_execute_to_dataframe", fake_execute)

    detail = connector.get_row_detail_for_row_numbers(
        source_catalog="cat_source",
        target_catalog="cat_target",
        schema="sch",
        table="tbl",
        order_by_columns=["cust_id"],
        row_numbers=[2],
        value_columns=["cust_id"],
        target_order_by_columns=["customer_id"],
        target_value_columns=["customer_id"],
    )

    assert len(detail) == 1
    assert detail[0]["mismatched_columns"] == ["customer_id"]
    assert detail[0]["source_values"]["customer_id"] == "111"
    assert detail[0]["target_values"]["customer_id"] == "222"

    source_query = next(q for q in queries if "`cat_source`" in q)
    target_query = next(q for q in queries if "`cat_target`" in q)
    assert "`cust_id`" in source_query
    assert "`customer_id`" not in source_query
    assert "`customer_id`" in target_query
    assert "`cust_id`" not in target_query
