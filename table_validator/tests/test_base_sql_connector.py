"""
Tests for BaseSqlConnector's row-filter mechanism (set_row_filters /
_scoped_table) - the shared logic behind validate_tables()'s
row_filter/source_row_filter/target_row_filter kwargs.

Exercised through SparkConnector (a concrete BaseSqlConnector subclass)
since that's the only connector that ever calls set_row_filters today -
the CLI/DatabricksConnector path never touches this mechanism, and these
tests double as the regression guard proving that "no filter set" always
produces byte-identical output to _qualify's own, unchanged since before
this feature existed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from table_validator.connectors.spark_connector import SparkConnector


def _connector() -> SparkConnector:
    return SparkConnector(spark=MagicMock())


# ---------------------------------------------------------------------------
# _scoped_table: no filter set -> byte-identical to _qualify (regression
# guard - every pre-existing caller's generated SQL must be unaffected).
# ---------------------------------------------------------------------------
def test_scoped_table_with_no_filters_matches_qualify_exactly():
    connector = _connector()
    assert connector._scoped_table("cat", "sch", "tbl") == connector._qualify("cat", "sch", "tbl")


def test_scoped_table_with_no_filters_ignores_side_argument():
    connector = _connector()
    expected = connector._qualify("cat", "sch", "tbl")
    assert connector._scoped_table("cat", "sch", "tbl", "source") == expected
    assert connector._scoped_table("cat", "sch", "tbl", "target") == expected
    assert connector._scoped_table("cat", "sch", "tbl", None) == expected


# ---------------------------------------------------------------------------
# set_row_filters / _scoped_table combinations.
# ---------------------------------------------------------------------------
def test_common_filter_applies_regardless_of_side():
    connector = _connector()
    connector.set_row_filters(common="status = 'active'")

    expected = "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (status = 'active')) AS filtered"
    assert connector._scoped_table("cat", "sch", "tbl") == expected
    assert connector._scoped_table("cat", "sch", "tbl", "source") == expected
    assert connector._scoped_table("cat", "sch", "tbl", "target") == expected


def test_source_only_filter_applies_only_to_source_side():
    connector = _connector()
    connector.set_row_filters(source="id > 20")

    assert connector._scoped_table("cat", "sch", "tbl", "source") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (id > 20)) AS filtered"
    )
    # No common filter and this isn't the source side - unaffected.
    assert connector._scoped_table("cat", "sch", "tbl", "target") == connector._qualify(
        "cat", "sch", "tbl"
    )
    assert connector._scoped_table("cat", "sch", "tbl") == connector._qualify("cat", "sch", "tbl")


def test_target_only_filter_applies_only_to_target_side():
    connector = _connector()
    connector.set_row_filters(target="id > 15")

    assert connector._scoped_table("cat", "sch", "tbl", "target") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (id > 15)) AS filtered"
    )
    assert connector._scoped_table("cat", "sch", "tbl", "source") == connector._qualify(
        "cat", "sch", "tbl"
    )


def test_common_and_source_filter_combine_with_and():
    connector = _connector()
    connector.set_row_filters(common="status = 'active'", source="id > 20")

    assert connector._scoped_table("cat", "sch", "tbl", "source") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (status = 'active') AND (id > 20)) AS filtered"
    )
    # Target only gets the common filter.
    assert connector._scoped_table("cat", "sch", "tbl", "target") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (status = 'active')) AS filtered"
    )


def test_common_source_and_target_filters_all_combine_correctly():
    connector = _connector()
    connector.set_row_filters(
        common="status = 'active'", source="id > 20", target="id > 15",
    )

    assert connector._scoped_table("cat", "sch", "tbl", "source") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (status = 'active') AND (id > 20)) AS filtered"
    )
    assert connector._scoped_table("cat", "sch", "tbl", "target") == (
        "(SELECT * FROM `cat`.`sch`.`tbl` WHERE (status = 'active') AND (id > 15)) AS filtered"
    )


def test_set_row_filters_can_be_cleared_by_calling_again_with_none():
    connector = _connector()
    connector.set_row_filters(common="status = 'active'")
    connector.set_row_filters()  # all None - clears every filter

    assert connector._scoped_table("cat", "sch", "tbl") == connector._qualify("cat", "sch", "tbl")


# ---------------------------------------------------------------------------
# End-to-end: a real query-building method actually applies the filter via
# _scoped_table, confirmed against generated SQL text.
# ---------------------------------------------------------------------------
def test_get_row_count_applies_filter_via_side_kwarg():
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame({"row_count": [3]})
    connector = SparkConnector(spark=fake_spark)
    connector.set_row_filters(common="id > 20")

    connector.get_row_count("cat", "sch", "tbl", side="source")

    called_query = fake_spark.sql.call_args[0][0]
    assert "WHERE (id > 20)" in called_query
    assert "AS filtered" in called_query


def test_get_row_count_without_side_kwarg_is_unaffected_by_filters():
    """A caller that doesn't pass `side` at all (side defaults to None)
    still gets the common filter (if any) applied - only per-side
    filters require the kwarg."""
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame({"row_count": [3]})
    connector = SparkConnector(spark=fake_spark)
    connector.set_row_filters(source="id > 20")  # source-only, no common

    connector.get_row_count("cat", "sch", "tbl")  # side defaults to None

    called_query = fake_spark.sql.call_args[0][0]
    assert "WHERE" not in called_query
    assert "AS filtered" not in called_query
