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


# ---------------------------------------------------------------------------
# get_column_statistics: real user-reported bug - a row_filter that
# legitimately matches zero rows on one side (e.g. row_filter="gender =
# 'male'" against a table/filter combination with no matches) still gets
# exactly one row back from the SUM/COUNT DISTINCT aggregate query (no
# GROUP BY), but the aggregate values themselves are SQL NULL - which a
# spark.sql(...).toPandas() round-trip represents as float NaN, not
# Python None. `value is not None` doesn't catch NaN (it's a different
# object), so int(NaN) reached int() and raised ValueError, crashing the
# whole comparison with "ValueError: cannot convert float NaN to integer"
# instead of reporting a clean SKIPPED/None result for that side.
# ---------------------------------------------------------------------------
def test_get_column_statistics_handles_nan_aggregates_without_crashing():
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame(
        {
            "id__nulls": [float("nan")],
            "id__distinct": [float("nan")],
            "gender__nulls": [float("nan")],
            "gender__distinct": [float("nan")],
        }
    )
    connector = SparkConnector(spark=fake_spark)
    connector.set_row_filters(common="gender = 'male'")

    # Must not raise.
    stats = connector.get_column_statistics("cat", "sch", "tbl", ["id", "gender"])

    assert stats["id"]["null_count"] is None
    assert stats["id"]["distinct_count"] is None
    assert stats["gender"]["null_count"] is None
    assert stats["gender"]["distinct_count"] is None


def test_get_column_statistics_still_converts_real_values_to_int():
    """Regression guard alongside the NaN fix above - a genuine numeric
    aggregate result must still come back as a real int, not accidentally
    treated as null-like by the new NaN check."""
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame(
        {"id__nulls": [0], "id__distinct": [42]}
    )
    connector = SparkConnector(spark=fake_spark)

    stats = connector.get_column_statistics("cat", "sch", "tbl", ["id"])

    assert stats["id"]["null_count"] == 0
    assert stats["id"]["distinct_count"] == 42
    assert isinstance(stats["id"]["null_count"], int)
    assert isinstance(stats["id"]["distinct_count"], int)


def test_get_column_statistics_min_max_nan_becomes_none():
    """Real bug found via live user testing: comparing a table against
    ITSELF with the exact same row_filter on both sides (matching zero
    rows) still reported Overall Status FAIL, specifically Min/Max FAIL -
    MIN/MAX over zero rows is SQL NULL -> NaN after the pandas round-
    trip, and NaN != NaN (IEEE 754), so compare_min_max's `==` check
    treated two equally-empty, equally-NaN sides as a real difference.
    min/max must come back as None (not NaN) so two empty sides compare
    equal."""
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame(
        {"id__nulls": [float("nan")], "id__distinct": [float("nan")],
         "id__min": [float("nan")], "id__max": [float("nan")]}
    )
    connector = SparkConnector(spark=fake_spark)

    stats = connector.get_column_statistics(
        "cat", "sch", "tbl", ["id"], min_max_columns=["id"],
    )

    assert stats["id"]["min"] is None
    assert stats["id"]["max"] is None


def test_get_column_statistics_min_max_real_values_pass_through_unchanged():
    """Regression guard alongside the NaN fix above - a genuine min/max
    result (of any type - string, date, number) must pass through
    completely unchanged, not be mistaken for NaN."""
    fake_spark = MagicMock()
    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame(
        {"name__nulls": [0], "name__distinct": [5],
         "name__min": ["Alice"], "name__max": ["Zoe"]}
    )
    connector = SparkConnector(spark=fake_spark)

    stats = connector.get_column_statistics(
        "cat", "sch", "tbl", ["name"], min_max_columns=["name"],
    )

    assert stats["name"]["min"] == "Alice"
    assert stats["name"]["max"] == "Zoe"


def test_two_identically_filtered_empty_sides_compare_as_pass():
    """End-to-end regression test for the exact reported scenario: the
    SAME table compared against itself with the SAME row_filter matching
    zero rows on both sides must report min/max as equal (both None),
    not a false mismatch."""
    from table_validator.validators.catalog_validator import CatalogValidator

    fake_spark = MagicMock()
    connector = SparkConnector(spark=fake_spark)
    connector.set_row_filters(common="gender = 'male'")

    fake_spark.sql.return_value.toPandas.return_value = pd.DataFrame(
        {"id__nulls": [float("nan")], "id__distinct": [float("nan")],
         "id__min": [float("nan")], "id__max": [float("nan")]}
    )
    source_stats = connector.get_column_statistics(
        "cat", "sch", "src_tbl", ["id"], min_max_columns=["id"], side="source",
    )
    target_stats = connector.get_column_statistics(
        "cat", "sch", "tgt_tbl", ["id"], min_max_columns=["id"], side="target",
    )

    validator = CatalogValidator(connector)
    status = validator.compare_min_max(
        source_stats["id"]["min"], source_stats["id"]["max"],
        target_stats["id"]["min"], target_stats["id"]["max"],
    )
    assert status.value == "PASS"
