"""
Tests for validators/mismatch_classifier.py's classify_mismatch() - one
test per category label, plus priority-order edge cases (e.g. a value that
would match more than one rule must resolve to the earlier rule).
"""

from __future__ import annotations

import datetime
from decimal import Decimal

from table_validator.validators.mismatch_classifier import (
    CASE_DIFFERENCE,
    FORMATTING_DIFF,
    NULL_MISMATCH,
    PRECISION_LOSS,
    STRING_TRUNCATION,
    VALUE_MISMATCH,
    WHITESPACE_DIFF,
    classify_mismatch,
)


# ---------------------------------------------------------------------------
# Rule 1: NULL_MISMATCH
# ---------------------------------------------------------------------------
def test_source_none_target_present_is_null_mismatch():
    assert classify_mismatch(None, "ACTIVE") == NULL_MISMATCH


def test_source_present_target_none_is_null_mismatch():
    assert classify_mismatch("ACTIVE", None) == NULL_MISMATCH


def test_nan_target_counts_as_null():
    assert classify_mismatch("ACTIVE", float("nan")) == NULL_MISMATCH


def test_both_none_is_not_null_mismatch_since_nothing_actually_differs():
    # Not a real mismatch to begin with - classify_mismatch doesn't
    # re-verify inequality, so this just must not crash and must not
    # claim NULL_MISMATCH (which implies exactly one side is null).
    assert classify_mismatch(None, None) == VALUE_MISMATCH


# ---------------------------------------------------------------------------
# Rule 2: STRING_TRUNCATION
# ---------------------------------------------------------------------------
def test_target_truncated_prefix_of_source():
    assert classify_mismatch("John Doe", "John Do") == STRING_TRUNCATION


def test_truncation_by_a_single_character_still_counts():
    # The canonical example ("John Doe" -> "John Do") is only 1 char
    # shorter - truncation isn't gated on a minimum character count, only
    # on "target is a shorter prefix of source".
    assert classify_mismatch("Johne", "John") == STRING_TRUNCATION


def test_truncation_requires_source_to_start_with_target():
    # Target is shorter by enough chars but is NOT a prefix of source -
    # not truncation.
    assert classify_mismatch("John Doe Smith", "Zzzzzzz") == VALUE_MISMATCH


# ---------------------------------------------------------------------------
# Rule 3: CASE_DIFFERENCE
# ---------------------------------------------------------------------------
def test_case_difference_detected():
    assert classify_mismatch("ACTIVE", "active") == CASE_DIFFERENCE


def test_case_difference_mixed_case():
    assert classify_mismatch("Pending", "PENDING") == CASE_DIFFERENCE


# ---------------------------------------------------------------------------
# Rule 4: WHITESPACE_DIFF
# ---------------------------------------------------------------------------
def test_whitespace_diff_trailing_space():
    assert classify_mismatch("ACTIVE", "ACTIVE ") == WHITESPACE_DIFF


def test_whitespace_diff_leading_and_trailing():
    assert classify_mismatch("  value", "value  ") == WHITESPACE_DIFF


# ---------------------------------------------------------------------------
# Rule 5: PRECISION_LOSS
# ---------------------------------------------------------------------------
def test_precision_loss_trailing_zero_dropped():
    assert classify_mismatch(1500.50, 1500.5) == PRECISION_LOSS


def test_precision_loss_decimal_vs_float():
    assert classify_mismatch(Decimal("1500.50"), 1500.5) == PRECISION_LOSS


def test_precision_loss_float_rounding_noise():
    assert classify_mismatch(1500.5, 1500.500001) == PRECISION_LOSS


def test_genuinely_different_numbers_are_value_mismatch_not_precision_loss():
    assert classify_mismatch(1500.50, 1600.50) == VALUE_MISMATCH


# ---------------------------------------------------------------------------
# Rule 6: FORMATTING_DIFF
# ---------------------------------------------------------------------------
def test_formatting_diff_thousands_separator():
    assert classify_mismatch("1,000", "1000") == FORMATTING_DIFF


def test_formatting_diff_iso_vs_us_date_string():
    assert classify_mismatch("2024-01-15", "01/15/2024") == FORMATTING_DIFF


def test_formatting_diff_date_object_vs_iso_string():
    assert classify_mismatch(datetime.date(2024, 1, 15), "2024-01-15") == FORMATTING_DIFF


def test_formatting_diff_datetime_object_vs_us_string():
    assert (
        classify_mismatch(datetime.datetime(2024, 1, 15, 0, 0, 0), "01/15/2024")
        == FORMATTING_DIFF
    )


# ---------------------------------------------------------------------------
# Rule 7: VALUE_MISMATCH (catch-all)
# ---------------------------------------------------------------------------
def test_completely_different_strings_is_value_mismatch():
    assert classify_mismatch("ACTIVE", "CANCELLED") == VALUE_MISMATCH


def test_different_dates_is_value_mismatch_not_formatting_diff():
    assert classify_mismatch("2024-01-15", "2024-02-20") == VALUE_MISMATCH


def test_type_change_number_vs_unrelated_string_is_value_mismatch():
    assert classify_mismatch(42, "banana") == VALUE_MISMATCH


# ---------------------------------------------------------------------------
# Priority order: a pair that could match multiple rules must resolve to
# whichever rule comes first.
# ---------------------------------------------------------------------------
def test_null_mismatch_takes_priority_over_everything_else():
    # Would also "look like" other things if null-checking weren't first,
    # e.g. an empty string vs None must not be misread as whitespace/case.
    assert classify_mismatch(None, "") == NULL_MISMATCH


def test_case_difference_checked_before_whitespace_diff():
    # Equal only after BOTH lower() and strip() would apply - case
    # (rule 3) must win since it's checked first.
    assert classify_mismatch("ACTIVE", "active") == CASE_DIFFERENCE


def test_precision_loss_checked_before_formatting_diff():
    # Numerically CLOSE but not exactly equal, with a string on one side
    # (float round-trip noise, not a real formatting choice) ->
    # PRECISION_LOSS, not FORMATTING_DIFF - an exact match with a string
    # side (e.g. "1,000" vs 1000) is the FORMATTING_DIFF case instead,
    # covered separately above.
    assert classify_mismatch("1500.500001", 1500.5) == PRECISION_LOSS


def test_exact_numeric_match_with_string_side_is_formatting_not_precision():
    assert classify_mismatch("1,500.50", 1500.5) == FORMATTING_DIFF
