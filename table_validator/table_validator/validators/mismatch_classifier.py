"""
Mismatch categorization: root-cause labeling for a single mismatched cell.

classify_mismatch() is a pure function - given the source/target values for
one mismatched (source_value, target_value) pair (as already captured on
RowMismatchDetail - see models.py), it returns a short category label
explaining WHY the two values differ, not just THAT they differ. This is
phase 1 of a larger reporting feature: later phases aggregate these labels
across a table's mismatches into a categorized summary (report-layer only -
this module has no I/O and makes no decisions about PASS/FAIL).

Deliberately takes raw values rather than a full RowMismatchDetail or a
dtype hint: RowMismatchDetail doesn't carry source_data_type/target_data_type
today (those live on the separate ColumnValidationResult, one level up, and
aren't always available at the point a mismatch is classified). Every rule
below that cares about "is this numeric" or "is this a date" infers it from
the values themselves - the same convention databricks_connector.values_differ
already uses (numbers.Number covers int/float/Decimal/numpy scalars in one
isinstance check; datetime.date/datetime.datetime for dates) - so this stays
usable from anywhere, including plain unit tests with bare Python values.
"""

from __future__ import annotations

import datetime
import numbers
import re
from typing import Any, Optional

# Mirrors databricks_connector._NUMERIC_TYPES - numbers.Number covers
# int/float/Decimal and every numpy numeric scalar type in one isinstance
# check, which matters since values compared here may come from either a
# pandas/Databricks result (numpy scalars) or a plain DB-API driver
# (int/float/Decimal).
_NUMERIC_TYPES = numbers.Number

# Category labels, in the required priority order (see classify_mismatch).
NULL_MISMATCH = "NULL_MISMATCH"
STRING_TRUNCATION = "STRING_TRUNCATION"
CASE_DIFFERENCE = "CASE_DIFFERENCE"
WHITESPACE_DIFF = "WHITESPACE_DIFF"
PRECISION_LOSS = "PRECISION_LOSS"
FORMATTING_DIFF = "FORMATTING_DIFF"
VALUE_MISMATCH = "VALUE_MISMATCH"

# Rule 5 (PRECISION_LOSS): two numeric values are considered "the same
# number, precision-truncated" rather than a real value change when they
# agree to within this tolerance - loose enough to catch e.g. 1500.5 vs
# 1500.500001 (float round-trip noise), tight enough that a genuinely
# different number (1500.50 vs 1600.50) still falls through to
# VALUE_MISMATCH. Deliberately NOT used for "are these exactly the same
# number" (see _NUMBERS_EQUAL_TOLERANCE below) - two values that are
# EXACTLY numerically equal but textually different (e.g. "1,000" vs
# "1000") are a formatting difference (rule 6), not precision loss.
_PRECISION_TOLERANCE = 1e-4

# Numbers within this (much tighter) tolerance of each other are treated
# as "the same number" for rule 6's formatting-difference check - a
# separate, tighter threshold than _PRECISION_TOLERANCE so an exact
# match (or float round-trip noise at the limits of float precision)
# routes to FORMATTING_DIFF, while anything precision-loss-like still
# routes to rule 5 first.
_NUMBERS_EQUAL_TOLERANCE = 1e-9

# Rule 6 (FORMATTING_DIFF): recognizes a handful of extremely common
# "same value, different text representation" shapes without pulling in
# a date-parsing dependency - deliberately conservative (stdlib-only, no
# new imports) rather than trying to cover every locale/format.
_THOUSANDS_SEP_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_US_DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")


def _is_null(value: Any) -> bool:
    """True for None and float NaN (pandas/numpy's usual stand-in for a
    SQL NULL once a value has round-tripped through a DataFrame) - NOT for
    an empty string or 0, which are real, present values."""
    if value is None:
        return True
    if isinstance(value, float):
        return value != value  # NaN is the only float that isn't equal to itself
    return False


def _as_number(value: Any) -> Optional[float]:
    """Best-effort float coercion for rule 5/6's numeric comparisons.
    Returns None (never raises) when value isn't numeric-like - a bare
    numeric type check (_NUMERIC_TYPES) misses the common case of a
    number that arrived as a string (e.g. from a CSV or a driver that
    returns everything as str), which formatting-difference detection
    specifically needs to handle."""
    if isinstance(value, bool):
        # bool is technically a numbers.Number subclass in Python, but
        # comparing True/False as 1/0 here would misclassify an actual
        # boolean-vs-boolean value mismatch as a numeric one.
        return None
    if isinstance(value, _NUMERIC_TYPES):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if _THOUSANDS_SEP_RE.match(stripped):
            stripped = stripped.replace(",", "")
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _as_iso_date(value: Any) -> Optional[str]:
    """Normalizes a date-shaped value (a real date/datetime, or a string
    in ISO 'YYYY-MM-DD' or US 'MM/DD/YYYY' form) to a canonical
    'YYYY-MM-DD' string for rule 6's formatting-difference comparison.
    Returns None for anything not recognizably a date - deliberately
    narrow rather than a general date parser, matching this module's
    stdlib-only, no-new-imports constraint."""
    if isinstance(value, datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        iso_match = _ISO_DATE_RE.match(text)
        if iso_match:
            return text
        us_match = _US_DATE_RE.match(text)
        if us_match:
            month, day, year = us_match.groups()
            return f"{year}-{month}-{day}"
    return None


def classify_mismatch(source_value: Any, target_value: Any) -> str:
    """
    Classify why two independently-fetched cell values differ, applying
    these rules in order (first match wins):

      1. NULL_MISMATCH     - one side is None/NaN, the other is not
      2. STRING_TRUNCATION - both strings, target is a shorter, non-empty
                              prefix of source (any length difference)
      3. CASE_DIFFERENCE   - values are equal after .lower()
      4. WHITESPACE_DIFF   - values are equal after .strip()
      5. PRECISION_LOSS    - both numeric (or numeric-looking), and equal
                              within a small tolerance
      6. FORMATTING_DIFF   - both represent the same number or date, just
                              formatted differently (e.g. "1,000" vs
                              "1000", "2024-01-01" vs "01/01/2024")
      7. VALUE_MISMATCH    - catch-all: a genuine value difference

    Pure function: no I/O, stdlib only. Callers are expected to have
    already confirmed the two values actually differ (e.g. via
    databricks_connector.values_differ) - classify_mismatch itself does
    not re-check equality, so calling it on two equal values will still
    return a label (harmlessly, but meaninglessly).
    """
    # Rule 1: NULL_MISMATCH.
    source_null = _is_null(source_value)
    target_null = _is_null(target_value)
    if source_null != target_null:
        return NULL_MISMATCH
    if source_null and target_null:
        # Both sides null - nothing to classify further (and not a real
        # mismatch at all), but still needs a label rather than falling
        # through to string/numeric rules below with two None values.
        return VALUE_MISMATCH

    # Rule 2: STRING_TRUNCATION - only meaningful when both sides are
    # actually strings (not e.g. two numbers that happen to stringify
    # differently in length). Any shorter, non-empty prefix counts (the
    # canonical example - "John Doe" -> "John Do" - is only 1 char
    # shorter), so the only real gate is "source starts with target" -
    # an empty target is excluded since every non-empty string trivially
    # "starts with" "", which would misclassify an unrelated value as
    # truncation.
    if isinstance(source_value, str) and isinstance(target_value, str):
        if (
            target_value
            and len(target_value) < len(source_value)
            and source_value.startswith(target_value)
        ):
            return STRING_TRUNCATION

        # Rule 3: CASE_DIFFERENCE.
        if source_value.lower() == target_value.lower():
            return CASE_DIFFERENCE

        # Rule 4: WHITESPACE_DIFF.
        if source_value.strip() == target_value.strip():
            return WHITESPACE_DIFF

    # Rules 5/6a both start from "do these coerce to the same underlying
    # number" - which one applies depends on WHY the text differs:
    #   - exactly (or near-exactly) equal, but at least one side arrived
    #     as a string with different formatting (e.g. "1,000" vs "1000")
    #     -> FORMATTING_DIFF (rule 6a): nothing was lost, just displayed
    #     differently.
    #   - close but not equal (e.g. 1500.5 vs 1500.500001, float
    #     round-trip noise) -> PRECISION_LOSS (rule 5): the values
    #     themselves differ slightly.
    # Order matters: FORMATTING_DIFF is checked first since an exact
    # match is also trivially "within tolerance" for rule 5.
    source_num = _as_number(source_value)
    target_num = _as_number(target_value)
    if source_num is not None and target_num is not None:
        if isinstance(source_value, str) or isinstance(target_value, str):
            if abs(source_num - target_num) <= _NUMBERS_EQUAL_TOLERANCE:
                return FORMATTING_DIFF

        if abs(source_num - target_num) <= _PRECISION_TOLERANCE:
            return PRECISION_LOSS

    # Rule 6b: FORMATTING_DIFF for dates - same calendar date, different
    # textual representation (ISO vs US, or a date/datetime object vs a
    # string form of the same date).
    source_date = _as_iso_date(source_value)
    target_date = _as_iso_date(target_value)
    if source_date is not None and target_date is not None and source_date == target_date:
        return FORMATTING_DIFF

    # Rule 7: catch-all.
    return VALUE_MISMATCH
