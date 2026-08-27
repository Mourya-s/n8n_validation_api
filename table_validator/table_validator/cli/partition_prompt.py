"""
Interactive partition-column prompt for `tablevalidator validate`.

Invoked by CatalogValidator (validators/catalog_validator.py) via the
optional partition_prompt callback, only for a table that is both large
(row count over CatalogValidationRequest.partition_threshold) and has a
confirmed mismatch (Tier 1 and/or Tier 2). The validator itself has no
I/O - this module is the only place that actually talks to a human about
which column to bucket by.

build_partition_prompt() is the factory `cli/main.py` calls once per
`validate` invocation; it captures --yes and TTY state so the returned
callback can decide, on every call, whether to actually prompt or skip
straight to "no partitioning" without ever risking a hang in a
non-interactive/CI run.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import questionary
import typer

from table_validator.models import PartitionPromptContext

logger = logging.getLogger(__name__)

_SKIP_LABEL = "Skip partitioning (compare the whole table)"


def run_partition_prompt(context: PartitionPromptContext) -> Optional[str]:
    """
    Ask the user which column to partition/bucket by for a large,
    confirmed-mismatched table, or let them skip. Returns the chosen
    column name, or None if they chose to skip.
    """
    typer.echo(
        f"\nTable '{context.schema_name}.{context.table}' has "
        f"{context.row_count:,} rows and a confirmed mismatch. Comparing "
        f"it row-by-row could be slow - partitioning first can narrow "
        f"down which part of the table actually differs.",
    )

    choices = list(context.candidate_columns) + [_SKIP_LABEL]
    answer = questionary.select(
        "Partition by which column?",
        choices=choices,
    ).ask()

    if answer is None or answer == _SKIP_LABEL:
        return None
    return answer


def build_partition_prompt(yes: bool):
    """
    Factory for the callback passed to CatalogValidator(partition_prompt=...).

    Returns None (meaning "never prompt, always fall back to unpartitioned
    Tier 4") when --yes was passed or stdin isn't a real terminal - belt
    and suspenders, so a non-interactive invocation can never hang on a
    prompt even if --yes was forgotten.
    """
    if yes:
        logger.debug("--yes passed - partition prompt disabled")
        return None

    if not sys.stdin.isatty():
        logger.debug("stdin is not a TTY - partition prompt disabled")
        return None

    def _callback(context: PartitionPromptContext) -> Optional[str]:
        return run_partition_prompt(context)

    return _callback
