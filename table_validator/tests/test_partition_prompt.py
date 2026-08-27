"""Tests for cli/partition_prompt.py's build_partition_prompt factory."""

from __future__ import annotations

from unittest.mock import MagicMock

from table_validator.cli.partition_prompt import build_partition_prompt, run_partition_prompt
from table_validator.models import PartitionPromptContext


def _context() -> PartitionPromptContext:
    return PartitionPromptContext(
        schema_name="bronze", table="customers", row_count=2_000_000,
        candidate_columns=["region", "created_at"],
    )


def test_build_partition_prompt_returns_none_when_yes_flag_set():
    callback = build_partition_prompt(yes=True)
    assert callback is None


def test_build_partition_prompt_returns_none_when_stdin_not_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    callback = build_partition_prompt(yes=False)
    assert callback is None


def test_build_partition_prompt_returns_callback_when_interactive(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    callback = build_partition_prompt(yes=False)
    assert callback is not None
    assert callable(callback)


def test_run_partition_prompt_returns_chosen_column(monkeypatch):
    mock_select = MagicMock()
    mock_select.return_value.ask.return_value = "region"
    monkeypatch.setattr("table_validator.cli.partition_prompt.questionary.select", mock_select)

    result = run_partition_prompt(_context())

    assert result == "region"
    call_kwargs = mock_select.call_args.kwargs
    assert "region" in call_kwargs["choices"]
    assert "created_at" in call_kwargs["choices"]


def test_run_partition_prompt_skip_choice_returns_none(monkeypatch):
    mock_select = MagicMock()
    mock_select.return_value.ask.return_value = "Skip partitioning (compare the whole table)"
    monkeypatch.setattr("table_validator.cli.partition_prompt.questionary.select", mock_select)

    assert run_partition_prompt(_context()) is None


def test_run_partition_prompt_none_answer_returns_none(monkeypatch):
    """Ctrl-C / EOF during the prompt -> questionary returns None."""
    mock_select = MagicMock()
    mock_select.return_value.ask.return_value = None
    monkeypatch.setattr("table_validator.cli.partition_prompt.questionary.select", mock_select)

    assert run_partition_prompt(_context()) is None
