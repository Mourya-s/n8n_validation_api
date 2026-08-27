"""
Tests for `tablevalidator report` and the shared summary_table module it
uses (summary_from_response/summary_from_excel/print_summary_table).
Confirms: no report found -> clear message + non-zero exit; report found
-> correct data pulled from a real generated fixture file; and that the
same data comes back whether built from a live response or read back
from the saved .xlsx (validate and report never disagree).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from table_validator.cli.main import app
from table_validator.cli.summary_table import (
    print_summary_table,
    summary_from_excel,
    summary_from_response,
)
from table_validator.models import (
    CatalogValidationResponse,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
)
from table_validator.reports.excel_report import generate_excel_report

runner = CliRunner()


def _make_response(statuses) -> CatalogValidationResponse:
    """statuses: list of ValidationStatus, one per table."""
    tables = [
        TableValidationResult(schema_name="bronze", table=f"table_{i}", status=status)
        for i, status in enumerate(statuses)
    ]
    overall = ValidationStatus.PASS
    if any(s == ValidationStatus.ERROR for s in statuses):
        overall = ValidationStatus.ERROR
    elif any(s == ValidationStatus.FAIL for s in statuses):
        overall = ValidationStatus.FAIL

    return CatalogValidationResponse(
        source_catalog="src_cat",
        target_catalog="tgt_cat",
        status=overall,
        validation_timestamp="2026-01-01T00:00:00+00:00",
        execution_time_seconds=1.234,
        schemas=[
            SchemaValidationResult(schema_name="bronze", status=overall, tables=tables)
        ],
    )


# ---------------------------------------------------------------------------
# `tablevalidator report`: no report found
# ---------------------------------------------------------------------------
def test_report_command_no_file_prints_clear_message_and_exits_nonzero(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.xlsx"
    result = runner.invoke(app, ["report", "--path", str(missing_path)])

    assert result.exit_code == 1
    assert "No validation report found" in result.output
    assert "tablevalidator validate" in result.output


# ---------------------------------------------------------------------------
# `tablevalidator report`: report exists, real generated fixture file
# ---------------------------------------------------------------------------
def test_report_command_reads_real_generated_report(tmp_path: Path) -> None:
    response = _make_response([ValidationStatus.PASS, ValidationStatus.FAIL])
    report_path = tmp_path / "validation_report.xlsx"
    generate_excel_report(
        response, str(report_path), source_type="databricks",
        enabled_validations={"catalog", "schema", "column", "row"},
    )

    result = runner.invoke(app, ["report", "--path", str(report_path)])

    assert result.exit_code == 0, result.output
    assert "FAIL" in result.output
    assert "2" in result.output  # total tables
    assert "1" in result.output  # passed tables
    assert "databricks" in result.output
    assert str(report_path) in result.output


def test_report_command_defaults_to_validation_report_xlsx_in_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    response = _make_response([ValidationStatus.PASS])
    generate_excel_report(response, "validation_report.xlsx")

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


# ---------------------------------------------------------------------------
# summary_from_response / summary_from_excel must agree on the same result
# ---------------------------------------------------------------------------
def test_summary_from_response_and_summary_from_excel_agree(tmp_path: Path) -> None:
    response = _make_response(
        [ValidationStatus.PASS, ValidationStatus.PASS, ValidationStatus.FAIL, ValidationStatus.ERROR]
    )
    report_path = tmp_path / "validation_report.xlsx"
    generate_excel_report(
        response, str(report_path), source_type="azure_sql",
    )

    from_response = summary_from_response(response, source_type="azure_sql")
    from_excel = summary_from_excel(report_path)

    assert from_response.overall_status == from_excel.overall_status
    assert from_response.total_tables == from_excel.total_tables
    assert from_response.passed_tables == from_excel.passed_tables
    assert from_response.failed_tables == from_excel.failed_tables
    assert from_response.error_tables == from_excel.error_tables
    assert from_response.pass_percentage == from_excel.pass_percentage
    assert from_response.source_type == from_excel.source_type


def test_summary_from_excel_handles_missing_optional_fields(tmp_path: Path) -> None:
    """A report generated without source_type/enabled_validations (older
    format, or a caller that doesn't pass them) must still read back
    cleanly - the Summary sheet's row layout shifts when these are absent."""
    response = _make_response([ValidationStatus.PASS])
    report_path = tmp_path / "validation_report.xlsx"
    generate_excel_report(response, str(report_path))  # no source_type/enabled_validations

    data = summary_from_excel(report_path)

    assert data.overall_status == "PASS"
    assert data.total_tables == 1
    assert data.source_type is None
    assert data.validations_run is None


def test_print_summary_table_does_not_raise(tmp_path: Path) -> None:
    """Smoke test: rendering must not crash regardless of which optional
    fields are populated."""
    from rich.console import Console

    response = _make_response([ValidationStatus.PASS, ValidationStatus.SKIPPED])
    data = summary_from_response(response, source_type="databricks", validations_run="row")

    console = Console(file=open(tmp_path / "out.txt", "w", encoding="utf-8"))
    print_summary_table(data, console=console)  # must not raise
