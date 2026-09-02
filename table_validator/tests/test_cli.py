"""Tests for the tablevalidator CLI (cli/main.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from table_validator.cli.main import app
from table_validator.cli.main import _open_in_default_app as _call_real_open_in_default_app
from table_validator.config.manager import load_config, save_config
from table_validator.config.schema import ValidatorConfig

runner = CliRunner()


@pytest.fixture(autouse=True)
def _never_launch_a_real_app(monkeypatch):
    """The `open` command (and _open_in_default_app itself) launches the
    OS's default app for the report file. Autouse-mocking this for every
    test in the file means a test exercising `open` never actually
    launches Excel/LibreOffice/etc. on the machine running the suite -
    confirmed as a real risk earlier (Excel was observed running after a
    plain `pytest` invocation before this fixture existed). Tests that
    want to assert on the call itself use the returned mock; tests that
    want to exercise the real launch logic import
    _call_real_open_in_default_app (bound before this fixture ever runs)
    instead of going through the module attribute."""
    mock_open = MagicMock()
    monkeypatch.setattr("table_validator.cli.main._open_in_default_app", mock_open)
    return mock_open


def test_app_is_importable() -> None:
    assert app is not None


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "configure" in result.output
    assert "validate" in result.output
    assert "info" in result.output


def test_info_command_describes_tool_and_workflow_order() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    # All three source types are named.
    assert "Databricks catalog" in result.output
    assert "Azure Blob Storage" in result.output
    assert "Azure SQL Database" in result.output
    # The three main commands appear in workflow order.
    configure_pos = result.output.index("tablevalidator configure")
    validate_pos = result.output.index("tablevalidator validate")
    open_pos = result.output.index("tablevalidator open")
    assert configure_pos < validate_pos < open_pos


def test_validate_errors_when_no_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"
    result = runner.invoke(app, ["validate", "--config-path", str(missing_path)])
    assert result.exit_code == 1
    assert "tablevalidator configure" in result.output


def test_validate_errors_when_config_incomplete(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    save_config(ValidatorConfig(), config_path)  # empty/default config

    result = runner.invoke(app, ["validate", "--config-path", str(config_path)])
    assert result.exit_code == 1
    assert "missing" in result.output.lower()


def test_validate_errors_when_databricks_token_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.source_table.table = "customers"
    config.target_table.catalog = "tgt_cat"
    config.target_table.table = "customers"
    save_config(config, config_path)

    with patch("table_validator.cli.main.get_databricks_token", return_value=None):
        result = runner.invoke(app, ["validate", "--config-path", str(config_path)])

    assert result.exit_code == 1
    assert "token" in result.output.lower()


def _schema_df(columns):
    return pd.DataFrame(
        [
            {
                "column_name": c[0],
                "data_type": c[1],
                "is_nullable": c[2],
                "ordinal_position": i + 1,
            }
            for i, c in enumerate(columns)
        ]
    )


def _mock_databricks_connector() -> MagicMock:
    mock = MagicMock()
    mock.catalog_exists.return_value = True
    mock.get_schemas.return_value = ["bronze"]
    mock.get_tables.return_value = ["customers"]
    mock.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "string", True)]
    )
    mock.get_row_count.return_value = 10
    mock.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
        "name": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
    }
    mock.is_min_max_eligible.side_effect = lambda dt: dt.lower().startswith("int")
    mock.get_row_hashes_by_row_number.side_effect = (
        lambda catalog, schema, table, cols, bucket_predicate=None: pd.DataFrame(
            [{"row_number": 1, "row_hash": "aaa"}, {"row_number": 2, "row_hash": "bbb"}]
        )
    )
    # Fingerprint disagrees between source/target so the tiered funnel
    # (Tier 0 -> 1 -> 2 -> 4) proceeds all the way to the row-hash stage,
    # matching this fixture's pre-existing "reaches row-hash" behavior.
    mock.get_table_fingerprint.side_effect = lambda catalog, schema, table, columns, spec=None: (
        {"row_count": 10, "hash_sum": 111, "hash_xor": 222}
        if catalog == "src_cat"
        else {"row_count": 10, "hash_sum": 999, "hash_xor": 888}
    )
    return mock


def _full_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.source_table.schema_name = "bronze"
    config.source_table.table = "customers"
    config.target_table.catalog = "tgt_cat"
    config.target_table.schema_name = "bronze"
    config.target_table.table = "customers"
    save_config(config, config_path)
    return config_path


def test_validate_runs_pipeline_and_writes_report(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config-path", str(config_path),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Overall status: PASS" in result.output
    assert "1 total, 1 passed" in result.output
    assert output_path.exists()


def test_validate_never_opens_report_automatically(tmp_path: Path, _never_launch_a_real_app) -> None:
    """`validate` only writes the report - opening it is a separate,
    explicit step via `tablevalidator open`, not something validate does
    on its own."""
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    _never_launch_a_real_app.assert_not_called()
    assert "tablevalidator open" in result.output


def test_open_in_default_app_uses_os_startfile_on_windows(tmp_path: Path, monkeypatch) -> None:
    # Call through the module (table_validator.cli.main), not a `from
    # ... import _open_in_default_app` name - the autouse
    # _never_launch_a_real_app fixture above already replaced that
    # module attribute with a mock for this file, so a direct import
    # would silently bind to the mock instead of the real function.
    import table_validator.cli.main as cli_main

    monkeypatch.setattr(cli_main.platform, "system", lambda: "Windows")
    mock_startfile = MagicMock()
    monkeypatch.setattr(cli_main.os, "startfile", mock_startfile, raising=False)

    path = tmp_path / "validation_report.xlsx"
    _call_real_open_in_default_app(path)

    mock_startfile.assert_called_once_with(str(path))


def test_open_in_default_app_uses_open_on_macos(tmp_path: Path, monkeypatch) -> None:
    import table_validator.cli.main as cli_main

    monkeypatch.setattr(cli_main.platform, "system", lambda: "Darwin")
    mock_run = MagicMock()
    monkeypatch.setattr(cli_main.subprocess, "run", mock_run)

    path = tmp_path / "validation_report.xlsx"
    _call_real_open_in_default_app(path)

    mock_run.assert_called_once_with(["open", str(path)], check=True)


def test_open_in_default_app_uses_xdg_open_on_linux(tmp_path: Path, monkeypatch) -> None:
    import table_validator.cli.main as cli_main

    monkeypatch.setattr(cli_main.platform, "system", lambda: "Linux")
    mock_run = MagicMock()
    monkeypatch.setattr(cli_main.subprocess, "run", mock_run)

    path = tmp_path / "validation_report.xlsx"
    _call_real_open_in_default_app(path)

    mock_run.assert_called_once_with(["xdg-open", str(path)], check=True)


def test_open_in_default_app_swallows_failure_and_prints_manual_path(tmp_path: Path, monkeypatch) -> None:
    """A headless/sandboxed environment with no associated app must not
    crash the whole validate run - the report was already written
    successfully by this point."""
    import table_validator.cli.main as cli_main

    monkeypatch.setattr(cli_main.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        cli_main.subprocess, "run",
        MagicMock(side_effect=FileNotFoundError("xdg-open not found")),
    )

    path = tmp_path / "validation_report.xlsx"
    _call_real_open_in_default_app(path)  # must not raise


def test_open_command_opens_existing_report(tmp_path: Path, _never_launch_a_real_app) -> None:
    report_path = tmp_path / "validation_report.xlsx"
    report_path.write_bytes(b"fake xlsx content")

    result = runner.invoke(app, ["open", "--path", str(report_path)])

    assert result.exit_code == 0, result.output
    _never_launch_a_real_app.assert_called_once_with(report_path.resolve())
    assert "Opening:" in result.output


def test_open_command_uses_default_path_when_not_specified(tmp_path: Path, monkeypatch, _never_launch_a_real_app) -> None:
    monkeypatch.chdir(tmp_path)
    default_report = tmp_path / "validation_report.xlsx"
    default_report.write_bytes(b"fake xlsx content")

    result = runner.invoke(app, ["open"])

    assert result.exit_code == 0, result.output
    _never_launch_a_real_app.assert_called_once_with(default_report.resolve())


def test_open_command_errors_cleanly_when_report_missing(tmp_path: Path, _never_launch_a_real_app) -> None:
    missing_path = tmp_path / "does-not-exist.xlsx"

    result = runner.invoke(app, ["open", "--path", str(missing_path)])

    assert result.exit_code == 1
    assert "tablevalidator validate" in result.output
    _never_launch_a_real_app.assert_not_called()


def test_validate_locked_report_file_errors_cleanly_not_traceback(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ), patch(
        "table_validator.cli.main.generate_excel_report",
        side_effect=PermissionError(13, "Permission denied"),
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config-path", str(config_path),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 1
    assert "open in Excel" in result.output
    assert "--output" in result.output
    assert "Traceback" not in result.output


def test_validate_exits_nonzero_on_failed_validation(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()
    # Source and target now disagree on row count -> table FAILs.
    mock_connector.get_row_count.side_effect = [10, 5]

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config-path", str(config_path),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 1
    assert "Overall status: FAIL" in result.output


def test_validate_output_flag_writes_to_custom_path_not_default(tmp_path: Path, monkeypatch) -> None:
    """A bug where --output is silently ignored (e.g. still writing to
    the default DEFAULT_OUTPUT_PATH in the cwd) would not be caught by a
    test that only checks the custom path exists - it must also confirm
    the default path was NOT written."""
    config_path = _full_config(tmp_path)
    custom_output = tmp_path / "custom" / "my_report.xlsx"
    mock_connector = _mock_databricks_connector()

    # Run with cwd set to an isolated directory so a stray default-path
    # write is easy to detect and never touches the real project dir.
    monkeypatch.chdir(tmp_path)

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config-path", str(config_path),
                "--output", str(custom_output),
            ],
        )

    assert result.exit_code == 0, result.output
    assert custom_output.exists()
    assert not (tmp_path / "validation_report.xlsx").exists()


def _discovery_config(tmp_path: Path) -> Path:
    """Config with a blank schema/table on both sides - triggers
    CatalogValidator's internal 'compare everything matching' discovery
    (list schemas/tables on both catalogs, intersect by name) rather than
    restricting to a single named pair."""
    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.target_table.catalog = "tgt_cat"
    # schema_name/table left as their default (None) on both sides.
    save_config(config, config_path)
    return config_path


def test_validate_with_blank_schema_triggers_discovery_across_multiple_tables(
    tmp_path: Path,
) -> None:
    config_path = _discovery_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()
    # Two schemas common to both catalogs, two tables common in each.
    mock_connector.get_schemas.return_value = ["bronze", "silver"]
    mock_connector.get_tables.return_value = ["customers", "orders"]
    mock_connector.get_row_count.return_value = 10

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate",
                "--config-path", str(config_path),
                "--output", str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    # No restriction was passed - CatalogValidator discovered every
    # matching schema (bronze, silver) and, within each, every matching
    # table (customers, orders) -> 2 schemas x 2 tables = 4 tables total.
    assert "4 total, 4 passed" in result.output
    assert "(all schemas)" in result.output
    assert "Per-table results:" in result.output
    assert output_path.exists()


def test_validate_missing_table_alone_does_not_require_full_table_name(
    tmp_path: Path,
) -> None:
    """A config with catalog+schema set but table left blank must still
    pass the completeness check (only catalogs are hard-required) and
    reach the connector/validation stage rather than being rejected as
    incomplete."""
    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.source_table.schema_name = "bronze"
    config.target_table.catalog = "tgt_cat"
    config.target_table.schema_name = "bronze"
    save_config(config, config_path)

    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(app, ["validate", "--config-path", str(config_path)])

    assert "Config is incomplete" not in result.output
    assert "(all tables)" in result.output


# ---------------------------------------------------------------------------
# source_type branching: azure_blob and azure_sql paths through `validate`
# ---------------------------------------------------------------------------
def _blob_config(tmp_path: Path) -> Path:
    from table_validator.config.schema import SourceType

    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.source_type = SourceType.AZURE_BLOB
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.azure.storage_account = "n8nstorages"
    config.blob_source.container = "n8ncontainer"
    config.target_table.catalog = "tgt_cat"
    config.target_table.schema_name = "bronze"
    save_config(config, config_path)
    return config_path


def test_validate_azure_blob_source_type_runs_blob_discovery(tmp_path: Path) -> None:
    from table_validator.auth.azure_auth import AzureCredential

    config_path = _blob_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"

    mock_azure = MagicMock()
    mock_azure.container_name = "n8ncontainer"
    mock_azure.list_blobs.return_value = ["customers.csv"]
    mock_azure.read_csv.return_value = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})

    mock_databricks = MagicMock()
    mock_databricks.get_tables.return_value = ["customers"]
    mock_databricks.get_row_count.return_value = 2
    mock_databricks.get_table_schema.return_value = _schema_df(
        [("id", "bigint", False), ("name", "string", True)]
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.AzureConnector", return_value=mock_azure
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(storage_account_key="fake_key"),
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert result.exit_code == 0, result.output
    assert "azure_blob" in result.output
    assert "Overall status: PASS" in result.output
    assert output_path.exists()
    mock_azure.list_blobs.assert_called_once()


def test_validate_azure_blob_blank_schema_matches_all_schemas_not_empty_string(
    tmp_path: Path,
) -> None:
    """Regression test: a real user hit SCHEMA_NOT_FOUND from Databricks
    because target_table.schema left blank (None, to mean "all schemas")
    was being passed through as "" instead. Confirms get_tables is never
    called with an empty-string schema, and every real catalog schema is
    matched against instead."""
    from table_validator.auth.azure_auth import AzureCredential
    from table_validator.config.schema import SourceType

    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.source_type = SourceType.AZURE_BLOB
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.azure.storage_account = "n8nstorages"
    config.blob_source.container = "n8ncontainer"
    config.target_table.catalog = "for_validation1"
    config.target_table.schema_name = None  # left blank, as in the wizard
    save_config(config, config_path)

    output_path = tmp_path / "validation_report.xlsx"

    mock_azure = MagicMock()
    mock_azure.container_name = "n8ncontainer"
    mock_azure.list_blobs.return_value = ["customers.csv"]
    mock_azure.read_csv.return_value = pd.DataFrame({"id": [1, 2]})

    mock_databricks = MagicMock()
    mock_databricks.get_schemas.return_value = ["for_schema_validation"]
    mock_databricks.get_tables.return_value = ["customers"]
    mock_databricks.get_row_count.return_value = 2
    mock_databricks.get_table_schema.return_value = _schema_df(
        [("id", "bigint", False)]
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.AzureConnector", return_value=mock_azure
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(storage_account_key="fake_key"),
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert result.exit_code == 0, result.output
    assert "SCHEMA_NOT_FOUND" not in result.output
    mock_databricks.get_schemas.assert_called_once_with("for_validation1")
    for call in mock_databricks.get_tables.call_args_list:
        args, kwargs = call
        schema_arg = args[1] if len(args) > 1 else kwargs.get("schema")
        assert schema_arg == "for_schema_validation"


def test_validate_azure_blob_missing_storage_key_errors_cleanly(tmp_path: Path) -> None:
    from table_validator.auth.azure_auth import AzureCredential

    config_path = _blob_config(tmp_path)
    mock_databricks = MagicMock()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(),  # no storage_account_key
    ):
        result = runner.invoke(app, ["validate", "--config-path", str(config_path)])

    assert result.exit_code == 1
    assert "storage account key" in result.output.lower()


def _sql_config(tmp_path: Path) -> Path:
    from table_validator.config.schema import SourceType

    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.source_type = SourceType.AZURE_SQL
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.azure.sql_server = "myserver.database.windows.net"
    config.azure.sql_database = "mydb"
    config.target_table.catalog = "tgt_cat"
    save_config(config, config_path)
    return config_path


def test_validate_azure_sql_source_type_runs_sql_validator(tmp_path: Path) -> None:
    from table_validator.auth.azure_auth import AzureCredential

    config_path = _sql_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"

    mock_sql = MagicMock()
    mock_sql.get_schemas.return_value = ["dbo"]
    mock_sql.get_tables.return_value = ["customers"]
    mock_sql.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "varchar", True)]
    )
    mock_sql.get_row_count.return_value = 10
    mock_sql.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 10},
        "name": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
    }
    mock_sql.is_min_max_eligible.side_effect = lambda dt: dt.lower() in ("int",)
    mock_sql.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    mock_databricks = _mock_databricks_connector()
    mock_databricks.get_schemas.return_value = ["dbo"]
    mock_databricks.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.AzureSqlConnector", return_value=mock_sql
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(sql_username="u", sql_password="p"),
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    # Asserting a specific PASS/FAIL outcome here would require mocking
    # AzureSqlValidator's full comparison pipeline (row-hash formatting,
    # column stats, min/max, nullable) exactly - out of scope for this
    # CLI-branching test. What matters here: the azure_sql branch was
    # reached (not the databricks/blob one), it called AzureSqlConnector
    # rather than crashing, and it converged on the same report/exit-code
    # contract as the other two source types regardless of the actual
    # comparison result.
    assert result.exit_code in (0, 1), result.output
    assert "azure_sql" in result.output
    assert "Overall status:" in result.output
    mock_sql.get_schemas.assert_called_once()
    assert output_path.exists()


def test_validate_azure_sql_wires_primary_key_and_full_mode(tmp_path: Path) -> None:
    """Regression test: config.primary_key and FULL mode (needed for
    Data Mismatches to ever populate on this path) were previously
    silently dropped by _run_sql_validation - AzureSqlValidationRequest
    was built without primary_keys or data_compare_mode at all, so
    Data Mismatches could never populate regardless of what the user
    configured. This confirms both are now threaded through."""
    from table_validator.auth.azure_auth import AzureCredential
    from table_validator.models import DataCompareMode

    config_path = _sql_config(tmp_path)
    config = load_config(config_path)
    config.source_table.schema_name = "dbo"
    config.source_table.table = "customers"
    config.target_table.schema_name = "dbo"
    config.target_table.table = "customers"
    config.sql_source.table = "customers"
    config.primary_key = ["id"]
    save_config(config, config_path)
    output_path = tmp_path / "validation_report.xlsx"

    mock_sql = MagicMock()
    mock_sql.get_schemas.return_value = ["dbo"]
    mock_sql.get_tables.return_value = ["customers"]
    mock_sql.get_table_schema.return_value = _schema_df(
        [("id", "int", False), ("name", "varchar", True)]
    )
    mock_sql.get_row_count.return_value = 10
    mock_sql.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 10, "min": 1, "max": 10},
        "name": {"null_count": 0, "distinct_count": 10, "min": None, "max": None},
    }
    mock_sql.is_min_max_eligible.side_effect = lambda dt: dt.lower() in ("int",)

    mock_databricks = _mock_databricks_connector()
    mock_databricks.get_schemas.return_value = ["dbo"]

    captured_requests = []
    from table_validator.validators.row_validator import AzureSqlValidator

    real_validate = AzureSqlValidator.validate

    def spy_validate(self, request):
        captured_requests.append(request)
        return real_validate(self, request)

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.AzureSqlConnector", return_value=mock_sql
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(sql_username="u", sql_password="p"),
    ), patch.object(AzureSqlValidator, "validate", spy_validate):
        runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.data_compare_mode == DataCompareMode.FULL
    assert request.primary_keys.get("customers") == ["id"]
    assert request.primary_keys.get("dbo.customers") == ["id"]


def test_validate_azure_sql_missing_credentials_errors_cleanly(tmp_path: Path) -> None:
    from table_validator.auth.azure_auth import AzureCredential

    config_path = _sql_config(tmp_path)
    mock_databricks = MagicMock()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(),  # no sql_username/password
    ):
        result = runner.invoke(app, ["validate", "--config-path", str(config_path)])

    assert result.exit_code == 1
    assert "username" in result.output.lower() or "password" in result.output.lower()


def test_validate_azure_sql_mismatched_schema_names_still_matches_via_schema_map(
    tmp_path: Path,
) -> None:
    """Regression test for a real user-reported bug: source schema 'dbo'
    (Azure SQL) and target schema 'for_schema_validation' (Databricks)
    have different names, so name-based matching alone finds zero common
    schemas and every report sheet ends up empty. Setting sql_source.schema
    + target_table.schema explicitly must compare that pair directly via
    schema_map, without requiring the names to match."""
    from table_validator.auth.azure_auth import AzureCredential
    from table_validator.config.schema import SourceType

    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.source_type = SourceType.AZURE_SQL
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.azure.sql_server = "forsample.database.windows.net"
    config.azure.sql_database = "forsampledatabse"
    config.target_table.catalog = "for_validation1"
    config.target_table.schema_name = "for_schema_validation"
    config.sql_source.schema_name = "dbo"
    save_config(config, config_path)

    output_path = tmp_path / "validation_report.xlsx"

    mock_sql = MagicMock()
    mock_sql.get_schemas.return_value = ["dbo"]
    mock_sql.get_tables.return_value = ["employees_sample"]
    mock_sql.get_table_schema.return_value = _schema_df(
        [("id", "int", False)]
    )
    mock_sql.get_row_count.return_value = 5
    mock_sql.get_column_statistics.return_value = {
        "id": {"null_count": 0, "distinct_count": 5, "min": 1, "max": 5},
    }
    mock_sql.is_min_max_eligible.side_effect = lambda dt: dt.lower() == "int"
    mock_sql.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    mock_databricks = _mock_databricks_connector()
    mock_databricks.get_schemas.return_value = ["for_schema_validation"]
    mock_databricks.get_tables.return_value = ["employees_sample"]
    mock_databricks.get_row_hashes_by_row_number.return_value = pd.DataFrame(
        [{"row_number": 1, "row_hash": "aaa"}]
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_databricks
    ), patch(
        "table_validator.cli.main.AzureSqlConnector", return_value=mock_sql
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential",
        return_value=AzureCredential(sql_username="u", sql_password="p"),
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    # The bug produced "0 total, 0 passed" with every sheet empty - this
    # must now find and compare at least the one matched table.
    assert "0 total, 0 passed" not in result.output
    assert "1 total" in result.output


def test_databricks_validation_wires_schema_map_and_table_map_when_names_differ(
    tmp_path: Path,
) -> None:
    """Regression test for a real user-reported bug: source and target
    table/schema names that don't match by name (typo, rename, different
    casing) silently produced 0 tables / SKIPPED with no error, even
    though the user explicitly named both sides. _run_databricks_validation
    must build schema_map/table_map from config.source_table/target_table
    whenever those names differ, mirroring _run_sql_validation's existing
    pattern."""
    config_path = tmp_path / "config.yaml"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.source_table.schema_name = "bronze"
    config.source_table.table = "jd_example_data_2"
    config.target_table.catalog = "tgt_cat"
    config.target_table.schema_name = "bronze"
    config.target_table.table = "jd_example_data2"
    save_config(config, config_path)
    output_path = tmp_path / "validation_report.xlsx"

    mock_connector = _mock_databricks_connector()
    mock_connector.get_tables.side_effect = lambda catalog, schema: (
        ["jd_example_data_2"] if catalog == "src_cat" else ["jd_example_data2"]
    )

    captured_requests = []
    from table_validator.validators.catalog_validator import CatalogValidator

    real_compare_catalogs = CatalogValidator.compare_catalogs

    def spy_compare_catalogs(self, request):
        captured_requests.append(request)
        return real_compare_catalogs(self, request)

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ), patch.object(CatalogValidator, "compare_catalogs", spy_compare_catalogs):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.table_map == {"jd_example_data_2": "jd_example_data2"}
    assert request.tables == ["jd_example_data_2"]
    assert request.schema_map == {}  # schema names are identical here
    assert result.exit_code == 0, result.output
    # The mapped pair must actually be found and validated, not reported
    # as 0 tables - same regression shape as the Azure SQL schema_map bug.
    assert "0 total, 0 passed" not in result.output
    assert "1 total" in result.output


def test_databricks_validation_leaves_maps_empty_when_names_are_identical(
    tmp_path: Path,
) -> None:
    """When source/target schema and table names already match (today's
    common case), schema_map/table_map must stay empty - preserving
    100% existing behavior."""
    config_path = _full_config(tmp_path)  # identical bronze/customers on both sides
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    captured_requests = []
    from table_validator.validators.catalog_validator import CatalogValidator

    real_compare_catalogs = CatalogValidator.compare_catalogs

    def spy_compare_catalogs(self, request):
        captured_requests.append(request)
        return real_compare_catalogs(self, request)

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ), patch.object(CatalogValidator, "compare_catalogs", spy_compare_catalogs):
        runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.schema_map == {}
    assert request.table_map == {}
    assert request.schemas == ["bronze"]
    assert request.tables == ["customers"]


# ---------------------------------------------------------------------------
# --verbose/--quiet: progress logging visibility, added after a real user
# reported a multi-minute catalog-wide validate looking hung because no
# progress reached the console at all.
# ---------------------------------------------------------------------------
def test_validate_shows_per_table_progress_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "validation_report.xlsx"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.target_table.catalog = "tgt_cat"
    save_config(config, config_path)

    mock_connector = _mock_databricks_connector()
    mock_connector.get_tables.return_value = ["customers", "orders"]

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert "Validating table 'bronze.customers'" in result.output
    assert "Validating table 'bronze.orders'" in result.output
    # Detailed diagnostics stay hidden without --verbose.
    assert "[row-hash]" not in result.output
    assert "[compare_data]" not in result.output


def test_validate_verbose_shows_detailed_diagnostics(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "validation_report.xlsx"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.target_table.catalog = "tgt_cat"
    save_config(config, config_path)

    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate", "--config-path", str(config_path),
                "--output", str(output_path), "--verbose",
            ],
        )

    assert "[compare_data]" in result.output or "[row-hash]" in result.output


def test_validate_quiet_suppresses_progress_logging(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "validation_report.xlsx"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.target_table.catalog = "tgt_cat"
    save_config(config, config_path)

    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate", "--config-path", str(config_path),
                "--output", str(output_path), "--quiet",
            ],
        )

    assert "Validating table" not in result.output
    # The summary must still print - quiet only suppresses progress logging.
    assert "Overall status:" in result.output


# ---------------------------------------------------------------------------
# config.primary_key: wired into CatalogValidationRequest.primary_keys so a
# configured key is used instead of the row-number fallback - the fallback
# is what caused a real timeout on a large table (full-table sort + full
# result download), so a real key must actually take effect end-to-end.
# ---------------------------------------------------------------------------
def test_validate_configured_primary_key_avoids_row_number_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "validation_report.xlsx"
    config = ValidatorConfig()
    config.databricks.workspace_url = "https://adb-123.databricks.net"
    config.databricks.http_path = "/sql/1.0/warehouses/abc123"
    config.source_table.catalog = "src_cat"
    config.source_table.schema_name = "bronze"
    config.source_table.table = "customers"
    config.target_table.catalog = "tgt_cat"
    config.target_table.schema_name = "bronze"
    config.target_table.table = "customers"
    config.primary_key = ["id"]
    save_config(config, config_path)

    mock_connector = _mock_databricks_connector()
    mock_connector.get_row_hashes.return_value = pd.DataFrame(
        [{"id": 1, "row_hash": "aaa"}, {"id": 2, "row_hash": "bbb"}]
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert result.exit_code == 0, result.output
    mock_connector.get_row_hashes.assert_called()
    mock_connector.get_row_hashes_by_row_number.assert_not_called()


def test_validate_no_primary_key_configured_uses_row_number_fallback(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path)  # no primary_key set
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert result.exit_code == 0, result.output
    mock_connector.get_row_hashes_by_row_number.assert_called()
    mock_connector.get_row_hashes.assert_not_called()


def test_validate_databricks_wires_only_columns_ignore_columns_and_ignore_datatype(
    tmp_path: Path,
) -> None:
    """config.only_columns/ignore_columns/ignore_datatype_columns must
    reach CatalogValidationRequest unchanged - these were previously
    silently dropped (the fields didn't exist on ValidatorConfig at all,
    and _run_databricks_validation never passed them even once added)."""
    config_path = _full_config(tmp_path)
    config = load_config(config_path)
    config.only_columns = ["id", "name"]
    config.ignore_columns = ["updated_at"]
    config.ignore_datatype_columns = ["legacy_flag"]
    save_config(config, config_path)
    output_path = tmp_path / "validation_report.xlsx"

    mock_connector = _mock_databricks_connector()

    captured_requests = []
    from table_validator.validators.catalog_validator import CatalogValidator

    real_compare_catalogs = CatalogValidator.compare_catalogs

    def spy_compare_catalogs(self, request):
        captured_requests.append(request)
        return real_compare_catalogs(self, request)

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ), patch.object(CatalogValidator, "compare_catalogs", spy_compare_catalogs):
        runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.only_columns == ["id", "name"]
    assert request.ignore_columns == ["updated_at"]
    assert request.ignore_datatype_columns == ["legacy_flag"]


def test_validate_mode_stats_stops_before_fingerprint_and_row_hash(tmp_path: Path) -> None:
    """--mode stats is the CLI-flag equivalent of the tiered funnel's
    "statistical only" prompt: it must stop after Tier 1 even though the
    fixture's statistics all match, so neither the whole-table fingerprint
    nor any row-hash SQL ever runs."""
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate", "--config-path", str(config_path),
                "--output", str(output_path), "--mode", "stats",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_connector.get_table_fingerprint.assert_not_called()
    mock_connector.get_row_hashes_by_row_number.assert_not_called()
    mock_connector.get_row_hashes.assert_not_called()


def test_validate_rejects_invalid_mode_value(tmp_path: Path) -> None:
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"

    result = runner.invoke(
        app,
        [
            "validate", "--config-path", str(config_path),
            "--output", str(output_path), "--mode", "bogus",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid --mode" in result.output


def test_validate_yes_flag_never_prompts_for_partition_column(tmp_path: Path, monkeypatch) -> None:
    """--yes must disable the partition prompt entirely - even for a
    large, confirmed-mismatched table, questionary must never be invoked
    (which would hang the CliRunner waiting for input that never comes)."""
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()
    mock_connector.get_row_count.return_value = 2_000_000  # over the default threshold

    questionary_select = MagicMock()
    monkeypatch.setattr(
        "table_validator.cli.partition_prompt.questionary.select", questionary_select,
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            [
                "validate", "--config-path", str(config_path),
                "--output", str(output_path), "--yes",
            ],
        )

    assert result.exit_code in (0, 1), result.output
    questionary_select.assert_not_called()


def test_validate_without_yes_still_never_prompts_when_not_a_tty(tmp_path: Path, monkeypatch) -> None:
    """Belt-and-suspenders: even without --yes, a non-interactive stdin
    (e.g. CliRunner's default, or any real CI invocation) must never
    invoke questionary - TTY detection alone is enough to skip safely."""
    config_path = _full_config(tmp_path)
    output_path = tmp_path / "validation_report.xlsx"
    mock_connector = _mock_databricks_connector()
    mock_connector.get_row_count.return_value = 2_000_000

    questionary_select = MagicMock()
    monkeypatch.setattr(
        "table_validator.cli.partition_prompt.questionary.select", questionary_select,
    )

    with patch(
        "table_validator.cli.main.DatabricksConnector", return_value=mock_connector
    ), patch(
        "table_validator.cli.main.get_databricks_token", return_value="dapi_fake"
    ), patch(
        "table_validator.cli.main.get_azure_credential", return_value=None
    ):
        result = runner.invoke(
            app,
            ["validate", "--config-path", str(config_path), "--output", str(output_path)],
        )

    assert result.exit_code in (0, 1), result.output
    questionary_select.assert_not_called()
