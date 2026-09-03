"""CLI entry point: `tablevalidator` console script."""

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Optional

import typer

from table_validator.auth.azure_auth import get_azure_credential
from table_validator.auth.databricks_auth import get_databricks_token, host_from_workspace_url
from table_validator.cli.summary_table import (
    print_summary_table,
    summary_from_excel,
    summary_from_response,
)
from table_validator.cli.partition_prompt import build_partition_prompt
from table_validator.cli.wizard import run_configure_wizard
from table_validator.config.manager import CONFIG_PATH, ConfigNotFoundError, require_config
from table_validator.config.schema import (
    SourceType,
    SynapseAuthMode,
    ValidationType,
    ValidatorConfig,
)
from table_validator.connectors.azure_connector import AzureConnector, AzureSqlConnector
from table_validator.connectors.databricks_connector import DatabricksConnector
from table_validator.models import (
    AzureSqlValidationRequest,
    CatalogValidationRequest,
    CatalogValidationResponse,
    DataCompareMode,
    ValidationStatus,
    ValidationTier,
)
from table_validator.reports.excel_report import generate_excel_report
from table_validator.validators.blob_discovery import BlobCatalogValidator
from table_validator.validators.catalog_validator import CatalogValidator
from table_validator.validators.row_validator import AzureSqlValidator

app = typer.Typer(
    name="tablevalidator",
    help="Validate data migrations between Azure and Databricks Delta Lake.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "validation_report.xlsx"


def _configure_logging(verbose: bool, quiet: bool) -> None:
    """
    Print progress to the console as validation runs, instead of going
    silent for the entire duration of a large catalog-wide run (the
    validators already log per-schema/per-table progress via `logging`;
    without this, none of it reaches the console and a multi-minute run
    looks indistinguishable from a hang).

    Default: plain "Validating table 'x.y' ..." progress lines only.
    --verbose: also show the detailed [row-hash]/statistics diagnostic
    logs the validators already emit at INFO level.
    --quiet: suppress progress logging entirely (only the final summary
    prints) - useful for CI/scripted runs where log noise is unwanted.
    """
    logger = logging.getLogger("table_validator")
    logger.propagate = False

    # Each CLI invocation must start with a clean handler set: leftover
    # handlers from a previous invocation in the same process (e.g. every
    # test in this suite drives the CLI through the same interpreter) can
    # point at an already-closed/detached stream, causing "Logging error"
    # noise on emit rather than the intended output.
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)

    if quiet:
        # No handler at all (not even a raised level) - otherwise Python's
        # logging module falls back to its own stderr "lastResort" handler
        # for any record with no handler in its logger chain, which would
        # leak WARNING+ messages (e.g. missing/extra table notices)
        # despite --quiet asking for none of that.
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if not verbose:
        # Quiet the noisier tagged diagnostic logs by default; the plain
        # per-table progress line added in _validate_table still comes
        # through since it's logged at INFO on the same logger tree -
        # only these specific detailed diagnostics are filtered out.
        _VERBOSE_ONLY_TAGS = ("[row-hash]", "[compare_data]", "[data-mismatch]")

        class _HideVerboseDiagnostics(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                return not any(tag in message for tag in _VERBOSE_ONLY_TAGS)

        handler.addFilter(_HideVerboseDiagnostics())


@app.command()
def info() -> None:
    """Show what this tool does, which platforms it supports, and what
    each command is for - a quick orientation for first-time use."""
    typer.echo(
        "\n"
        "tablevalidator - data migration validator\n"
        "------------------------------------------\n"
        "Compares a source table against a target Databricks table/catalog "
        "and reports whether the migration is correct: matching schema, "
        "row counts, statistics, and (where a difference is found) the "
        "exact row/column that changed.\n"
        "\n"
        "Supported sources (target is always Databricks):\n"
        "  - Databricks catalog  -> Databricks catalog\n"
        "  - Azure Blob Storage  -> Databricks catalog\n"
        "  - Azure SQL Database  -> Databricks catalog\n"
        "\n"
        "Typical workflow, in order:\n"
        "\n"
        "  1. tablevalidator configure\n"
        "     Interactively set up credentials and which source/target "
        "table(s) to compare. Run this first, and again any time you "
        "need to change settings (source type, tables, primary key, "
        "which validations to run).\n"
        "\n"
        "  2. tablevalidator validate\n"
        "     Runs the comparison using the saved configuration and "
        "writes an Excel report (validation_report.xlsx by default). "
        "Prints a pass/fail summary in the terminal when it finishes.\n"
        "     To write to a different file instead of overwriting the "
        "default (e.g. if it's still open in Excel from a previous run):\n"
        "       tablevalidator validate --output validation_report_new.xlsx\n"
        "\n"
        "  3. tablevalidator open\n"
        "     Opens the most recently generated report in your default "
        "spreadsheet app (Excel/LibreOffice/etc.), so you can inspect "
        "exactly what passed, failed, or mismatched.\n"
        "\n"
        "Other commands:\n"
        "  tablevalidator report   Print the summary table from an "
        "existing report without opening it.\n"
        "\n"
        "Run 'tablevalidator <command> --help' for a command's full "
        "options (e.g. tablevalidator validate --help).\n"
        "\n"
        "-MS-"
    )


@app.command()
def configure() -> None:
    """Interactively configure Azure and Databricks credentials."""
    run_configure_wizard()


@app.command()
def validate(
    config_path: Path = typer.Option(
        CONFIG_PATH,
        "--config-path",
        help="Path to config.yaml (default: ~/.table_validator/config.yaml).",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_PATH,
        "--output",
        help="Path to write the Excel validation report to.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show detailed per-table row-hash/statistics diagnostic logs.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress progress logging entirely (only the final summary prints).",
    ),
    mode: str = typer.Option(
        "full",
        "--mode",
        help=(
            "Databricks-to-Databricks only: how far the tiered fail-fast "
            "funnel is allowed to go. 'stats' stops after row count/null/"
            "distinct/min-max statistics (never runs a fingerprint or "
            "row-hash comparison, even on a match). 'full' (default) lets "
            "the funnel continue through the whole-table fingerprint and, "
            "if that disagrees, row-hash/column-level diff - but only if "
            "cheaper tiers couldn't already prove the tables equal or "
            "different. Ignored for azure_blob/azure_sql source types."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help=(
            "Never prompt interactively. A large table (Databricks-to-"
            "Databricks only) with a confirmed mismatch is normally "
            "offered a partition-column choice before row-hash "
            "comparison; --yes skips that prompt and always compares the "
            "whole table unpartitioned instead. Use for CI/non-interactive "
            "runs - a run with no attached terminal already skips the "
            "prompt on its own, but --yes makes that explicit."
        ),
    ),
    skip_category_summary: bool = typer.Option(
        False,
        "--skip-category-summary",
        help=(
            "Omit the 'Mismatch Categories' sheet (root-cause "
            "classification + plain-English summary of Data Mismatches) "
            "from the Excel report. Every other sheet, including Data "
            "Mismatches itself, is unaffected - this recovers the exact "
            "report shape from before the mismatch-categorization "
            "feature existed, for anyone who doesn't want it."
        ),
    ),
) -> None:
    """Run validation and produce an Excel validation report."""

    if mode not in ("stats", "full"):
        typer.secho(f"Invalid --mode '{mode}' - must be 'stats' or 'full'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Print progress as validation runs (which table it's on, etc.) -
    # without this, a large catalog-wide run goes completely silent
    # until the final summary, indistinguishable from a hang.
    _configure_logging(verbose=verbose, quiet=quiet)

    # ------------------------------------------------------------------
    # 1. Load config. require_config() is the single place that decides
    # "config is missing" - anything else (e.g. the wizard's own
    # load_config() pre-populate call) is a different, legitimate use of
    # a missing file and must not duplicate this check.
    # ------------------------------------------------------------------
    try:
        config = require_config(config_path)
    except ConfigNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    missing = _missing_config_fields(config)
    if missing:
        typer.secho(
            "Config is incomplete - missing: " + ", ".join(missing) + ". "
            "Run 'tablevalidator configure' to fill these in.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # 2. Load secrets from ~/.table_validator/.env via the auth
    # abstraction, and 3. build the Databricks connector from them - every
    # source_type targets Databricks, so this connector is always needed.
    # ------------------------------------------------------------------
    token = get_databricks_token(config)
    if not token:
        typer.secho(
            "No Databricks token found in ~/.table_validator/.env. "
            "Run 'tablevalidator configure' first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        databricks = DatabricksConnector(
            host=host_from_workspace_url(config.databricks.workspace_url),
            token=token,
            http_path=config.databricks.http_path,
        )
    except ValueError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    azure_credential = get_azure_credential(config)

    scope_desc = _describe_scope(config)
    typer.echo(f"Validating ({config.source_type.value}) {scope_desc} ...")

    # ------------------------------------------------------------------
    # 4/5/6. Determine scope and run the source-type-specific comparison
    # path. Each branch resolves its own connector(s) and calls its own
    # validator, but all three converge on the same CatalogValidationResponse
    # shape, so report generation and summary printing (7/8) are identical
    # regardless of source_type.
    # ------------------------------------------------------------------
    try:
        if config.source_type == SourceType.AZURE_BLOB:
            result = _run_blob_validation(config, azure_credential, databricks)
        elif config.source_type == SourceType.AZURE_SQL:
            result = _run_sql_validation(config, azure_credential, databricks)
        elif config.source_type == SourceType.SYNAPSE:
            result = _run_synapse_validation(config, azure_credential, databricks)
        else:
            result = _run_databricks_validation(config, databricks, mode=mode, yes=yes)
    except ValueError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # 7. Generate the Excel report (one row per table across every
    # matched schema, already aggregated onto a single
    # CatalogValidationResponse). generate_excel_report itself doesn't
    # create the parent directory (openpyxl's wb.save() requires it to
    # already exist), so --output pointing at a not-yet-created directory
    # is handled here rather than failing with a raw FileNotFoundError.
    # ------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    # enabled_validations only actually gates behavior for source_type ==
    # databricks today (CatalogValidator) - azure_blob/azure_sql always
    # run their full fixed pipeline regardless of config.validations, so
    # filtering their report would hide data that was genuinely computed.
    report_enabled_validations = (
        {v.value for v in config.validations}
        if config.source_type == SourceType.DATABRICKS
        else None
    )
    try:
        generate_excel_report(
            result, str(output),
            source_type=config.source_type.value,
            enabled_validations=report_enabled_validations,
            skip_category_summary=skip_category_summary,
        )
    except PermissionError:
        typer.secho(
            f"Could not write '{output}' - the file is open in Excel (or "
            "another program) and locked for writing. Close it and re-run "
            "'tablevalidator validate', or pass a different path with "
            "--output.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # 8. Print a summary - per-table PASS/FAIL lines, then the same
    # aggregate summary table used by `tablevalidator report` and by the
    # Excel report's own Summary sheet (built from the same
    # _build_summary_metrics() call), so the two never disagree.
    # ------------------------------------------------------------------
    _print_summary(result, config, output)

    validations_run_str = (
        ", ".join(sorted(report_enabled_validations))
        if report_enabled_validations is not None
        else None
    )
    print_summary_table(
        summary_from_response(result, config.source_type.value, validations_run_str)
    )
    typer.echo(f"Report written to: {output.resolve()}")
    typer.echo("Run 'tablevalidator open' to open it now.")

    if result.status in (ValidationStatus.FAIL, ValidationStatus.ERROR):
        raise typer.Exit(code=1)


@app.command()
def report(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help=(
            "Path to a validation_report.xlsx to read. Defaults to "
            "./validation_report.xlsx (the same default 'validate' writes to)."
        ),
    ),
) -> None:
    """Print the summary table from the most recently generated (or a
    specified) validation report."""
    report_path = (path or Path(DEFAULT_OUTPUT_PATH)).resolve()

    if not report_path.exists():
        typer.secho(
            "No validation report found. Run 'tablevalidator validate' first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        data = summary_from_excel(report_path)
    except Exception as exc:
        typer.secho(f"Unable to read report at {report_path}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_summary_table(data)
    # str(report_path) on Windows already uses backslashes consistently
    # (Path.resolve() normalizes separators for the current OS), so this
    # prints an absolute, OS-native path that's easy to Ctrl+click or
    # copy straight into Explorer/another terminal.
    typer.echo(f"Report file: {report_path}")


@app.command(name="open")
def open_report_command(
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        help=(
            "Path to a validation_report.xlsx to open. Defaults to "
            "./validation_report.xlsx (the same default 'validate' writes to)."
        ),
    ),
) -> None:
    """Open the most recently generated (or a specified) validation
    report in the OS's default application (Excel/LibreOffice/etc.)."""
    report_path = (path or Path(DEFAULT_OUTPUT_PATH)).resolve()

    if not report_path.exists():
        typer.secho(
            "No validation report found. Run 'tablevalidator validate' first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    _open_in_default_app(report_path)
    typer.echo(f"Opening: {report_path}")


# ---------------------------------------------------------------------------
# Source-type-specific comparison paths
# ---------------------------------------------------------------------------
def _run_databricks_validation(
    config: ValidatorConfig,
    databricks: DatabricksConnector,
    mode: str = "full",
    yes: bool = False,
) -> CatalogValidationResponse:
    """source_type == databricks: source is another Databricks catalog.
    A blank schema/table on EITHER side means "compare everything
    matching" for that level - CatalogValidator already performs the
    list-and-intersect discovery internally whenever request.schemas/
    tables is left unrestricted (None), so leaving the restriction off
    IS the discovery trigger; there is no separate discovery step to call.

    If the user explicitly named BOTH a source and target schema/table,
    compare that exact pair directly - schema_map/table_map bypass
    name-based matching entirely, so the two sides don't need to share a
    name (a typo, a rename, different casing). schemas_restriction/
    tables_restriction scope by the SOURCE-side name, matching
    CatalogValidator._validate_schema/compare_catalogs' filtering
    convention (mirrors _run_sql_validation's identical pattern below).
    When names are identical (today's common case) schema_map/table_map
    end up empty and behavior is unchanged."""
    schema_map: Dict[str, str] = {}
    schemas_restriction = None
    if config.source_table.schema_name and config.target_table.schema_name:
        schemas_restriction = [config.source_table.schema_name]
        if config.source_table.schema_name.lower() != config.target_table.schema_name.lower():
            schema_map = {config.source_table.schema_name: config.target_table.schema_name}

    table_map: Dict[str, str] = {}
    tables_restriction = None
    if config.source_table.table and config.target_table.table:
        tables_restriction = [config.source_table.table]
        if config.source_table.table.lower() != config.target_table.table.lower():
            table_map = {config.source_table.table: config.target_table.table}
    else:
        # Schema-wide sweep (no single table named on both sides) - apply
        # the wizard's optional only_tables allowlist / table_map renaming
        # for this schema, if the user set either. Both are no-ops
        # (tables_restriction stays None, table_map stays {}) unless the
        # user actually configured them, so a plain schema sweep behaves
        # exactly as before this feature existed.
        if config.only_tables:
            tables_restriction = list(config.only_tables)
        if config.table_map:
            table_map = dict(config.table_map)

    # A configured primary key only applies to the single named table
    # (not a catalog-wide sweep) - CatalogValidator looks it up by
    # "schema.table" first, falling back to a bare table name, so provide
    # both forms when a schema is known; falls back to row-number matching
    # as before when unset. Keyed by the TARGET-side name, since that's
    # what _lookup_primary_key resolves against.
    primary_keys: Dict[str, list] = {}
    if config.primary_key and config.target_table.table:
        primary_keys[config.target_table.table] = config.primary_key
        if config.target_table.schema_name:
            key = f"{config.target_table.schema_name}.{config.target_table.table}"
            primary_keys[key] = config.primary_key

    max_tier = ValidationTier.STATISTICAL if mode == "stats" else ValidationTier.COLUMN_DIFF

    request = CatalogValidationRequest(
        source_catalog=config.source_table.catalog or "",
        target_catalog=config.target_table.catalog or "",
        schemas=schemas_restriction,
        schema_map=schema_map,
        tables=tables_restriction,
        table_map=table_map,
        enabled_validations=set(config.validations),
        primary_keys=primary_keys,
        max_tier=max_tier,
        only_columns=config.only_columns,
        ignore_columns=config.ignore_columns,
        ignore_datatype_columns=config.ignore_datatype_columns,
        column_map=config.column_map,
    )

    # Row-filter predicate - lives purely as connector instance state
    # (BaseSqlConnector.set_row_filters/_scoped_table), same mechanism as
    # the notebook-native validate_tables() API, not a CatalogValidationRequest
    # field. A no-op unless the user configured at least one of the three.
    if config.row_filter or config.source_row_filter or config.target_row_filter:
        databricks.set_row_filters(
            common=config.row_filter,
            source=config.source_row_filter,
            target=config.target_row_filter,
        )

    partition_prompt = build_partition_prompt(yes=yes)
    validator = CatalogValidator(databricks, partition_prompt=partition_prompt)
    return validator.compare_catalogs(request)


def _run_blob_validation(
    config: ValidatorConfig,
    azure_credential,
    databricks: DatabricksConnector,
) -> CatalogValidationResponse:
    """source_type == azure_blob: discover blobs matching folder_prefix/
    file_pattern, match them to Databricks tables by inferred filename,
    and compare each matched pair (row count + column name/type only -
    see validators/blob_discovery.py for why row-hash comparison is out
    of scope for this multi-blob-match path)."""
    if not azure_credential.storage_account_key:
        raise ValueError(
            "No Azure Storage account key found in ~/.table_validator/.env. "
            "Run 'tablevalidator configure' first."
        )
    if not config.azure.storage_account or not config.blob_source.container:
        raise ValueError(
            "azure.storage_account and blob_source.container are required "
            "for an Azure Blob source. Run 'tablevalidator configure' first."
        )

    azure = AzureConnector(
        account_name=config.azure.storage_account,
        account_key=azure_credential.storage_account_key,
        container_name=config.blob_source.container,
    )

    validator = BlobCatalogValidator(azure, databricks)
    return validator.validate(
        target_catalog=config.target_table.catalog or "",
        # None (not "") when left blank - BlobCatalogValidator treats
        # None as "match blobs against every schema in the catalog".
        target_schema=config.target_table.schema_name,
        folder_prefix=config.blob_source.folder_prefix,
        file_pattern=config.blob_source.file_pattern,
        # If BOTH an exact blob and target table are named, compare that
        # pair directly - bypasses filename-to-table discovery entirely,
        # even if the names don't match.
        blob_path=config.blob_source.blob_path,
        target_table=config.target_table.table,
    )


def _run_sql_validation(
    config: ValidatorConfig,
    azure_credential,
    databricks: DatabricksConnector,
) -> CatalogValidationResponse:
    """source_type == azure_sql: structurally identical to the Databricks
    catalog-to-catalog path (schema discovery, table discovery, row/
    column comparison) - AzureSqlValidator already implements this
    against AzureSqlConnector's schema/table listing methods, so it's
    reused directly rather than duplicated."""
    if not azure_credential.sql_username or not azure_credential.sql_password:
        raise ValueError(
            "No Azure SQL username/password found in ~/.table_validator/.env. "
            "Run 'tablevalidator configure' first."
        )
    if not config.azure.sql_server or not config.azure.sql_database:
        raise ValueError(
            "azure.sql_server and azure.sql_database are required for an "
            "Azure SQL source. Run 'tablevalidator configure' first."
        )

    azure_sql = AzureSqlConnector(
        server=config.azure.sql_server,
        database=config.azure.sql_database,
        username=azure_credential.sql_username,
        password=azure_credential.sql_password,
    )

    # If the user explicitly named BOTH a source (SQL) and target
    # (Databricks) schema/table, compare that exact pair directly -
    # schema_map/table_map bypass name-based matching entirely, so the
    # two sides don't need to share a name (e.g. SQL's 'dbo' vs a
    # purpose-named Databricks schema). schemas/tables restrict by the
    # SOURCE (Azure SQL) name, since that's what _compare_schemas/
    # _compare_tables filter against.
    schema_map: Dict[str, str] = {}
    schemas_restriction = None
    if config.sql_source.schema_name and config.target_table.schema_name:
        schema_map = {config.sql_source.schema_name: config.target_table.schema_name}
        schemas_restriction = [config.sql_source.schema_name]

    table_map: Dict[str, str] = {}
    tables_restriction = None
    if config.sql_source.table and config.target_table.table:
        table_map = {config.sql_source.table: config.target_table.table}
        tables_restriction = [config.sql_source.table]

    # Mirrors _run_databricks_validation's primary_keys construction: a
    # configured key only applies to the single named table (not a
    # catalog-wide sweep), looked up by "schema.table" first, falling
    # back to a bare table name. Without this, config.primary_key was
    # silently dropped and Data Mismatches could never populate even
    # when FULL mode found real mismatches.
    primary_keys: Dict[str, list] = {}
    if config.primary_key and config.target_table.table:
        primary_keys[config.target_table.table] = config.primary_key
        if config.target_table.schema_name:
            key = f"{config.target_table.schema_name}.{config.target_table.table}"
            primary_keys[key] = config.primary_key

    # FULL mode is what actually populates Data Mismatches/row-level
    # detail (see AzureSqlValidator's `mode == DataCompareMode.FULL and
    # mismatch_count > 0 and not using_row_number_fallback` gate) - the
    # model's own default (STATISTICS) never triggers that path, so this
    # was silently unreachable from the CLI before.
    request = AzureSqlValidationRequest(
        target_catalog=config.target_table.catalog or "",
        schemas=schemas_restriction,
        schema_map=schema_map,
        tables=tables_restriction,
        table_map=table_map,
        primary_keys=primary_keys,
        data_compare_mode=DataCompareMode.FULL,
    )

    validator = AzureSqlValidator(azure_sql, databricks)
    return validator.validate(request)


def _run_synapse_validation(
    config: ValidatorConfig,
    azure_credential,
    databricks: DatabricksConnector,
) -> CatalogValidationResponse:
    """source_type == synapse: Azure Synapse SQL pool (dedicated or
    serverless) against a Databricks catalog. Synapse SQL is
    protocol-identical T-SQL over the same ODBC driver Azure SQL Database
    uses, so this reuses AzureSqlConnector/AzureSqlValidator UNCHANGED,
    just constructed with the Synapse server/database/credentials instead
    of an Azure SQL DB's - no new connector or validator class, mirrors
    _run_sql_validation's logic exactly (see that function for the
    schema_map/table_map/primary_keys/FULL-mode rationale, identical
    here).

    Auth is the one place Synapse genuinely diverges from the Azure SQL
    path: config.azure.synapse_auth_mode selects between a SQL login
    (SYNAPSE_USERNAME/SYNAPSE_PASSWORD) and a Microsoft Entra ID service
    principal (azure.tenant_id + azure.synapse_client_id +
    SYNAPSE_CLIENT_SECRET). Entra is what a workspace with SQL
    authentication disabled requires - the connector fetches an access
    token and passes it via the ODBC access-token attribute rather than
    sending UID/PWD at all.

    Not yet verified against a live Synapse SQL pool for pool-specific
    T-SQL quirks (dedicated pools have historically had some system-view/
    HASHBYTES-adjacent gaps vs. Azure SQL DB) - flagged here rather than
    assumed away; report back if a specific query fails against your pool
    and it can be special-cased in AzureSqlConnector the same way
    decimal_as_integer/float formatting already are for Databricks
    compatibility.
    """
    if not config.azure.synapse_server or not config.azure.synapse_database:
        raise ValueError(
            "azure.synapse_server and azure.synapse_database are required "
            "for a Synapse source. Run 'tablevalidator configure' first."
        )

    if config.azure.synapse_auth_mode == SynapseAuthMode.ENTRA_SERVICE_PRINCIPAL:
        missing_entra = []
        if not config.azure.tenant_id:
            missing_entra.append("azure.tenant_id (in config.yaml)")
        if not config.azure.synapse_client_id:
            missing_entra.append("azure.synapse_client_id (in config.yaml)")
        if not azure_credential.synapse_client_secret:
            missing_entra.append("SYNAPSE_CLIENT_SECRET (in ~/.table_validator/.env)")
        if missing_entra:
            raise ValueError(
                "Synapse is configured for Entra service-principal auth but "
                "these are missing: " + ", ".join(missing_entra)
                + ". Run 'tablevalidator configure' first."
            )

        synapse = AzureSqlConnector(
            server=config.azure.synapse_server,
            database=config.azure.synapse_database,
            tenant_id=config.azure.tenant_id,
            client_id=config.azure.synapse_client_id,
            client_secret=azure_credential.synapse_client_secret,
        )
    else:
        if not azure_credential.synapse_username or not azure_credential.synapse_password:
            raise ValueError(
                "No Synapse username/password found in ~/.table_validator/.env. "
                "Run 'tablevalidator configure' first."
            )

        synapse = AzureSqlConnector(
            server=config.azure.synapse_server,
            database=config.azure.synapse_database,
            username=azure_credential.synapse_username,
            password=azure_credential.synapse_password,
        )

    schema_map: Dict[str, str] = {}
    schemas_restriction = None
    if config.synapse_source.schema_name and config.target_table.schema_name:
        schema_map = {config.synapse_source.schema_name: config.target_table.schema_name}
        schemas_restriction = [config.synapse_source.schema_name]

    table_map: Dict[str, str] = {}
    tables_restriction = None
    if config.synapse_source.table and config.target_table.table:
        table_map = {config.synapse_source.table: config.target_table.table}
        tables_restriction = [config.synapse_source.table]

    primary_keys: Dict[str, list] = {}
    if config.primary_key and config.target_table.table:
        primary_keys[config.target_table.table] = config.primary_key
        if config.target_table.schema_name:
            key = f"{config.target_table.schema_name}.{config.target_table.table}"
            primary_keys[key] = config.primary_key

    request = AzureSqlValidationRequest(
        target_catalog=config.target_table.catalog or "",
        schemas=schemas_restriction,
        schema_map=schema_map,
        tables=tables_restriction,
        table_map=table_map,
        primary_keys=primary_keys,
        data_compare_mode=DataCompareMode.FULL,
    )

    validator = AzureSqlValidator(synapse, databricks)
    return validator.validate(request)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _missing_config_fields(config: ValidatorConfig) -> list:
    """
    Only the Databricks connection, the target catalog, and whatever the
    selected source_type hard-requires are checked here. schema_name/
    table being blank (on either source or target) is intentionally
    optional everywhere (Phase 4 Part 2): it means "compare everything
    matching" rather than "config incomplete".
    """
    missing = []
    if not config.databricks.workspace_url:
        missing.append("databricks.workspace_url")
    if not config.databricks.http_path:
        missing.append("databricks.http_path")
    if not config.target_table.catalog:
        missing.append("target_table.catalog")

    if config.source_type == SourceType.AZURE_BLOB:
        if not config.azure.storage_account:
            missing.append("azure.storage_account")
        if not config.blob_source.container:
            missing.append("blob_source.container")
    elif config.source_type == SourceType.AZURE_SQL:
        if not config.azure.sql_server:
            missing.append("azure.sql_server")
        if not config.azure.sql_database:
            missing.append("azure.sql_database")
    elif config.source_type == SourceType.SYNAPSE:
        if not config.azure.synapse_server:
            missing.append("azure.synapse_server")
        if not config.azure.synapse_database:
            missing.append("azure.synapse_database")
    else:
        if not config.source_table.catalog:
            missing.append("source_table.catalog")

    return missing


def _open_in_default_app(path: Path) -> None:
    """
    Launch `path` in whatever application the OS has associated with its
    file extension (Excel/LibreOffice/etc. for .xlsx) - so a user running
    this as an installed package gets the report opened for them instead
    of having to go find the file themselves. Best-effort only: any
    failure (no GUI, no associated app, sandboxed environment) is logged
    and swallowed rather than failing the whole `validate` run, since the
    report was already written successfully at this point.
    """
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except Exception as exc:
        logger.debug("Could not auto-open report at %s: %s", path, exc)
        typer.echo(
            f"(Could not open the report automatically - open it manually: {path})"
        )


def _describe_scope(config: ValidatorConfig) -> str:
    """Human-readable description of what's being compared, for the
    'Validating ...' progress line - reflects blank schema/table as
    'all schemas'/'all tables' rather than printing an empty segment."""

    def _side(ref) -> str:
        catalog = ref.catalog or "?"
        if not ref.schema_name:
            return f"{catalog} (all schemas)"
        if not ref.table:
            return f"{catalog}.{ref.schema_name} (all tables)"
        return f"{catalog}.{ref.schema_name}.{ref.table}"

    target_desc = _side(config.target_table)

    if config.source_type == SourceType.AZURE_BLOB:
        container = config.blob_source.container or "?"
        scope_bits = []
        if config.blob_source.folder_prefix:
            scope_bits.append(config.blob_source.folder_prefix)
        if config.blob_source.file_pattern:
            scope_bits.append(config.blob_source.file_pattern)
        scope_desc = f" ({', '.join(scope_bits)})" if scope_bits else ""
        source_desc = f"blob:{container}{scope_desc}"
    elif config.source_type == SourceType.AZURE_SQL:
        database = config.azure.sql_database or "?"
        if not config.sql_source.schema_name:
            source_desc = f"sql:{database} (all schemas)"
        elif not config.sql_source.table:
            source_desc = f"sql:{database}.{config.sql_source.schema_name} (all tables)"
        else:
            source_desc = f"sql:{database}.{config.sql_source.schema_name}.{config.sql_source.table}"
    elif config.source_type == SourceType.SYNAPSE:
        database = config.azure.synapse_database or "?"
        if not config.synapse_source.schema_name:
            source_desc = f"synapse:{database} (all schemas)"
        elif not config.synapse_source.table:
            source_desc = f"synapse:{database}.{config.synapse_source.schema_name} (all tables)"
        else:
            source_desc = (
                f"synapse:{database}.{config.synapse_source.schema_name}."
                f"{config.synapse_source.table}"
            )
    else:
        source_desc = _side(config.source_table)

    return f"{source_desc} -> {target_desc}"


def _print_summary(result, config: ValidatorConfig, output: Path) -> None:
    total = passed = failed = errors = skipped = 0
    per_table_lines = []

    for schema in result.schemas:
        for table in schema.tables:
            total += 1
            if table.status == ValidationStatus.PASS:
                passed += 1
            elif table.status == ValidationStatus.ERROR:
                errors += 1
            elif table.status == ValidationStatus.SKIPPED:
                skipped += 1
            else:
                failed += 1
            per_table_lines.append(
                (f"{schema.schema_name}.{table.table}", table.status)
            )

    status_color = {
        ValidationStatus.PASS: typer.colors.GREEN,
        ValidationStatus.FAIL: typer.colors.RED,
        ValidationStatus.ERROR: typer.colors.YELLOW,
        ValidationStatus.SKIPPED: typer.colors.WHITE,
    }

    typer.echo("")
    if total > 1:
        typer.echo("Per-table results:")
        for name, status in per_table_lines:
            typer.secho(
                f"  {name}: {status.value}",
                fg=status_color.get(status, typer.colors.WHITE),
            )
        typer.echo("")

    overall_color = status_color.get(result.status, typer.colors.WHITE)
    typer.secho(f"Overall status: {result.status.value}", fg=overall_color, bold=True)
    typer.echo(f"Tables: {total} total, {passed} passed, {failed} failed, "
               f"{errors} error, {skipped} skipped")
    if result.error:
        typer.secho(f"Error: {result.error}", fg=typer.colors.RED)

    enabled = ", ".join(v.value for v in config.validations) or "none"
    typer.echo(f"Validations requested: {enabled}")
    if ValidationType.ROW not in config.validations and config.source_type != SourceType.DATABRICKS:
        typer.echo(
            "Note: for this source type, row-level comparison always runs "
            "when a table exists (it cannot be disabled independently in "
            "that pipeline yet); 'row' being unselected is informational only."
        )


if __name__ == "__main__":
    app()
