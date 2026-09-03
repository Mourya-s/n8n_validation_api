"""Interactive configuration wizard for `tablevalidator configure`.

Walks the user through: what's being compared (source type), Databricks
credentials (always needed - the target is always a Databricks catalog),
source-specific credentials/scoping, target table details, and which
validations to run. Non-secret answers are saved into ValidatorConfig via
config/manager.py; secrets are written to ~/.table_validator/.env with
owner-only file permissions.

Phase 1 auth only: credentials are entered manually here and read back by
auth/azure_auth.py and auth/databricks_auth.py. Nothing else in the
codebase should read these credentials directly.

Every free-text answer goes through _ask()/_ask_secret(), which strip()
whitespace and normalize "" to None, so no field ever silently saves a
value the user didn't actually type (this is also where the earlier
' for_schema_validation' leading-space bug and the
myserver.database.windows.net/mydb placeholder-default bug were fixed -
both were free-text prompts that skipped this normalization).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Dict, Optional

import questionary
import typer

from table_validator.config.manager import CONFIG_PATH, load_config, save_config
from table_validator.config.schema import (
    SourceType,
    SynapseAuthMode,
    ValidationType,
    ValidatorConfig,
)

ENV_PATH = Path.home() / ".table_validator" / ".env"

_ALL_VALIDATIONS = [
    (ValidationType.CATALOG, "catalog"),
    (ValidationType.SCHEMA, "schema"),
    (ValidationType.COLUMN, "column"),
    (ValidationType.ROW, "row"),
]

# Wizard-only convenience choice, not a ValidationType and never itself
# stored in config.yaml - selecting it (alone or alongside individual
# choices) resolves to the full validations list at answer time. See
# _resolve_validation_selection().
_ALL_LABEL = "All"

# What the "Which validations should run?" checkbox actually shows.
# catalog/schema are always presented (and toggled) as a single combined
# option - CatalogValidator's catalog-existence check is a prerequisite
# for the schema stage anyway (there's nothing meaningful to check at
# schema level if the catalog itself wasn't found), and in practice
# users never want one without the other. column/row stay their own
# checkboxes since deselecting either has a real, distinct effect
# (skipping column-level or row-level checks/report sheets). Each group
# is (label, [ValidationTypes it resolves to]); _resolve_validation_selection
# expands a checked group label back into its full ValidationType set.
_VALIDATION_GROUPS = [
    ("catalog & schema", [ValidationType.CATALOG, ValidationType.SCHEMA]),
    ("column", [ValidationType.COLUMN]),
    ("row", [ValidationType.ROW]),
]

# What's being compared, shown as the wizard's first question. Labels are
# wizard-only display text; SourceType is what's actually stored.
_SOURCE_TYPE_CHOICES = [
    (SourceType.DATABRICKS, "Databricks catalog -> Databricks catalog"),
    (SourceType.AZURE_BLOB, "Azure Blob Storage -> Databricks catalog"),
    (SourceType.AZURE_SQL, "Azure SQL Database -> Databricks catalog"),
    (SourceType.SYNAPSE, "Azure Synapse SQL pool -> Databricks catalog"),
]

# Synapse auth-mode choices. Wizard-only display text; SynapseAuthMode is
# what's actually stored on config.azure.synapse_auth_mode.
_SYNAPSE_AUTH_SQL_LABEL = "SQL login (username + password)"
_SYNAPSE_AUTH_ENTRA_LABEL = (
    "Microsoft Entra ID service principal (tenant + client ID + secret)"
)


def _ask(prompt: questionary.Question) -> Optional[str]:
    """
    Resolve a questionary text/password prompt, stripping surrounding
    whitespace and normalizing a blank answer to None.

    This is the single choke point every free-text prompt in this wizard
    goes through, specifically so "leave blank" and "stray whitespace"
    are handled consistently everywhere instead of per-call-site (which
    is how both bugs this fixes originally slipped through - one call
    site normalized, the next one didn't).
    """
    answer = prompt.ask()
    if answer is None:
        return None
    stripped = answer.strip()
    return stripped or None


def _prompt_table_ref(label: str, existing) -> Dict[str, Optional[str]]:
    typer.echo(f"\n{label} table:")
    catalog = _ask(questionary.text("  Catalog:", default=existing.catalog or ""))
    schema_name = _ask(
        questionary.text(
            "  Schema (leave blank to compare all schemas in this catalog):",
            default=existing.schema_name or "",
        )
    )
    table = _ask(
        questionary.text(
            "  Table (leave blank to compare all tables in this schema):",
            default=existing.table or "",
        )
    )
    # Keyed by field name ("schema_name"), not the "schema" alias -
    # model_copy(update=...) matches by field name and silently ignores
    # unknown keys, so using the alias here would drop the schema value.
    return {"catalog": catalog, "schema_name": schema_name, "table": table}


def _prompt_primary_key(existing: Optional[list]) -> Optional[list]:
    """
    Optional primary/business key for the single named source/target
    table - only asked when both are set to a specific table (not a
    catalog-wide sweep). If left blank, row-level comparison falls back
    to a synthetic ROW_NUMBER() match as before; a real key is cheaper
    (no full-table sort) and avoids the row-number fallback's known
    timeout risk on large tables.
    """
    default_str = ", ".join(existing) if existing else ""
    answer = _ask(
        questionary.text(
            "Primary key column(s) for this table, comma-separated "
            "(optional - leave blank to match rows by row-number instead):",
            default=default_str,
        )
    )
    if not answer:
        return None
    return [col.strip() for col in answer.split(",") if col.strip()]


def _prompt_column_list(prompt_text: str, existing: Optional[list]) -> Optional[list]:
    """Shared free-text parser for a comma-separated column list answer -
    used by all three customization sub-options below. Returns None for
    a blank answer (caller decides the actual default: None vs [])."""
    default_str = ", ".join(existing) if existing else ""
    answer = _ask(questionary.text(prompt_text, default=default_str))
    if not answer:
        return None
    return [col.strip() for col in answer.split(",") if col.strip()]


def _prompt_column_customization(config: ValidatorConfig) -> None:
    """Optional column-level customization, asked right after the
    primary key - only meaningful for the single named table
    (primary_key's same scope). Skipped entirely (leaving any existing
    only_columns/ignore_columns/ignore_datatype_columns untouched) unless
    the user opts in, so a user who never touches this gets identical
    behavior to before this feature existed."""
    customize = questionary.confirm(
        "Customize column validation? (skip specific columns, compare "
        "only specific columns, or ignore datatype mismatches for "
        "specific columns)",
        default=False,
    ).ask()

    if not customize:
        return

    config.only_columns = _prompt_column_list(
        "Compare ONLY these columns, comma-separated (leave blank to "
        "compare every common column as usual):",
        config.only_columns,
    )
    config.ignore_columns = _prompt_column_list(
        "SKIP these columns entirely, comma-separated (leave blank to "
        "skip none):",
        config.ignore_columns,
    ) or []
    config.ignore_datatype_columns = _prompt_column_list(
        "Ignore DATATYPE mismatches only for these columns, "
        "comma-separated - their other checks (nullable, statistics, "
        "row values) still run (leave blank to skip none):",
        config.ignore_datatype_columns,
    ) or []


def _prompt_row_filter(config: ValidatorConfig) -> None:
    """Optional row-filter predicate, asked right after column
    customization - same single-table scope, Databricks source only
    (AzureSqlConnector has no row-filter mechanism, unlike
    BaseSqlConnector's set_row_filters/_scoped_table that the Databricks
    connector and the notebook-native validate_tables() API both share).
    Skipped entirely (leaving any existing row_filter/source_row_filter/
    target_row_filter untouched) unless the user opts in, so declining
    gives identical behavior to before this feature existed."""
    customize = questionary.confirm(
        "Filter to a subset of rows? (only rows matching a condition are "
        "compared, e.g. \"id > 10 and id < 30\" or \"gender = 'male'\")",
        default=False,
    ).ask()

    if not customize:
        return

    same_condition = questionary.confirm(
        "Apply the same condition to both source and target?",
        default=True,
    ).ask()

    # Mutually exclusive in this UI - picking one must clear whatever
    # stale value the OTHER carried over from an earlier configure run,
    # same convention as _prompt_schema_scoping below.
    if same_condition:
        config.row_filter = _ask(
            questionary.text(
                "Row filter condition (SQL WHERE-fragment):",
                default=config.row_filter or "",
            )
        ) or None
        config.source_row_filter = None
        config.target_row_filter = None
    else:
        config.row_filter = None
        config.source_row_filter = _ask(
            questionary.text(
                "Source row filter condition:",
                default=config.source_row_filter or "",
            )
        ) or None
        config.target_row_filter = _ask(
            questionary.text(
                "Target row filter condition:",
                default=config.target_row_filter or "",
            )
        ) or None


def _prompt_schema_scoping(config: ValidatorConfig) -> None:
    """
    Optional schema-wide scoping, asked only when a schema is named on
    both sides but no specific table is (a schema-wide sweep - the
    single-table case already covers table renaming by naming both
    source_table.table/target_table.table directly). Lets the user
    either restrict the sweep to a plain allowlist of table names, or
    map individual renamed tables by hand as plain 'source=target' text
    pairs - deliberately kept as simple free-text entry (no live
    Databricks picker, unlike column_map's picker), mirroring the
    existing --table-map CLI flag's own plain-text convention.

    Skipped entirely (leaving only_tables/table_map untouched) unless
    the user opts in, so declining gives identical behavior to before
    this feature existed.
    """
    customize = questionary.confirm(
        "Customize this schema? (validate only specific tables, or map "
        "renamed tables)",
        default=False,
    ).ask()

    if not customize:
        return

    choice = questionary.select(
        "What do you want to do?",
        choices=[
            "Select specific tables to validate",
            "Map renamed tables (source name -> target name)",
        ],
    ).ask()

    # The two choices are mutually exclusive in this UI - picking one must
    # clear whatever stale value the OTHER one carried over from an
    # earlier configure run, or a leftover only_tables allowlist from a
    # previous session silently restricts/breaks a table_map set just
    # now (and vice versa). Without this, a user re-running configure to
    # switch from "select tables" to "map tables" keeps the old
    # only_tables value fighting with their new mapping.
    if choice == "Select specific tables to validate":
        config.only_tables = _prompt_column_list(
            "Table names to validate, comma-separated (leave blank to "
            "validate every common table in this schema as usual):",
            config.only_tables,
        )
        config.table_map = {}
    elif choice == "Map renamed tables (source name -> target name)":
        default_str = ", ".join(
            f"{src}={tgt}" for src, tgt in (config.table_map or {}).items()
        )
        answer = _ask(
            questionary.text(
                "Renamed table pairs, as source=target, comma-separated "
                "(e.g. cust=customers, ord=orders):",
                default=default_str,
            )
        )
        new_map: Dict[str, str] = {}
        if answer:
            for pair in answer.split(","):
                pair = pair.strip()
                if not pair or "=" not in pair:
                    continue
                src, _, tgt = pair.partition("=")
                src, tgt = src.strip(), tgt.strip()
                if src and tgt:
                    new_map[src] = tgt
        config.table_map = new_map
        config.only_tables = None


def _prompt_column_mapping(config: ValidatorConfig, secrets: Dict[str, str]) -> None:
    """
    Optional column-name mapping for the single named table (same scope
    as primary_key/column customization) - lets the user pair up
    individual columns that were renamed between source and target (e.g.
    source has 'cust_id', target has 'customer_id').

    Connects to Databricks LIVE (the first time this wizard ever does so
    during `configure`, rather than only at `validate` time) to fetch
    both tables' real column lists, so the picker can be built from
    actual columns rather than blind free-text entry. Any failure along
    the way (missing/bad credentials, network issue, wrong table name,
    insufficient permissions) is caught broadly and degrades to silently
    skipping this step - `configure` must never crash just because this
    optional, nice-to-have step couldn't reach Databricks. secrets may
    not yet contain a freshly-typed token if the user is configuring for
    the first time in this same run, so DATABRICKS_TOKEN is checked
    there first, falling back to whatever's already on disk.
    """
    from table_validator.auth.databricks_auth import (
        ENV_PATH,
        get_databricks_token,
        host_from_workspace_url,
    )
    from table_validator.connectors.databricks_connector import DatabricksConnector

    try:
        token = secrets.get("DATABRICKS_TOKEN") or get_databricks_token(config, ENV_PATH)
        if not token:
            return
        databricks = DatabricksConnector(
            host=host_from_workspace_url(config.databricks.workspace_url),
            token=token,
            http_path=config.databricks.http_path,
        )
        source_columns_df = databricks.get_table_schema(
            config.source_table.catalog, config.source_table.schema_name, config.source_table.table
        )
        target_columns_df = databricks.get_table_schema(
            config.target_table.catalog, config.target_table.schema_name, config.target_table.table
        )
    except Exception:
        # Any connection/auth/query failure here just means the live
        # picker isn't available this run - column_map can still be set
        # later by re-running configure, or left unset entirely.
        return

    source_names = [str(c) for c in source_columns_df["column_name"]]
    target_names = [str(c) for c in target_columns_df["column_name"]]
    target_lower = {t.lower() for t in target_names}

    unmatched_source = [s for s in source_names if s.lower() not in target_lower]
    if not unmatched_source:
        # Every source column already has an identical-name match -
        # nothing to map, so don't ask anything at all.
        return

    source_lower = {s.lower() for s in source_names}
    remaining_target = [t for t in target_names if t.lower() not in source_lower]

    typer.echo(
        f"\n{len(unmatched_source)} source column(s) have no identical-name "
        "match in the target table - map any that were renamed (optional):"
    )

    new_map: Dict[str, str] = dict(config.column_map or {})
    skip_label = "(skip this column)"
    for src_col in unmatched_source:
        if not remaining_target:
            break
        choice = questionary.select(
            f"  Source column '{src_col}' has no matching target column - map it to:",
            choices=remaining_target + [skip_label],
        ).ask()
        if choice and choice != skip_label:
            new_map[src_col] = choice
            remaining_target = [t for t in remaining_target if t != choice]

    config.column_map = new_map


def _write_env_file(values: Dict[str, str], env_path: Optional[Path] = None) -> None:
    """
    Write secrets to env_path (default: ENV_PATH, resolved at call time)
    as KEY=VALUE lines, restricted to owner read/write only. Only
    non-empty values are written, so a step the user skipped doesn't
    clobber a credential set in a previous run with an empty string.
    """
    env_path = env_path or ENV_PATH
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, val = line.partition("=")
                existing[key.strip()] = val

    existing.update({k: v for k, v in values.items() if v})

    content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(content, encoding="utf-8")

    # Owner read/write only (chmod 600). On Windows this is best-effort -
    # NTFS ACLs don't map 1:1 onto POSIX mode bits, but os.chmod still
    # clears the broadest "everyone" write/execute bits where supported.
    try:
        os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _normalize_workspace_url(url: str) -> str:
    """Prepend https:// if the user typed a bare hostname."""
    if not url.startswith(("https://", "http://")):
        return f"https://{url}"
    return url


def _resolve_validation_selection(selected_labels):
    """
    Resolve the checkbox answer from the "Which validations should run?"
    prompt into a deduplicated List[ValidationType].

    "All" is a wizard-only convenience label, not a ValidationType - if
    it's present (alone or alongside individual choices), the result is
    always the full four-type list, regardless of what else was checked.
    Otherwise, each checked group label (see _VALIDATION_GROUPS) expands
    to every ValidationType it represents - "catalog & schema" expands
    to both CATALOG and SCHEMA together, since they're shown as one
    combined checkbox. Result is deduplicated and ordered to match
    _ALL_VALIDATIONS.
    """
    selected = selected_labels or []

    if _ALL_LABEL in selected:
        return [vtype for vtype, _label in _ALL_VALIDATIONS]

    resolved = {
        vtype
        for label, vtypes in _VALIDATION_GROUPS
        if label in selected
        for vtype in vtypes
    }
    # Preserve _ALL_VALIDATIONS order, drop duplicates.
    return [vtype for vtype, _label in _ALL_VALIDATIONS if vtype in resolved]


def _prompt_source_type(existing: SourceType) -> SourceType:
    label_to_type = {label: stype for stype, label in _SOURCE_TYPE_CHOICES}
    existing_label = next(
        label for stype, label in _SOURCE_TYPE_CHOICES if stype == existing
    )
    answer = questionary.select(
        "What are you comparing?",
        choices=[label for _stype, label in _SOURCE_TYPE_CHOICES],
        default=existing_label,
    ).ask()
    return label_to_type.get(answer, SourceType.DATABRICKS)


def _prompt_databricks_credentials(config: ValidatorConfig, secrets: Dict[str, str]) -> None:
    """Databricks credentials - always collected, since every source type
    targets a Databricks catalog."""
    typer.echo("\n== Databricks (target) ==")
    workspace_url = _ask(
        questionary.text(
            "Databricks workspace URL (e.g. https://adb-123.databricks.net):",
            default=config.databricks.workspace_url or "",
        )
    )
    config.databricks.workspace_url = (
        _normalize_workspace_url(workspace_url) if workspace_url else None
    )
    config.databricks.http_path = _ask(
        questionary.text(
            "Databricks SQL Warehouse HTTP path (e.g. /sql/1.0/warehouses/abc123):",
            default=config.databricks.http_path or "",
        )
    )
    token = _ask(questionary.password("Databricks personal access token:"))
    if token:
        secrets["DATABRICKS_TOKEN"] = token


def _prompt_azure_ad_ids(config: ValidatorConfig) -> None:
    """Optional tenant/subscription IDs, reserved for a future Service
    Principal auth phase - relevant regardless of which Azure source (Blob
    or SQL) is selected, so asked once rather than duplicated per branch."""
    config.azure.tenant_id = _ask(
        questionary.text(
            "Azure AD tenant ID (optional, reserved for future Azure CLI / "
            "Service Principal auth - leave blank to skip):",
            default=config.azure.tenant_id or "",
        )
    )
    config.azure.subscription_id = _ask(
        questionary.text(
            "Azure subscription ID (optional, leave blank to skip):",
            default=config.azure.subscription_id or "",
        )
    )


def _prompt_databricks_source(config: ValidatorConfig) -> None:
    """source_type == databricks: source is another Databricks catalog,
    same shape/prompts as the target."""
    typer.echo("\n== Source table (Databricks) ==")
    source = _prompt_table_ref("Source", config.source_table)
    config.source_table = config.source_table.model_copy(update=source)


def _prompt_azure_blob_source(config: ValidatorConfig, secrets: Dict[str, str]) -> None:
    """source_type == azure_blob: Storage account/container/key, then
    optional folder_prefix/file_pattern scoping which blobs are compared."""
    typer.echo("\n== Azure Blob Storage (source) ==")
    _prompt_azure_ad_ids(config)

    config.azure.storage_account = _ask(
        questionary.text(
            "Azure Storage account name:",
            default=config.azure.storage_account or "",
        )
    )
    config.blob_source.container = _ask(
        questionary.text(
            "Container name:",
            default=config.blob_source.container or config.azure.container or "",
        )
    )
    # azure.container mirrors blob_source.container for backward
    # compatibility with the Databricks-source-only AzureConfig shape.
    config.azure.container = config.blob_source.container
    storage_key = _ask(questionary.password("Azure Storage account key:"))
    if storage_key:
        secrets["AZURE_STORAGE_KEY"] = storage_key

    config.blob_source.folder_prefix = _ask(
        questionary.text(
            "Folder prefix to scope blob discovery to "
            "(leave blank to scan the whole container):",
            default=config.blob_source.folder_prefix or "",
        )
    )
    config.blob_source.file_pattern = _ask(
        questionary.text(
            "File pattern to scope blob discovery to, e.g. '*.csv' or "
            "'*.parquet' (leave blank to consider every supported format):",
            default=config.blob_source.file_pattern or "",
        )
    )
    config.blob_source.blob_path = _ask(
        questionary.text(
            "Exact path to one specific source blob, e.g. "
            "'n8ndirectory/customers.csv' (leave blank to match multiple "
            "blobs by filename against catalog tables using folder_prefix/"
            "file_pattern above instead). If set together with a Target "
            "table name below, that exact blob and table are compared "
            "directly, even if their names don't match:",
            default=config.blob_source.blob_path or "",
        )
    )


def _prompt_azure_sql_source(config: ValidatorConfig, secrets: Dict[str, str]) -> None:
    """source_type == azure_sql: SQL server/database/credentials, then
    optional schema/table scoping (blank = compare all, same convention
    as the Databricks source/target prompts)."""
    typer.echo("\n== Azure SQL Database (source) ==")
    _prompt_azure_ad_ids(config)

    config.azure.sql_server = _ask(
        questionary.text(
            "Azure SQL server:",
            default=config.azure.sql_server or "",
        )
    )
    config.azure.sql_database = _ask(
        questionary.text(
            "Azure SQL database name:",
            default=config.azure.sql_database or "",
        )
    )
    sql_username = _ask(questionary.text("Azure SQL username:"))
    if sql_username:
        secrets["AZURE_SQL_USERNAME"] = sql_username
    sql_password = _ask(questionary.password("Azure SQL password:"))
    if sql_password:
        secrets["AZURE_SQL_PASSWORD"] = sql_password

    config.sql_source.schema_name = _ask(
        questionary.text(
            "Schema (leave blank to compare all schemas in this database):",
            default=config.sql_source.schema_name or "",
        )
    )
    config.sql_source.table = _ask(
        questionary.text(
            "Table (leave blank to compare all tables in this schema):",
            default=config.sql_source.table or "",
        )
    )


def _prompt_synapse_source(config: ValidatorConfig, secrets: Dict[str, str]) -> None:
    """source_type == synapse: Synapse SQL pool (dedicated or serverless)
    server/database/credentials, then optional schema/table scoping -
    identical shape to _prompt_azure_sql_source, since Synapse SQL speaks
    the same T-SQL/ODBC protocol Azure SQL Database does and is validated
    via the same AzureSqlConnector/AzureSqlValidator classes, just pointed
    at a different server/database/credential pair."""
    typer.echo("\n== Azure Synapse SQL pool (source) ==")
    _prompt_azure_ad_ids(config)

    config.azure.synapse_server = _ask(
        questionary.text(
            "Synapse SQL endpoint, e.g. "
            "'myworkspace.sql.azuresynapse.net' (serverless) or "
            "'myworkspace.sql.azuresynapse.net' (dedicated pool - same "
            "hostname, the pool/database name below is what selects it):",
            default=config.azure.synapse_server or "",
        )
    )
    config.azure.synapse_database = _ask(
        questionary.text(
            "Synapse SQL pool / database name (e.g. 'myworkspace' for "
            "serverless's built-in pool, or your dedicated pool's name):",
            default=config.azure.synapse_database or "",
        )
    )
    # How to authenticate. A workspace with SQL authentication disabled
    # (Entra-only, increasingly the default for new workspaces) rejects
    # any username/password with a bare "Login failed", so this choice is
    # asked explicitly rather than inferred from which fields are filled.
    auth_label = questionary.select(
        "How should this Synapse pool authenticate?",
        choices=[
            _SYNAPSE_AUTH_SQL_LABEL,
            _SYNAPSE_AUTH_ENTRA_LABEL,
        ],
        default=(
            _SYNAPSE_AUTH_ENTRA_LABEL
            if config.azure.synapse_auth_mode == SynapseAuthMode.ENTRA_SERVICE_PRINCIPAL
            else _SYNAPSE_AUTH_SQL_LABEL
        ),
    ).ask()

    if auth_label == _SYNAPSE_AUTH_ENTRA_LABEL:
        config.azure.synapse_auth_mode = SynapseAuthMode.ENTRA_SERVICE_PRINCIPAL
        # tenant_id was already collected by _prompt_azure_ad_ids above -
        # but it's optional there ("reserved for future auth"), and Entra
        # auth genuinely requires it, so re-ask if it's still blank rather
        # than letting the run fail later with a missing-field error.
        if not config.azure.tenant_id:
            config.azure.tenant_id = _ask(
                questionary.text("Entra ID tenant (directory) ID:")
            )
        config.azure.synapse_client_id = _ask(
            questionary.text(
                "Entra ID application (client) ID of the service principal:",
                default=config.azure.synapse_client_id or "",
            )
        )
        client_secret = _ask(questionary.password("Entra ID client secret:"))
        if client_secret:
            secrets["SYNAPSE_CLIENT_SECRET"] = client_secret
    else:
        config.azure.synapse_auth_mode = SynapseAuthMode.SQL
        synapse_username = _ask(questionary.text("Synapse SQL username:"))
        if synapse_username:
            secrets["SYNAPSE_USERNAME"] = synapse_username
        synapse_password = _ask(questionary.password("Synapse SQL password:"))
        if synapse_password:
            secrets["SYNAPSE_PASSWORD"] = synapse_password

    config.synapse_source.schema_name = _ask(
        questionary.text(
            "Schema (leave blank to compare all schemas in this pool):",
            default=config.synapse_source.schema_name or "",
        )
    )
    config.synapse_source.table = _ask(
        questionary.text(
            "Table (leave blank to compare all tables in this schema):",
            default=config.synapse_source.table or "",
        )
    )


def run_configure_wizard() -> None:
    """Run the full interactive configuration wizard."""
    if CONFIG_PATH.exists():
        overwrite = questionary.confirm(
            f"A config already exists at {CONFIG_PATH}. Overwrite it?",
            default=False,
        ).ask()
        if not overwrite:
            typer.echo("Cancelled. Existing configuration was not changed.")
            raise typer.Exit(code=0)

    config = load_config()
    secrets: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. What's being compared
    # ------------------------------------------------------------------
    config.source_type = _prompt_source_type(config.source_type)

    # ------------------------------------------------------------------
    # 2. Databricks credentials - always needed (target is always
    # Databricks); asked before branching so it's never duplicated below.
    # ------------------------------------------------------------------
    _prompt_databricks_credentials(config, secrets)

    # ------------------------------------------------------------------
    # 3. Source-specific credentials/scoping
    # ------------------------------------------------------------------
    if config.source_type == SourceType.AZURE_BLOB:
        _prompt_azure_blob_source(config, secrets)
    elif config.source_type == SourceType.AZURE_SQL:
        _prompt_azure_sql_source(config, secrets)
    elif config.source_type == SourceType.SYNAPSE:
        _prompt_synapse_source(config, secrets)
    else:
        _prompt_databricks_source(config)

    # ------------------------------------------------------------------
    # 4. Target table (always Databricks, regardless of source_type)
    # ------------------------------------------------------------------
    typer.echo("\n== Target table (Databricks) ==")
    target = _prompt_table_ref("Target", config.target_table)
    config.target_table = config.target_table.model_copy(update=target)

    # Primary key - only meaningful for a single specific table (both
    # source and target named, not a catalog-wide sweep). Every source
    # type's row-level comparison has the same synthetic-ROW_NUMBER()
    # fallback and the same cost/timeout risk without a real key
    # (databricks: source_table.table; azure_sql: sql_source.table;
    # synapse: synapse_source.table; azure_blob: blob_source.blob_path),
    # so this is asked whenever a specific table is named on both sides,
    # not just for Databricks. Blank schema/table on either side means
    # "compare many tables", which a single primary_key field can't
    # represent.
    if config.source_type == SourceType.AZURE_SQL:
        source_table_named = bool(config.sql_source.table)
    elif config.source_type == SourceType.SYNAPSE:
        source_table_named = bool(config.synapse_source.table)
    elif config.source_type == SourceType.AZURE_BLOB:
        source_table_named = bool(config.blob_source.blob_path)
    else:
        source_table_named = bool(config.source_table.table)

    if source_table_named and config.target_table.table:
        config.primary_key = _prompt_primary_key(config.primary_key)
        # Column-level customization - same single-table scope as the
        # primary key above (only_columns/ignore_columns/
        # ignore_datatype_columns are only meaningful when there's one
        # specific table to compare). Optional: a user who declines gets
        # identical behavior to before this feature existed.
        typer.echo("\n== Customize validation (optional) ==")
        _prompt_column_customization(config)
        # Row-filter predicate and column-name mapping (renamed columns) -
        # Databricks-to-Databricks only, since set_row_filters/column_map/
        # CatalogValidator don't apply to the Azure Blob/SQL source paths.
        if config.source_type == SourceType.DATABRICKS:
            _prompt_row_filter(config)
            _prompt_column_mapping(config, secrets)
    else:
        config.primary_key = None
        config.only_columns = None
        config.ignore_columns = []
        config.ignore_datatype_columns = []
        config.column_map = {}
        config.row_filter = None
        config.source_row_filter = None
        config.target_row_filter = None

    # Schema-wide sweep scoping - only meaningful when a schema is named
    # on both sides but no specific table is (the single-table branch
    # above already covers renaming via source_table.table/
    # target_table.table directly), and only for a Databricks source
    # (only_tables/table_map here feed CatalogValidator, which is the
    # Databricks-to-Databricks path only).
    if (
        config.source_type == SourceType.DATABRICKS
        and config.source_table.schema_name
        and config.target_table.schema_name
        and not (source_table_named and config.target_table.table)
    ):
        typer.echo("\n== Customize schema (optional) ==")
        _prompt_schema_scoping(config)
    else:
        config.only_tables = None
        config.table_map = {}

    # ------------------------------------------------------------------
    # 5. Validations to run
    # ------------------------------------------------------------------
    typer.echo("\n== Validations ==")
    typer.echo(
        "  catalog & schema - catalog/schema/table existence only (missing/\n"
        "  extra tables). column - column names, order, data types, "
        "nullable.\n"
        "  row - row counts, statistics, and row-level data mismatches.\n"
        "  Note: with column and row BOTH left unchecked, a matched table "
        "has\n"
        "  nothing left to check and will show as SKIPPED - "
        "catalog/schema\n"
        "  existence is already reported separately, at the schema level."
    )
    all_already_selected = set(config.validations) == {v for v, _label in _ALL_VALIDATIONS}
    selected_labels = questionary.checkbox(
        "Which validations should run?",
        choices=[
            questionary.Choice(_ALL_LABEL, checked=all_already_selected),
            *[
                # A group is pre-checked only if EVERY ValidationType it
                # covers is already selected - "catalog & schema" must
                # not show as checked from a config that only had one of
                # the two (a state only reachable pre-merge, e.g. hand-
                # edited YAML).
                questionary.Choice(
                    label,
                    checked=all(vtype in config.validations for vtype in vtypes),
                )
                for label, vtypes in _VALIDATION_GROUPS
            ],
        ],
    ).ask()
    # "All" (alone or combined with individual choices) always resolves to
    # the full set - it's a wizard convenience, never itself stored.
    config.validations = _resolve_validation_selection(selected_labels)

    if (
        ValidationType.COLUMN not in config.validations
        and ValidationType.ROW not in config.validations
    ):
        typer.secho(
            "\nNote: with column and row both unchecked, every matched "
            "table will show as SKIPPED in the report - only catalog/"
            "schema existence (missing/extra tables) is actually checked.",
            fg=typer.colors.YELLOW,
        )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    save_config(config)
    _write_env_file(secrets)

    typer.echo(f"\nConfiguration saved to {CONFIG_PATH}")
    typer.echo(
        "Credentials are stored in plaintext at ~/.table_validator/.env for now. "
        "A future version will support Azure CLI / Databricks OAuth login."
    )
