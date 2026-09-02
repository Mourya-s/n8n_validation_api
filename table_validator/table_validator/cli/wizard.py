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
from table_validator.config.schema import SourceType, ValidationType, ValidatorConfig

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

# What's being compared, shown as the wizard's first question. Labels are
# wizard-only display text; SourceType is what's actually stored.
_SOURCE_TYPE_CHOICES = [
    (SourceType.DATABRICKS, "Databricks catalog -> Databricks catalog"),
    (SourceType.AZURE_BLOB, "Azure Blob Storage -> Databricks catalog"),
    (SourceType.AZURE_SQL, "Azure SQL Database -> Databricks catalog"),
]


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
    Otherwise, each selected label maps to its ValidationType, in
    _ALL_VALIDATIONS order and with duplicates removed (checkbox answers
    shouldn't contain duplicates, but this stays defensive either way).
    """
    label_to_type = {label: vtype for vtype, label in _ALL_VALIDATIONS}
    selected = selected_labels or []

    if _ALL_LABEL in selected:
        return [vtype for vtype, _label in _ALL_VALIDATIONS]

    resolved = [label_to_type[label] for label in selected if label in label_to_type]
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
    # azure_blob: blob_source.blob_path), so this is asked whenever a
    # specific table is named on both sides, not just for Databricks.
    # Blank schema/table on either side means "compare many tables",
    # which a single primary_key field can't represent.
    if config.source_type == SourceType.AZURE_SQL:
        source_table_named = bool(config.sql_source.table)
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
        # Column-name mapping (renamed columns) - Databricks-to-Databricks
        # only, since column_map/CatalogValidator don't apply to the
        # Azure Blob/SQL source paths.
        if config.source_type == SourceType.DATABRICKS:
            _prompt_column_mapping(config, secrets)
    else:
        config.primary_key = None
        config.only_columns = None
        config.ignore_columns = []
        config.ignore_datatype_columns = []
        config.column_map = {}

    # ------------------------------------------------------------------
    # 5. Validations to run
    # ------------------------------------------------------------------
    typer.echo("\n== Validations ==")
    all_already_selected = set(config.validations) == {v for v, _label in _ALL_VALIDATIONS}
    selected_labels = questionary.checkbox(
        "Which validations should run?",
        choices=[
            questionary.Choice(_ALL_LABEL, checked=all_already_selected),
            *[
                questionary.Choice(label, checked=(vtype in config.validations))
                for vtype, label in _ALL_VALIDATIONS
            ],
        ],
    ).ask()
    # "All" (alone or combined with individual choices) always resolves to
    # the full set - it's a wizard convenience, never itself stored.
    config.validations = _resolve_validation_selection(selected_labels)

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
