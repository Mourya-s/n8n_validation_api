"""
Tests for cli/wizard.py's answer normalization: _ask() strip()/None
behavior, _prompt_table_ref()'s optional schema/table handling, and
_normalize_workspace_url(). These are pure-function-level tests (no real
questionary prompt is constructed), so they run everywhere including
terminals without a real console (avoids the prompt_toolkit
NoConsoleScreenBufferError seen under Git Bash / MinTTY on Windows).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from table_validator.cli.wizard import _ask, _normalize_workspace_url, _prompt_table_ref


def _fake_prompt(answer):
    """A stand-in for a questionary.Question whose .ask() returns `answer`."""
    q = MagicMock()
    q.ask.return_value = answer
    return q


# ---------------------------------------------------------------------------
# _ask(): strip() + blank -> None normalization
# ---------------------------------------------------------------------------
def test_ask_strips_leading_and_trailing_whitespace():
    assert _ask(_fake_prompt("  for_schema_validation  ")) == "for_schema_validation"


def test_ask_strips_internal_value_unaffected():
    assert _ask(_fake_prompt("bronze")) == "bronze"


def test_ask_blank_string_becomes_none():
    assert _ask(_fake_prompt("")) is None


def test_ask_whitespace_only_becomes_none():
    assert _ask(_fake_prompt("   ")) is None


def test_ask_none_answer_stays_none():
    # e.g. Ctrl-C / EOF during the prompt -> questionary returns None
    assert _ask(_fake_prompt(None)) is None


# ---------------------------------------------------------------------------
# _prompt_table_ref(): optional schema/table -> None when left blank
# ---------------------------------------------------------------------------
def test_prompt_table_ref_all_filled_in():
    import table_validator.cli.wizard as wizard_mod

    existing = SimpleNamespace(catalog=None, schema_name=None, table=None)
    answers = iter(["src_cat", "bronze", "orders"])

    def fake_text(*args, **kwargs):
        return _fake_prompt(next(answers))

    orig_text = wizard_mod.questionary.text
    wizard_mod.questionary.text = fake_text
    try:
        result = _prompt_table_ref("Source", existing)
    finally:
        wizard_mod.questionary.text = orig_text

    assert result == {"catalog": "src_cat", "schema_name": "bronze", "table": "orders"}


def test_prompt_table_ref_blank_schema_and_table_become_none():
    """Leaving Schema/Table blank must save None, not empty string - this
    is what makes 'compare all schemas/tables' work; an empty string
    would be a real (wrong) value, not 'unset'."""
    import table_validator.cli.wizard as wizard_mod

    existing = SimpleNamespace(catalog=None, schema_name=None, table=None)
    answers = iter(["src_cat", "", ""])

    def fake_text(*args, **kwargs):
        return _fake_prompt(next(answers))

    orig_text = wizard_mod.questionary.text
    wizard_mod.questionary.text = fake_text
    try:
        result = _prompt_table_ref("Source", existing)
    finally:
        wizard_mod.questionary.text = orig_text

    assert result == {"catalog": "src_cat", "schema_name": None, "table": None}


def test_prompt_table_ref_strips_stray_whitespace():
    """Reproduces the exact bug seen in a real config.yaml:
    schema: ' for_schema_validation' (leading space)."""
    import table_validator.cli.wizard as wizard_mod

    existing = SimpleNamespace(catalog=None, schema_name=None, table=None)
    answers = iter(["for_validation2", " for_schema_validation", "sample_30_mb "])

    def fake_text(*args, **kwargs):
        return _fake_prompt(next(answers))

    orig_text = wizard_mod.questionary.text
    wizard_mod.questionary.text = fake_text
    try:
        result = _prompt_table_ref("Target", existing)
    finally:
        wizard_mod.questionary.text = orig_text

    assert result == {
        "catalog": "for_validation2",
        "schema_name": "for_schema_validation",
        "table": "sample_30_mb",
    }


# ---------------------------------------------------------------------------
# _normalize_workspace_url(): prepend https:// if missing
# ---------------------------------------------------------------------------
def test_normalize_workspace_url_prepends_https_when_missing():
    assert (
        _normalize_workspace_url("adb-123.databricks.net")
        == "https://adb-123.databricks.net"
    )


def test_normalize_workspace_url_leaves_https_url_unchanged():
    url = "https://adb-123.databricks.net"
    assert _normalize_workspace_url(url) == url


def test_normalize_workspace_url_leaves_http_url_unchanged():
    url = "http://adb-123.databricks.net"
    assert _normalize_workspace_url(url) == url


# ---------------------------------------------------------------------------
# run_configure_wizard(): blank top-level Azure fields must null out their
# dependent fields too, not just skip setting them (the real bug: a
# previous run's sql_server='myserver...'/sql_database='mydb' surviving
# into a re-run where the user left those prompts blank).
# ---------------------------------------------------------------------------
def test_blank_azure_fields_clear_stale_dependent_values(tmp_path, monkeypatch):
    import table_validator.config.manager as manager_mod
    import table_validator.cli.wizard as wizard_mod
    from table_validator.config.schema import ValidatorConfig

    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    # Pre-existing config has real-looking stale values from a prior run.
    stale = ValidatorConfig()
    stale.azure.storage_account = "n8nstorages"
    stale.azure.container = "n8ncontainer"
    stale.azure.sql_server = "myserver.database.windows.net"
    stale.azure.sql_database = "mydb"
    manager_mod.save_config(stale, config_path)

    monkeypatch.setattr(manager_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wizard_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wizard_mod, "ENV_PATH", env_path)

    # Overwrite=True, source type = Databricks (today's flow), then blank
    # every Azure field, blank Databricks/table/validations prompts too
    # (only Azure fields matter for this test - the Databricks source
    # branch never touches azure.storage_account/sql_server itself, but
    # those fields being left alone rather than reset is exactly the bug
    # being guarded against).
    answers = iter([
        True,                      # overwrite confirm
        "Databricks catalog -> Databricks catalog",  # source type select
        "", "", "",                # databricks workspace_url, http_path, token
        "", "", "",                # source table catalog/schema/table
        "", "", "",                # target table catalog/schema/table
        [],                        # validations checkbox
    ])

    def fake_prompt(*args, **kwargs):
        q = MagicMock()
        q.ask.return_value = next(answers)
        return q

    monkeypatch.setattr(wizard_mod.questionary, "confirm", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "text", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "password", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "checkbox", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "select", fake_prompt)

    wizard_mod.run_configure_wizard()

    reloaded = manager_mod.load_config(config_path)
    # The Databricks source branch doesn't prompt for these at all, so
    # they simply keep whatever load_config() read from the pre-existing
    # stale config - this test now documents that (rather than a reset),
    # since azure_blob/azure_sql are the branches that actually own and
    # clear these fields (covered by the dedicated tests below).
    assert reloaded.azure.storage_account == "n8nstorages"
    assert reloaded.azure.sql_server == "myserver.database.windows.net"


def test_blank_blob_source_fields_are_saved_as_none(tmp_path, monkeypatch):
    """azure_blob branch: leaving storage account/container blank must
    save None, not carry forward a prior run's stale value - same class
    of bug as the original Databricks-source version of this test."""
    import table_validator.config.manager as manager_mod
    import table_validator.cli.wizard as wizard_mod
    from table_validator.config.schema import ValidatorConfig

    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    stale = ValidatorConfig()
    stale.azure.storage_account = "n8nstorages"
    stale.azure.container = "n8ncontainer"
    stale.blob_source.container = "n8ncontainer"
    stale.blob_source.folder_prefix = "old/prefix/"
    manager_mod.save_config(stale, config_path)

    monkeypatch.setattr(manager_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wizard_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wizard_mod, "ENV_PATH", env_path)

    answers = iter([
        True,                                             # overwrite confirm
        "Azure Blob Storage -> Databricks catalog",        # source type select
        "", "", "",                                        # databricks workspace_url, http_path, token
        None, None,                                        # tenant_id, subscription_id
        "",                                                 # storage account blank
        "",                                                 # container blank
        "",                                                 # storage key blank
        "",                                                 # folder_prefix blank
        "",                                                 # file_pattern blank
        "",                                                 # blob_path blank
        "", "", "",                                         # target table catalog/schema/table
        [],                                                 # validations checkbox
    ])

    def fake_prompt(*args, **kwargs):
        q = MagicMock()
        q.ask.return_value = next(answers)
        return q

    monkeypatch.setattr(wizard_mod.questionary, "confirm", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "text", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "password", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "checkbox", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "select", fake_prompt)

    wizard_mod.run_configure_wizard()

    reloaded = manager_mod.load_config(config_path)
    assert reloaded.source_type.value == "azure_blob"
    assert reloaded.azure.storage_account is None
    assert reloaded.azure.container is None
    assert reloaded.blob_source.container is None
    assert reloaded.blob_source.folder_prefix is None
    assert reloaded.blob_source.file_pattern is None
    assert reloaded.blob_source.blob_path is None


# ---------------------------------------------------------------------------
# _resolve_validation_selection(): "All" checkbox convenience
# ---------------------------------------------------------------------------
def test_all_option_alone_resolves_to_every_validation_type():
    from table_validator.cli.wizard import _resolve_validation_selection
    from table_validator.config.schema import ValidationType

    result = _resolve_validation_selection(["All"])

    assert result == [
        ValidationType.CATALOG,
        ValidationType.SCHEMA,
        ValidationType.COLUMN,
        ValidationType.ROW,
    ]


def test_all_option_combined_with_individual_choice_has_no_duplicates():
    from table_validator.cli.wizard import _resolve_validation_selection
    from table_validator.config.schema import ValidationType

    result = _resolve_validation_selection(["All", "row"])

    assert result == [
        ValidationType.CATALOG,
        ValidationType.SCHEMA,
        ValidationType.COLUMN,
        ValidationType.ROW,
    ]
    assert len(result) == len(set(result)) == 4


# ---------------------------------------------------------------------------
# source_type branching: each of the three choices must drive the wizard
# through its own set of prompts and persist the right config sections.
# ---------------------------------------------------------------------------
def _run_wizard_with_answers(
    tmp_path, monkeypatch, answers_list, databricks_token="", databricks_connector=None,
):
    """Helper: run run_configure_wizard() with every questionary prompt
    type patched to pop answers off answers_list in order, against an
    isolated config/env path. Returns the reloaded ValidatorConfig.

    databricks_token/databricks_connector default to "" (falsy) / a
    MagicMock, patched into the auth/databricks_auth and
    databricks_connector modules that _prompt_column_mapping imports
    LOCALLY (inside the function body, not at module import time) - a
    falsy token makes that step's live-connection attempt take its
    graceful-fallback early return deterministically, so tests that
    don't care about column-mapping never touch a real network call or
    need an extra stubbed answer. Tests that DO want to exercise the
    live-picker path pass a truthy token and a MagicMock connector whose
    get_table_schema is configured with the desired column lists."""
    import table_validator.config.manager as manager_mod
    import table_validator.cli.wizard as wizard_mod
    import table_validator.auth.databricks_auth as databricks_auth_mod
    import table_validator.connectors.databricks_connector as databricks_connector_mod

    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"

    answers = iter(answers_list)

    def fake_prompt(*args, **kwargs):
        q = MagicMock()
        q.ask.return_value = next(answers)
        return q

    monkeypatch.setattr(wizard_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(manager_mod, "CONFIG_PATH", config_path)
    monkeypatch.setattr(wizard_mod, "ENV_PATH", env_path)
    monkeypatch.setattr(wizard_mod.questionary, "confirm", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "text", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "password", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "checkbox", fake_prompt)
    monkeypatch.setattr(wizard_mod.questionary, "select", fake_prompt)
    monkeypatch.setattr(databricks_auth_mod, "get_databricks_token", lambda *a, **k: databricks_token)
    monkeypatch.setattr(
        databricks_connector_mod, "DatabricksConnector",
        MagicMock(return_value=databricks_connector or MagicMock()),
    )

    wizard_mod.run_configure_wizard()

    return manager_mod.load_config(config_path)


def test_databricks_source_type_prompts_for_source_table(tmp_path, monkeypatch):
    """Selecting the Databricks->Databricks choice must ask for a source
    catalog/schema/table (today's flow) and never touch blob_source/sql_source."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",  # source type select
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",  # databricks
        "src_cat", "bronze", "customers",   # source table
        "tgt_cat", "silver", "customers",   # target table
        "",                                   # primary key (left blank)
        False,                                # customize validation? (declined)
        [],                                  # validations
    ])

    assert config.source_type.value == "databricks"
    assert config.source_table.catalog == "src_cat"
    assert config.source_table.schema_name == "bronze"
    assert config.source_table.table == "customers"
    assert config.blob_source.container is None
    assert config.sql_source.schema_name is None
    assert config.primary_key is None


def test_databricks_source_type_prompts_for_primary_key_when_both_tables_named(
    tmp_path, monkeypatch
):
    """When both source and target table are specific (not a catalog-wide
    sweep), the wizard must ask for an optional primary key, and a
    comma-separated answer must be parsed into a clean column list."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "bronze", "customers",
        "tgt_cat", "silver", "customers",
        "id, region_id ",                    # primary key, comma-separated with stray spaces
        False,                                # customize validation? (declined)
        [],
    ])

    assert config.primary_key == ["id", "region_id"]


def test_databricks_wizard_declining_customization_leaves_defaults_untouched(
    tmp_path, monkeypatch
):
    """Answering No to the customize-validation gate must leave
    only_columns/ignore_columns/ignore_datatype_columns at their defaults
    - a user who never opts in gets identical behavior to before this
    feature existed."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "bronze", "customers",
        "tgt_cat", "silver", "customers",
        "",                                    # primary key (left blank)
        False,                                 # customize validation? (declined)
        [],
    ])

    assert config.only_columns is None
    assert config.ignore_columns == []
    assert config.ignore_datatype_columns == []


def test_databricks_wizard_customize_populates_all_three_column_lists(
    tmp_path, monkeypatch
):
    """Answering Yes to the customize-validation gate must ask for and
    save all three column lists - only_columns (allowlist),
    ignore_columns (denylist), ignore_datatype_columns."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "bronze", "customers",
        "tgt_cat", "silver", "customers",
        "id",                                  # primary key
        True,                                  # customize validation? (accepted)
        "id, name ",                            # only_columns, stray spaces
        "updated_at",                           # ignore_columns
        "legacy_flag, notes",                   # ignore_datatype_columns
        [],
    ])

    assert config.only_columns == ["id", "name"]
    assert config.ignore_columns == ["updated_at"]
    assert config.ignore_datatype_columns == ["legacy_flag", "notes"]


def test_databricks_wizard_customize_all_three_left_blank(tmp_path, monkeypatch):
    """Opting into customization but leaving all three prompts blank must
    resolve to only_columns=None, ignore_columns=[], ignore_datatype_columns=[] -
    the same effective defaults as declining outright."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "bronze", "customers",
        "tgt_cat", "silver", "customers",
        "id",
        True,                                  # customize validation? (accepted)
        "",                                    # only_columns left blank
        "",                                    # ignore_columns left blank
        "",                                    # ignore_datatype_columns left blank
        [],
    ])

    assert config.only_columns is None
    assert config.ignore_columns == []
    assert config.ignore_datatype_columns == []


def test_databricks_wizard_catalog_wide_sweep_skips_customization_prompt(
    tmp_path, monkeypatch
):
    """Catalog-wide sweep (no specific table named) must skip the
    customize-validation gate entirely, same scope rule as primary_key -
    a single column list can't apply to many different tables."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "", "",                    # source: catalog only
        "tgt_cat", "", "",                    # target: catalog only
        [],                                    # validations (no customize prompt in between)
    ])

    assert config.only_columns is None
    assert config.ignore_columns == []
    assert config.ignore_datatype_columns == []


def test_databricks_wizard_skips_primary_key_prompt_for_catalog_wide_sweep(
    tmp_path, monkeypatch
):
    """When either source or target table is left blank (catalog-wide
    sweep), the wizard must NOT ask for a primary key at all - a single
    key can't apply to many different tables."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "", "",                   # source: catalog only, schema/table blank
        "tgt_cat", "", "",                   # target: catalog only, schema/table blank
        [],                                   # validations (no primary key prompt in between)
    ])

    assert config.primary_key is None


def test_azure_blob_source_type_prompts_for_blob_scoping(tmp_path, monkeypatch):
    """Selecting the Azure Blob choice must ask for storage account/
    container/key + folder_prefix/file_pattern, and never prompt for a
    source_table catalog/schema/table (there is no source catalog for a
    blob source)."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Azure Blob Storage -> Databricks catalog",  # source type select
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",  # databricks
        None, None,                          # tenant_id, subscription_id
        "n8nstorages",                       # storage account
        "n8ncontainer",                      # container
        "supersecretkey",                    # storage key
        "validation/2024/",                  # folder_prefix
        "*.csv",                             # file_pattern
        "",                                   # blob_path (left blank)
        "tgt_cat", "silver", "",             # target table (table left blank)
        [],                                   # validations
    ])

    assert config.source_type.value == "azure_blob"
    assert config.azure.storage_account == "n8nstorages"
    assert config.blob_source.container == "n8ncontainer"
    assert config.blob_source.folder_prefix == "validation/2024/"
    assert config.blob_source.file_pattern == "*.csv"
    assert config.blob_source.blob_path is None
    assert config.target_table.catalog == "tgt_cat"
    assert config.target_table.table is None
    # No Databricks source catalog concept for a blob source.
    assert config.source_table.catalog is None


def test_azure_sql_source_type_prompts_for_sql_scoping(tmp_path, monkeypatch):
    """Selecting the Azure SQL choice must ask for server/database/
    credentials + optional schema/table, and never prompt for blob scoping."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Azure SQL Database -> Databricks catalog",  # source type select
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",  # databricks
        None, None,                          # tenant_id, subscription_id
        "myserver.database.windows.net",     # sql server
        "mydb",                              # sql database
        "sqluser",                           # sql username
        "sqlpass",                           # sql password
        "",                                   # sql_source schema left blank
        "",                                   # sql_source table left blank
        "tgt_cat", "silver", "customers",    # target table
        [],                                   # validations
    ])

    assert config.source_type.value == "azure_sql"
    assert config.azure.sql_server == "myserver.database.windows.net"
    assert config.azure.sql_database == "mydb"
    assert config.sql_source.schema_name is None
    assert config.sql_source.table is None
    assert config.blob_source.container is None


def test_azure_sql_prompts_for_primary_key_when_table_named(tmp_path, monkeypatch):
    """Regression test: the primary-key prompt was previously gated to
    source_type == DATABRICKS only, so an Azure SQL config with a specific
    table named on both sides silently never got to set a primary key -
    config.primary_key was always forced to None, permanently disabling
    Tier 5-equivalent column-level detail (Data Mismatches) for this
    source type regardless of what the user wanted."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Azure SQL Database -> Databricks catalog",  # source type select
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",  # databricks
        None, None,                          # tenant_id, subscription_id
        "myserver.database.windows.net",     # sql server
        "mydb",                              # sql database
        "sqluser",                           # sql username
        "sqlpass",                           # sql password
        "dbo",                                # sql_source schema
        "employees",                          # sql_source table
        "tgt_cat", "dbo", "employees_sample",  # target table
        "EmployeeID",                         # primary key
        False,                                # customize validation? (declined)
        [],                                   # validations
    ])

    assert config.primary_key == ["EmployeeID"]


def test_azure_sql_skips_primary_key_prompt_when_table_left_blank(tmp_path, monkeypatch):
    """No sql_source.table named -> catalog-wide sweep -> no primary key
    prompt (a single key can't apply to many different tables)."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Azure SQL Database -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        None, None,
        "myserver.database.windows.net",
        "mydb",
        "sqluser",
        "sqlpass",
        "",                                    # sql_source schema left blank
        "",                                    # sql_source table left blank
        "tgt_cat", "", "",                     # target: catalog only
        [],                                    # validations (no primary key prompt in between)
    ])

    assert config.primary_key is None


def test_azure_blob_prompts_for_primary_key_when_blob_path_named(tmp_path, monkeypatch):
    """Same fix, for the Azure Blob path: an explicit blob_path + target
    table pair should also be offered a primary key."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Azure Blob Storage -> Databricks catalog",  # source type select
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",  # databricks
        None, None,                          # tenant_id, subscription_id
        "mystorageaccount",                  # storage account
        "mycontainer",                       # container
        "storagekey",                        # storage account key
        "",                                    # folder_prefix left blank
        "*.csv",                               # file_pattern
        "validation/customers.csv",            # blob_path (explicit)
        "tgt_cat", "bronze", "customers",      # target table
        "id",                                  # primary key
        False,                                 # customize validation? (declined)
        [],                                    # validations
    ])

    assert config.primary_key == ["id"]


# ---------------------------------------------------------------------------
# _prompt_column_mapping: live column-name-mapping picker (Databricks-to-
# Databricks only), with a graceful fallback if the live connection fails.
# ---------------------------------------------------------------------------
def _schema_df(columns):
    """columns: list of names (data_type/is_nullable irrelevant here)."""
    import pandas as pd
    return pd.DataFrame([
        {"column_name": c, "data_type": "string", "is_nullable": True, "ordinal_position": i + 1}
        for i, c in enumerate(columns)
    ])


def test_column_mapping_skipped_when_no_token_available(tmp_path, monkeypatch):
    """Default _run_wizard_with_answers behavior (no token) - the
    column-mapping step must skip silently, asking nothing at all, and
    leave column_map empty."""
    config = _run_wizard_with_answers(tmp_path, monkeypatch, [
        "Databricks catalog -> Databricks catalog",
        "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
        "src_cat", "bronze", "customers",
        "tgt_cat", "silver", "customers",
        "id",
        False,  # customize validation? (declined)
        [],
    ])

    assert config.column_map == {}


def test_column_mapping_live_connection_success_populates_map(tmp_path, monkeypatch):
    """A live connection succeeds, one source column has no identical-
    name match, and the user picks a target column for it."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df(["id", "cust_id"]) if catalog == "src_cat"
        else _schema_df(["id", "customer_id"])
    )
    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Databricks catalog -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            "src_cat", "bronze", "customers",
            "tgt_cat", "silver", "customers",
            "id",
            False,           # customize validation? (declined)
            "customer_id",   # column mapping: cust_id -> customer_id
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {"cust_id": "customer_id"}


def test_column_mapping_live_connection_failure_skips_gracefully(tmp_path, monkeypatch):
    """A truthy token but a connection/query failure must still degrade
    to silently skipping, never crashing configure."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = RuntimeError("connection refused")

    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Databricks catalog -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            "src_cat", "bronze", "customers",
            "tgt_cat", "silver", "customers",
            "id",
            False,  # customize validation? (declined)
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {}


def test_column_mapping_only_asks_about_unmatched_source_columns(tmp_path, monkeypatch):
    """Two of three source columns already have an identical-name target
    match - only the genuinely unmatched one should trigger a select
    prompt. Proven by the answer sequence needing exactly ONE
    column-mapping answer ("customer_id") despite three source columns
    existing - if 'id'/'name' were also prompted, the answers iterator
    would be exhausted early and the wizard would raise StopIteration."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df(["id", "name", "cust_id"]) if catalog == "src_cat"
        else _schema_df(["id", "name", "customer_id"])
    )
    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Databricks catalog -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            "src_cat", "bronze", "customers",
            "tgt_cat", "silver", "customers",
            "id",
            False,
            "customer_id",  # only ONE column-mapping answer, for cust_id
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {"cust_id": "customer_id"}


def test_column_mapping_skip_leaves_column_unmapped(tmp_path, monkeypatch):
    """Answering the skip label for an unmatched column must leave it
    absent from column_map entirely."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: (
        _schema_df(["id", "cust_id"]) if catalog == "src_cat"
        else _schema_df(["id", "customer_id"])
    )
    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Databricks catalog -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            "src_cat", "bronze", "customers",
            "tgt_cat", "silver", "customers",
            "id",
            False,
            "(skip this column)",
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {}


def test_column_mapping_no_unmatched_columns_asks_nothing(tmp_path, monkeypatch):
    """Every source column already has an identical-name target match -
    the step must ask nothing at all (no extra answer consumed)."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = lambda catalog, schema, table: _schema_df(["id", "name"])

    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Databricks catalog -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            "src_cat", "bronze", "customers",
            "tgt_cat", "silver", "customers",
            "id",
            False,
            # no column-mapping answer - none should be consumed
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {}


def test_column_mapping_skipped_for_azure_sql_source_type(tmp_path, monkeypatch):
    """Non-Databricks source types must never attempt the live
    connection or prompt at all, even with a valid token/connector."""
    connector = MagicMock()
    connector.get_table_schema.side_effect = AssertionError("should never be called")

    config = _run_wizard_with_answers(
        tmp_path, monkeypatch,
        [
            "Azure SQL Database -> Databricks catalog",
            "https://adb-1.databricks.net", "/sql/1.0/warehouses/x", "tok",
            None, None,
            "myserver.database.windows.net",
            "mydb",
            "sqluser",
            "sqlpass",
            "dbo",
            "employees",
            "tgt_cat", "dbo", "employees_sample",
            "EmployeeID",
            False,  # customize validation? (declined)
            [],
        ],
        databricks_token="dapi_fake",
        databricks_connector=connector,
    )

    assert config.column_map == {}
