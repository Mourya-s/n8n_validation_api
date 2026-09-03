"""Config schema: pydantic models describing stored (non-secret) configuration.

Secrets (Databricks PAT, Azure Storage key, Azure SQL password, etc.) are
handled separately in Phase 4 and never appear on these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationType(str, Enum):
    CATALOG = "catalog"
    SCHEMA = "schema"
    COLUMN = "column"
    ROW = "row"


class SourceType(str, Enum):
    """
    What's being compared against the (always-Databricks) target catalog.
    Defaults to DATABRICKS so existing saved configs without this field
    (from before source_type existed) load unchanged - they were always
    Databricks-to-Databricks comparisons.
    """

    DATABRICKS = "databricks"
    AZURE_BLOB = "azure_blob"
    AZURE_SQL = "azure_sql"
    SYNAPSE = "synapse"


class SynapseAuthMode(str, Enum):
    """
    How the Synapse SQL pool connection authenticates.

    Defaults to SQL so existing saved configs (from before Entra support
    existed) keep working unchanged - they were all SQL-auth logins.
    ENTRA_SERVICE_PRINCIPAL swaps UID/PWD for an Entra access token
    fetched from an app registration, which is what a workspace with SQL
    authentication disabled (Entra-only) requires.
    """

    SQL = "sql"
    ENTRA_SERVICE_PRINCIPAL = "entra_service_principal"


class AzureConfig(BaseModel):
    """Non-secret Azure connection details.

    tenant_id / subscription_id are captured now as groundwork for the
    later Azure CLI / Service Principal auth phase; the current manual-auth
    connectors don't need them yet. storage_account/container and
    sql_server/sql_database cover the two connector types this tool
    actually talks to today (Blob Storage and Azure SQL Database).
    synapse_server/synapse_database are the equivalent pair for a Synapse
    SQL pool (dedicated or serverless) source - kept as separate fields
    rather than reusing sql_server/sql_database so a user can have both an
    Azure SQL Database config and a Synapse config saved at once (only the
    section matching source_type is read at validate time, same
    convention as blob_source/sql_source).
    """

    tenant_id: Optional[str] = None
    subscription_id: Optional[str] = None

    storage_account: Optional[str] = None
    container: Optional[str] = None

    sql_server: Optional[str] = None
    sql_database: Optional[str] = None

    synapse_server: Optional[str] = None
    synapse_database: Optional[str] = None

    synapse_auth_mode: SynapseAuthMode = Field(
        default=SynapseAuthMode.SQL,
        description=(
            "How to authenticate to the Synapse SQL pool. 'sql' uses a "
            "SQL login (SYNAPSE_USERNAME/SYNAPSE_PASSWORD in .env); "
            "'entra_service_principal' uses a Microsoft Entra ID app "
            "registration (tenant_id + synapse_client_id here, plus "
            "SYNAPSE_CLIENT_SECRET in .env) and never sends a "
            "username/password at all."
        ),
    )

    synapse_client_id: Optional[str] = Field(
        default=None,
        description=(
            "Entra ID application (client) ID of the service principal "
            "used when synapse_auth_mode is 'entra_service_principal'. "
            "The matching tenant is read from azure.tenant_id, and the "
            "secret from SYNAPSE_CLIENT_SECRET in ~/.table_validator/.env."
        ),
    )


class DatabricksConfig(BaseModel):
    """Non-secret Databricks connection details."""

    workspace_url: Optional[str] = None
    http_path: Optional[str] = None


class TableRef(BaseModel):
    """
    Reference to a table, schema, or whole catalog.

    catalog is required. schema_name/table are optional: leaving
    schema_name unset means "compare every schema common to both
    catalogs"; leaving table unset (with schema_name set) means "compare
    every table common to both sides of that schema". CatalogValidator's
    compare_schemas/compare_tables perform this discovery internally
    whenever the corresponding restriction is left unset.
    """

    catalog: Optional[str] = None
    schema_name: Optional[str] = Field(default=None, alias="schema")
    table: Optional[str] = None

    model_config = {"populate_by_name": True}


class BlobSourceConfig(BaseModel):
    """
    Non-secret scoping for an Azure Blob Storage source (source_type =
    azure_blob). The storage account/container credentials themselves
    live on AzureConfig (storage_account/container, shared with any other
    use of the same storage account); this section only scopes WHICH
    blobs within that container are treated as comparison sources.
    """

    container: Optional[str] = None
    folder_prefix: Optional[str] = Field(
        default=None,
        description=(
            "Optional path prefix to restrict blob discovery to (e.g. "
            "'validation/2024/'). If unset, the whole container is scanned."
        ),
    )
    file_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Optional glob pattern to restrict blob discovery to (e.g. "
            "'*.csv' or '*.parquet'). If unset, every supported file "
            "extension (.csv/.txt/.xlsx/.xls/.parquet) is considered."
        ),
    )
    blob_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional exact path to a single source blob (e.g. "
            "'n8ndirectory/file_example_XLSX_100.csv'). If set together "
            "with target_table.table, that exact blob is compared "
            "directly against that exact table - bypassing filename-to-"
            "table-name discovery entirely, even if the names don't "
            "match. If unset, folder_prefix/file_pattern-based discovery "
            "across multiple blobs applies as usual."
        ),
    )


class SqlSourceConfig(BaseModel):
    """
    Non-secret scoping for an Azure SQL Database source (source_type =
    azure_sql). Server/database themselves live on AzureConfig
    (sql_server/sql_database); this section scopes which schema/table
    within that database are compared, same optional-means-"compare all"
    convention as TableRef.
    """

    schema_name: Optional[str] = Field(default=None, alias="schema")
    table: Optional[str] = None

    model_config = {"populate_by_name": True}


class SynapseSourceConfig(BaseModel):
    """
    Non-secret scoping for a Synapse SQL pool source (source_type =
    synapse, dedicated or serverless). Server/database themselves live on
    AzureConfig (synapse_server/synapse_database); this section scopes
    which schema/table within that pool are compared - identical shape
    and convention to SqlSourceConfig, kept as its own class (rather than
    reusing SqlSourceConfig) so a saved Azure SQL scoping and a saved
    Synapse scoping don't overwrite each other when switching source
    types back and forth in the wizard.
    """

    schema_name: Optional[str] = Field(default=None, alias="schema")
    table: Optional[str] = None

    model_config = {"populate_by_name": True}


class ValidatorConfig(BaseModel):
    """
    Top-level, non-secret configuration persisted to config.yaml.

    source_type selects which of source_table/blob_source/sql_source is
    read by `validate` - only the section matching source_type is
    meaningful; the other two may be present (e.g. left over from
    switching source types in the wizard) but are ignored. target_table
    is always a Databricks catalog/schema/table ref regardless of
    source_type, since every comparison path targets Databricks.
    """

    source_type: SourceType = Field(default=SourceType.DATABRICKS)

    azure: AzureConfig = Field(default_factory=AzureConfig)
    databricks: DatabricksConfig = Field(default_factory=DatabricksConfig)

    source_table: TableRef = Field(default_factory=TableRef)
    target_table: TableRef = Field(default_factory=TableRef)

    primary_key: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional primary/business key column(s) for the single named "
            "table in source_table/target_table (only meaningful when both "
            "are set to a specific table, not left blank for a catalog-"
            "wide sweep). When set, row-level comparison uses this key "
            "instead of falling back to a synthetic ROW_NUMBER() match - "
            "cheaper and more reliable, and avoids the full-table sort "
            "the row-number fallback needs on large tables. If unset, the "
            "row-number fallback is used as before."
        ),
    )

    only_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist: if set, only these columns (plus the "
            "primary key, if any) are compared - every other common "
            "column is excluded from every check. Only meaningful for the "
            "single named table in source_table/target_table, same scope "
            "as primary_key. If a column named here isn't actually common "
            "to both sides, it's silently absent from the effective set "
            "(same convention as tables/schemas restrictions elsewhere). "
            "If both only_columns and ignore_columns name the same "
            "column, ignore_columns wins - it is always excluded."
        ),
    )

    ignore_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Columns to exclude entirely from every check (name, type, "
            "nullable, statistics, row-hash) - useful for auto-generated "
            "columns like timestamps that are expected to always differ."
        ),
    )

    ignore_datatype_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Columns whose data-type mismatch should be ignored - the "
            "column's other checks (nullable, statistics, row-hash) still "
            "run normally, but a real type difference here is reported as "
            "SKIPPED rather than FAIL and never fails the table or aborts "
            "the schema stage, even for a cross-family type change that "
            "would otherwise be BLOCKING."
        ),
    )

    column_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of source-table column name -> target-table "
            "column name, for an individual column renamed between "
            "source and target (e.g. source has 'cust_id', target has "
            "'customer_id'). Only meaningful for the single named table "
            "in source_table/target_table, same scope as primary_key. "
            "Set interactively by `configure`'s column-mapping picker, "
            "or by hand. A column configured as both a primary key and "
            "an entry here is rejected with a clear error at validate "
            "time - a column used as the row-level join key must have "
            "the identical name on both sides."
        ),
    )

    only_tables: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist: if set, only these table names (within "
            "the schema named in source_table/target_table) are "
            "validated - every other common table in that schema is "
            "skipped. Only meaningful when a schema is named but no "
            "specific table is (a schema-wide sweep); ignored otherwise. "
            "A name here that isn't actually a common table is silently "
            "absent from the effective set, same convention as "
            "only_columns. If a table appears in both only_tables and "
            "table_map, the table_map entry wins for that table."
        ),
    )

    table_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of source-schema table name -> target-schema "
            "table name, for tables renamed between source and target "
            "within an otherwise schema-wide sweep (e.g. source has "
            "'cust', target has 'customers'). Only meaningful when a "
            "schema is named but no specific table is - for the single-"
            "table case, naming both source_table.table and "
            "target_table.table already pairs them directly regardless "
            "of name. Unmapped tables are still matched by identical "
            "name as usual. Set interactively by `configure`'s table-"
            "mapping prompt, or by hand."
        ),
    )

    row_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional SQL WHERE-fragment restricting comparison to matching "
            "rows on BOTH sides (row count, statistics, row-hash/column "
            "diff) - e.g. 'id > 20 and id < 100' or \"gender = 'male'\". "
            "Only meaningful for the single named table in source_table/"
            "target_table (Databricks source only). Combined with "
            "source_row_filter/target_row_filter via AND if both are set. "
            "Used as-is, not parsed or validated - a typo surfaces as a "
            "normal SQL error when the query runs."
        ),
    )

    source_row_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional SQL WHERE-fragment applied only to the source side, "
            "ANDed with row_filter if both are set. Use this (with "
            "target_row_filter) instead of row_filter when the condition "
            "must differ per side."
        ),
    )

    target_row_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional SQL WHERE-fragment applied only to the target side, "
            "ANDed with row_filter if both are set."
        ),
    )

    blob_source: BlobSourceConfig = Field(default_factory=BlobSourceConfig)
    sql_source: SqlSourceConfig = Field(default_factory=SqlSourceConfig)
    synapse_source: SynapseSourceConfig = Field(default_factory=SynapseSourceConfig)

    validations: List[ValidationType] = Field(
        default_factory=lambda: [
            ValidationType.CATALOG,
            ValidationType.SCHEMA,
            ValidationType.COLUMN,
            ValidationType.ROW,
        ]
    )
