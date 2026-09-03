"""Azure auth abstraction.

Phase 1 auth: credentials are entered manually via the CLI wizard and
stored in ~/.table_validator/.env. get_azure_credential() is the single
place that reads them - every connector must call it instead of reading
os.environ directly, so only this function needs to change when a later
phase adds Azure CLI / Service Principal auth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from table_validator.config.schema import ValidatorConfig

ENV_PATH = Path.home() / ".table_validator" / ".env"


@dataclass
class AzureCredential:
    """Resolved Azure credentials for the connectors this tool uses today.

    storage_account_key authenticates AzureConnector (Blob Storage);
    sql_username/sql_password authenticate AzureSqlConnector (Azure SQL
    Database); synapse_username/synapse_password authenticate the same
    AzureSqlConnector class when pointed at a Synapse SQL pool instead
    (Synapse SQL is protocol-identical T-SQL/ODBC, so no separate
    connector class exists - only a separate credential pair, since a
    Synapse SQL login is a distinct principal from an Azure SQL DB one).
    synapse_client_secret is the alternative to that pair: the Entra ID
    service-principal secret used when config.azure.synapse_auth_mode is
    'entra_service_principal', paired with the non-secret tenant_id /
    synapse_client_id that live in config.yaml.
    All fields are optional here since a given validation run may only
    need one pair.
    """

    storage_account_key: Optional[str] = None
    sql_username: Optional[str] = None
    sql_password: Optional[str] = None
    synapse_username: Optional[str] = None
    synapse_password: Optional[str] = None
    synapse_client_secret: Optional[str] = None


def get_azure_credential(config: ValidatorConfig, env_path: Path = ENV_PATH) -> AzureCredential:
    """
    Resolve Azure credentials for the given config.

    Phase 1: reads AZURE_STORAGE_KEY / AZURE_SQL_USERNAME /
    AZURE_SQL_PASSWORD / SYNAPSE_USERNAME / SYNAPSE_PASSWORD /
    SYNAPSE_CLIENT_SECRET from ~/.table_validator/.env. A later phase can
    swap this body for Azure CLI / managed-identity auth without changing
    any caller.
    """
    values = dotenv_values(env_path) if env_path.exists() else {}

    return AzureCredential(
        storage_account_key=values.get("AZURE_STORAGE_KEY") or None,
        sql_username=values.get("AZURE_SQL_USERNAME") or None,
        sql_password=values.get("AZURE_SQL_PASSWORD") or None,
        synapse_username=values.get("SYNAPSE_USERNAME") or None,
        synapse_password=values.get("SYNAPSE_PASSWORD") or None,
        synapse_client_secret=values.get("SYNAPSE_CLIENT_SECRET") or None,
    )
