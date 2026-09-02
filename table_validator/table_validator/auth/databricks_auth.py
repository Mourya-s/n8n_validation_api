"""Databricks auth abstraction.

Phase 1 auth: the personal access token (PAT) is entered manually via the
CLI wizard and stored in ~/.table_validator/.env. get_databricks_token() is
the single place that reads it - every connector must call it instead of
reading os.environ directly, so only this function needs to change when a
later phase adds Databricks CLI / OAuth auth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from table_validator.config.schema import ValidatorConfig

ENV_PATH = Path.home() / ".table_validator" / ".env"


def get_databricks_token(config: ValidatorConfig, env_path: Path = ENV_PATH) -> Optional[str]:
    """
    Resolve the Databricks personal access token for the given config.

    Phase 1: reads DATABRICKS_TOKEN from ~/.table_validator/.env. A later
    phase can swap this body for Databricks CLI / OAuth auth without
    changing any caller.
    """
    values = dotenv_values(env_path) if env_path.exists() else {}
    return values.get("DATABRICKS_TOKEN") or None


def host_from_workspace_url(workspace_url: Optional[str]) -> Optional[str]:
    """DatabricksConnector wants a bare hostname; the wizard stores a full
    https:// workspace URL, so strip the scheme and any trailing path.

    Shared by cli/main.py (building the connector for `validate`) and
    cli/wizard.py (building a connector during `configure` for the
    column-mapping live picker) - kept here rather than in either CLI
    module so neither has to import from the other.
    """
    if not workspace_url:
        return None
    host = workspace_url.replace("https://", "").replace("http://", "")
    return host.split("/")[0]
