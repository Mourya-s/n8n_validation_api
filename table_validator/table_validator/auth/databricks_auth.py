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
