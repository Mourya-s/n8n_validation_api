"""Config manager: loads/saves non-secret config under ~/.table_validator/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from table_validator.config.schema import ValidatorConfig

CONFIG_DIR = Path.home() / ".table_validator"
CONFIG_PATH = CONFIG_DIR / "config.yaml"


class ConfigNotFoundError(Exception):
    """Raised when a config file is required but doesn't exist yet.

    Distinct from a generic FileNotFoundError so callers (e.g. the
    `validate` command) can catch it specifically and print a targeted
    "run configure first" message instead of a raw traceback.
    """


def default_config() -> ValidatorConfig:
    """Return an empty/default config (used when no config file exists yet)."""
    return ValidatorConfig()


def load_config(path: Optional[Path] = None) -> ValidatorConfig:
    """Load config from `path` (default: CONFIG_PATH), returning a default
    config if it doesn't exist.

    `path` defaults to None (resolved to CONFIG_PATH at call time, not
    definition time) so this always reflects the current value of
    CONFIG_PATH - including in tests that patch it.

    Use this when a missing config is a legitimate, silent starting point
    (e.g. the wizard pre-populating its prompts with existing values).
    For a context where a config file is required - like `validate` - use
    require_config() instead, which raises ConfigNotFoundError.
    """
    path = path or CONFIG_PATH

    if not path.exists():
        return default_config()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ValidatorConfig.model_validate(raw)


def require_config(path: Optional[Path] = None) -> ValidatorConfig:
    """Load config from `path` (default: CONFIG_PATH), raising
    ConfigNotFoundError if it doesn't exist.

    Intended for `tablevalidator validate`: running validation before
    `tablevalidator configure` has ever been run is a user error that
    deserves a clear, actionable message rather than silently validating
    against an empty config.
    """
    path = path or CONFIG_PATH

    if not path.exists():
        raise ConfigNotFoundError(
            f"No configuration found at {path}. "
            "Run 'tablevalidator configure' first to set up your source/target "
            "tables and Databricks connection."
        )

    return load_config(path)


def save_config(config: ValidatorConfig, path: Optional[Path] = None) -> None:
    """Save `config` to `path` (default: CONFIG_PATH), creating the parent
    directory if missing."""
    path = path or CONFIG_PATH

    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json", by_alias=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
