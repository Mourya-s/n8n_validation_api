"""table_validator: validates data migrations between Azure and Databricks Delta Lake."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("table-validator")
except PackageNotFoundError:
    __version__ = "0.0.0"

from table_validator.config.manager import (
    ConfigNotFoundError,
    default_config,
    load_config,
    require_config,
    save_config,
)
from table_validator.config.schema import ValidatorConfig
from table_validator.connectors.azure_connector import AzureConnector, AzureSqlConnector
from table_validator.connectors.databricks_connector import DatabricksConnector
from table_validator.models import CatalogValidationRequest, CatalogValidationResponse
from table_validator.validators.blob_discovery import BlobCatalogValidator
from table_validator.validators.catalog_validator import CatalogValidator
from table_validator.validators.row_validator import AzureCsvValidator, AzureSqlValidator

__all__ = [
    "__version__",
    # Validators
    "CatalogValidator",
    "AzureCsvValidator",
    "AzureSqlValidator",
    "BlobCatalogValidator",
    # Connectors
    "DatabricksConnector",
    "AzureConnector",
    "AzureSqlConnector",
    # Config
    "ValidatorConfig",
    "ConfigNotFoundError",
    "default_config",
    "load_config",
    "require_config",
    "save_config",
    # Core request/response models
    "CatalogValidationRequest",
    "CatalogValidationResponse",
    # Notebook-native API (lazily loaded - see __getattr__ below; pyspark
    # is only imported when one of these two names is actually accessed).
    "SparkConnector",
    "validate_tables",
]


def __getattr__(name: str):
    """
    PEP 562 module-level lazy attribute access, for the two names above
    that transitively depend on pyspark. pyspark is NOT a hard dependency
    of this package (CLI-only users never install it) - deferring the
    import to actual first-access time, rather than importing eagerly
    like everything else in this file, keeps `import table_validator`
    working with no pyspark installed at all. Every other name in
    __all__ is already bound by the eager imports above and resolved by
    normal attribute lookup before Python ever calls this function -
    __getattr__ only runs as a fallback for names lookup didn't resolve.
    """
    if name == "SparkConnector":
        from table_validator.connectors.spark_connector import SparkConnector

        return SparkConnector
    if name == "validate_tables":
        from table_validator.notebook import validate_tables

        return validate_tables
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
