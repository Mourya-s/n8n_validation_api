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
]
