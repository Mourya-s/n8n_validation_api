"""
Data Migration Comparison Service - FastAPI Application

POC for comparing data between Azure Storage CSV files
and Databricks Delta Lake.

Exposes REST APIs intended for consumption by n8n workflows.
"""

import logging
import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from azure_connector import AzureConnector
from comparison_engine import ComparisonEngine
from databricks_connector import DatabricksConnector
from models import ComparisonRequest, ComparisonResponse


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Data Migration Comparison Service",
    description=(
        "POC service that compares Azure Storage CSV data "
        "with Databricks Delta Lake tables and returns a "
        "standardized JSON response for n8n validation."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------
def get_azure_connector() -> AzureConnector:
    """
    Create and return an Azure Storage connector.

    Required environment variables:

        AZURE_STORAGE_ACCOUNT
        AZURE_STORAGE_KEY
        AZURE_CONTAINER
    """

    account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
    account_key = os.getenv("AZURE_STORAGE_KEY")
    container_name = os.getenv("AZURE_CONTAINER")

    if not account_name:
        account_name = "n8nstorages"

    if not container_name:
        container_name = "n8ncontainer"

    if not account_key:
        logger.error("AZURE_STORAGE_KEY is not set")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Azure Storage configuration is missing",
        )

    return AzureConnector(
        account_name=account_name,
        account_key=account_key,
        container_name=container_name,
    )


def get_databricks_connector() -> DatabricksConnector:
    """
    Create and return a DatabricksConnector instance.

    Required environment variables:

        DATABRICKS_HOST
        DATABRICKS_TOKEN
        DATABRICKS_HTTP_PATH
    """

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")

    if not host or not token:
        logger.error(
            "DATABRICKS_HOST and/or DATABRICKS_TOKEN are not configured"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Databricks configuration is missing",
        )

    return DatabricksConnector(
        host=host,
        token=token,
        http_path=http_path,
    )


def get_comparison_engine(
    azure: Annotated[AzureConnector, Depends(get_azure_connector)],
    databricks: Annotated[DatabricksConnector, Depends(get_databricks_connector)],
) -> ComparisonEngine:
    """
    Create comparison engine.
    """

    return ComparisonEngine(
        azure_connector=azure,
        databricks_connector=databricks,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    summary="Health Check",
    response_description="Service health status",
    tags=["System"],
)
async def health() -> dict[str, str]:
    """
    Health check endpoint.
    """

    return {"status": "healthy"}


@app.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare Azure Storage CSV and Databricks",
    response_description="Standardized comparison result",
    tags=["Comparison"],
    status_code=status.HTTP_200_OK,
)
async def compare(
    request: ComparisonRequest,
    engine: Annotated[ComparisonEngine, Depends(get_comparison_engine)],
) -> ComparisonResponse:
    """
    Compare source CSV data from Azure Storage
    against target Databricks Delta data.
    """

    logger.info(
        "Comparison request received | source=%s | target=%s",
        request.source_table,
        request.target_table,
    )

    try:
        result = engine.compare(request)

        logger.info("Comparison completed successfully")

        return result

    except ValueError as exc:

        logger.warning(
            "Validation error during comparison: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except ConnectionError as exc:

        logger.error(
            "Connectivity error during comparison: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception:

        logger.exception(
            "Unexpected error during comparison"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during comparison",
        )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):

    logger.exception(
        "Unhandled exception on %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Local development entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )