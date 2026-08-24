"""
Data Migration Comparison Service - FastAPI Application

POC for comparing data between Azure Storage CSV files
and Databricks Delta Lake, plus Databricks catalog-to-catalog validation.

Exposes REST APIs intended for consumption by n8n workflows.
"""

import argparse
import logging
import os
import sys
import tempfile
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

from azure_connector import AzureConnector
from comparison_engine import CatalogValidator, ComparisonEngine
from databricks_connector import DatabricksConnector
from models import (
    CatalogValidationRequest,
    CatalogValidationResponse,
    ComparisonRequest,
    ComparisonResponse,
)
from report_generator import generate_csv_report, generate_excel_report
from dotenv import load_dotenv
load_dotenv()


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
        "with Databricks Delta Lake tables, validates Databricks "
        "catalog-to-catalog migrations, and returns standardized "
        "JSON / CSV results for n8n validation."
    ),
    version="0.2.0",
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


def get_catalog_validator(
    databricks: Annotated[DatabricksConnector, Depends(get_databricks_connector)],
) -> CatalogValidator:
    """
    Create the Databricks catalog-to-catalog validator.
    No Azure dependency - this path is Databricks-only.
    """

    return CatalogValidator(databricks_connector=databricks)


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


@app.post(
    "/validate-catalogs",
    response_model=CatalogValidationResponse,
    summary="Validate a Databricks catalog against another (JSON result)",
    response_description="Structured catalog-to-catalog validation result",
    tags=["Catalog Validation"],
    status_code=status.HTTP_200_OK,
)
async def validate_catalogs_endpoint(
    request: CatalogValidationRequest,
    validator: Annotated[CatalogValidator, Depends(get_catalog_validator)],
) -> CatalogValidationResponse:
    """
    Recursively validate source_catalog against target_catalog:
    catalog -> schemas -> tables -> columns -> data.
    """

    logger.info(
        "Catalog validation request received | source=%s | target=%s",
        request.source_catalog,
        request.target_catalog,
    )

    try:
        result = validator.compare_catalogs(request)
        logger.info("Catalog validation completed | status=%s", result.status)
        return result

    except ConnectionError as exc:
        logger.error("Connectivity error during catalog validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected error during catalog validation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred during catalog validation",
        )


@app.post(
    "/validate-catalogs/report",
    summary="Validate a Databricks catalog against another and download a CSV or Excel report",
    response_description="CSV or Excel file with the per-table validation results",
    tags=["Catalog Validation"],
    status_code=status.HTTP_200_OK,
)
async def validate_catalogs_report_endpoint(
    request: CatalogValidationRequest,
    validator: Annotated[CatalogValidator, Depends(get_catalog_validator)],
    format: Literal["csv", "excel"] = "csv",
) -> FileResponse:
    """
    Same validation as /validate-catalogs, but returns the per-table
    validation results as a downloadable file instead of JSON.

    format=csv (default) returns a .csv file; format=excel returns a
    single-sheet, formatted .xlsx workbook - same columns either way.
    """

    logger.info(
        "Catalog validation + %s report requested | source=%s | target=%s",
        format,
        request.source_catalog,
        request.target_catalog,
    )

    try:
        result = validator.compare_catalogs(request)

        tmp_dir = tempfile.mkdtemp(prefix="catalog_validation_")

        if format == "excel":
            filename = (
                f"catalog_validation_{request.source_catalog}_vs_"
                f"{request.target_catalog}.xlsx"
            )
            output_path = os.path.join(tmp_dir, filename)
            generate_excel_report(result, output_path)
            media_type = (
                "application/vnd.openxmlformats-officedocument"
                ".spreadsheetml.sheet"
            )
        else:
            filename = (
                f"catalog_validation_{request.source_catalog}_vs_"
                f"{request.target_catalog}.csv"
            )
            output_path = os.path.join(tmp_dir, filename)
            generate_csv_report(result, output_path)
            media_type = "text/csv"

        return FileResponse(
            path=output_path,
            filename=filename,
            media_type=media_type,
        )

    except ConnectionError as exc:
        logger.error("Connectivity error during catalog validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected error generating catalog validation report")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred generating the validation report",
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
# CLI entry-point for catalog validation (no server required)
#
#   python app.py validate-catalogs --source-catalog A --target-catalog B \
#       [--schemas s1,s2] [--tables t1,t2] [--no-column-order] \
#       [--mode COUNT_ONLY|STATISTICS|HASH|FULL] [--csv out.csv] [--excel out.xlsx] \
#       [--primary-keys 'schema.table=col1,col2;other_table=col']
#
# Reuses the same DATABRICKS_HOST / DATABRICKS_TOKEN / DATABRICKS_HTTP_PATH
# environment variables as the FastAPI app - no separate config mechanism.
# ---------------------------------------------------------------------------
def _run_cli() -> int:

    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Data Migration Comparison Service CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser(
        "validate-catalogs",
        help="Run a Databricks catalog-to-catalog validation",
    )
    validate_parser.add_argument("--source-catalog", required=True)
    validate_parser.add_argument("--target-catalog", required=True)
    validate_parser.add_argument(
        "--schemas", default=None,
        help="Comma-separated list of schemas to restrict validation to",
    )
    validate_parser.add_argument(
        "--tables", default=None,
        help="Comma-separated list of table names to restrict validation to",
    )
    validate_parser.add_argument(
        "--no-column-order", action="store_true",
        help="Do not fail a table because of column order differences",
    )
    validate_parser.add_argument(
        "--mode", default="STATISTICS",
        choices=["COUNT_ONLY", "STATISTICS", "HASH", "FULL"],
        help="Row-level data comparison mode (default: STATISTICS)",
    )
    validate_parser.add_argument(
        "--csv", default=None,
        help="If set, write the per-table report to this .csv path",
    )
    validate_parser.add_argument(
        "--excel", default=None,
        help="If set, write the per-table report to this .xlsx path",
    )
    validate_parser.add_argument(
        "--primary-keys", default=None,
        help=(
            "Primary/business key column(s) per table, required for "
            "--mode HASH/FULL. Format: 'schema.table=col1,col2;"
            "other_schema.other_table=col'. Table name alone (no schema "
            "prefix) also matches, e.g. 'table=col'."
        ),
    )

    args = parser.parse_args()

    if args.command != "validate-catalogs":
        parser.print_help()
        return 1

    # DatabricksConnector reads DATABRICKS_HOST / DATABRICKS_TOKEN /
    # DATABRICKS_HTTP_PATH from the environment itself when not passed
    # explicitly - same config mechanism the FastAPI dependency uses.
    try:
        databricks = DatabricksConnector()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    validator = CatalogValidator(databricks_connector=databricks)

    primary_keys = {}
    if args.primary_keys:
        for entry in args.primary_keys.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            table_key, _, cols = entry.partition("=")
            if not table_key or not cols:
                print(
                    f"Configuration error: invalid --primary-keys entry '{entry}' "
                    "(expected 'schema.table=col1,col2')",
                    file=sys.stderr,
                )
                return 1
            primary_keys[table_key.strip()] = [c.strip() for c in cols.split(",") if c.strip()]

    request = CatalogValidationRequest(
        source_catalog=args.source_catalog,
        target_catalog=args.target_catalog,
        schemas=args.schemas.split(",") if args.schemas else None,
        tables=args.tables.split(",") if args.tables else None,
        validate_column_order=not args.no_column_order,
        data_compare_mode=args.mode,
        primary_keys=primary_keys,
    )

    result = validator.compare_catalogs(request)

    print(
        f"Catalog validation: {result.source_catalog} vs "
        f"{result.target_catalog} -> {result.status.value}"
    )
    print(
        f"Schemas: {result.summary.passed_schemas}/{result.summary.total_schemas} passed | "
        f"Tables: {result.summary.passed_tables}/{result.summary.total_tables} passed "
        f"({result.summary.error_tables} errors, "
        f"{result.summary.missing_tables} missing, {result.summary.extra_tables} extra)"
    )

    if args.csv:
        generate_csv_report(result, args.csv)
        print(f"CSV report written to: {args.csv}")

    if args.excel:
        generate_excel_report(result, args.excel)
        print(f"Excel report written to: {args.excel}")

    return 0 if result.status.value in ("PASS", "SKIPPED") else 2


# ---------------------------------------------------------------------------
# Local development entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "validate-catalogs":
        sys.exit(_run_cli())

    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )