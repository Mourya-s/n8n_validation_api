"""
Azure Storage Connector

Reads data files from Azure Storage Blob Container and returns
Pandas DataFrames. Supports CSV, Excel (.xlsx/.xls), and Parquet -
auto-detected from the blob path's extension - since a blob's file
format has nothing to do with the target Databricks table's own storage
format (always queried via SQL regardless of what format Databricks
stores it in internally).

Contains no comparison logic.
"""

from __future__ import annotations

import logging
from io import BytesIO, StringIO
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class AzureConnector:
    """
    Azure Storage connector for reading CSV/Excel/Parquet files from
    Blob Storage.
    """

    def __init__(
        self,
        account_name: str,
        account_key: str,
        container_name: str,
    ) -> None:

        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name

        self.blob_service_client: Optional[BlobServiceClient] = None

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------
    def connect(self) -> None:

        if self.blob_service_client is not None:
            return

        try:
            account_url = (
                f"https://{self.account_name}.blob.core.windows.net"
            )

            self.blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=self.account_key,
            )

            logger.info(
                "Successfully connected to Azure Storage Account: %s",
                self.account_name,
            )

        except Exception as exc:
            logger.exception(
                "Failed to connect to Azure Storage"
            )

            raise ConnectionError(
                f"Unable to connect to Azure Storage: {exc}"
            ) from exc

    def disconnect(self) -> None:

        self.blob_service_client = None

        logger.info("Azure Storage connection released")

    def test_connection(self) -> bool:

        try:
            self.connect()

            container_client = (
                self.blob_service_client.get_container_client(
                    self.container_name
                )
            )

            container_client.get_container_properties()

            logger.info(
                "Azure Storage connection test successful"
            )

            return True

        except Exception as exc:

            logger.error(
                "Azure Storage connection test failed: %s",
                exc,
            )

            return False

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------
    def read_csv(
        self,
        blob_path: str,
    ) -> pd.DataFrame:
        """
        Read a data file from Azure Storage and return a DataFrame.

        Despite the name (kept for backward compatibility - existing
        callers all say "read_csv"), the format is auto-detected from
        blob_path's extension: .csv/.txt -> CSV, .xlsx/.xls -> Excel,
        .parquet -> Parquet. The source file's format is independent of
        the target Databricks table's own storage format, which is
        always queried via SQL regardless.

        Example blob_path:
            n8ndirectory/day.csv
            n8ndirectory/day.xlsx
            n8ndirectory/day.parquet
        """

        self.connect()

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_path,
        )

        raw_bytes = blob_client.download_blob().readall()

        lower_path = blob_path.lower()

        if lower_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(raw_bytes))
        elif lower_path.endswith(".parquet"):
            df = pd.read_parquet(BytesIO(raw_bytes))
        elif lower_path.endswith((".csv", ".txt")):
            df = pd.read_csv(StringIO(raw_bytes.decode("utf-8")))
        else:
            raise ValueError(
                f"Unsupported file type for blob '{blob_path}'. "
                "Supported extensions: .csv, .txt, .xlsx, .xls, .parquet"
            )

        logger.info(
            "File loaded successfully | file=%s | shape=%s",
            blob_path,
            df.shape,
        )

        return df

    def get_schema(
        self,
        blob_path: str,
    ) -> pd.DataFrame:
        """
        Return schema information for a source file (any supported format).

        Returns:
            column_name
            data_type
        """

        df = self.read_csv(blob_path)

        schema_df = pd.DataFrame(
            {
                "column_name": df.columns,
                "data_type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
            }
        )

        return schema_df

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> "AzureConnector":

        self.connect()

        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ) -> None:

        self.disconnect()