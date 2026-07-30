"""
Azure Storage Connector

Reads CSV files from Azure Storage Blob Container and returns
Pandas DataFrames.

Contains no comparison logic.
"""

from __future__ import annotations

import logging
from io import StringIO
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class AzureConnector:
    """
    Azure Storage connector for reading CSV files from Blob Storage.
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
        Read a CSV file from Azure Storage and return a DataFrame.

        Example blob_path:
            n8ndirectory/day.csv
        """

        self.connect()

        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_path,
        )

        csv_data = (
            blob_client
            .download_blob()
            .readall()
            .decode("utf-8")
        )

        df = pd.read_csv(StringIO(csv_data))

        logger.info(
            "CSV loaded successfully | file=%s | shape=%s",
            blob_path,
            df.shape,
        )

        return df

    def get_schema(
        self,
        blob_path: str,
    ) -> pd.DataFrame:
        """
        Return schema information for a CSV file.

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