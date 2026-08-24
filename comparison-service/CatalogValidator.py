"""
Azure Data Lake Storage (ADLS Gen2) Connector

Responsible solely for reading data from an Azure Data Lake container.
Supports Parquet and CSV files (most common for migration POCs).
Contains no comparison logic.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient

logger = logging.getLogger(__name__)


class AzureDataLakeConnector:
    """
    Lightweight, reusable connector for Azure Data Lake Storage Gen2.

    Authentication options (in order of precedence):
      1. Explicit account_url + credential passed to the constructor
      2. Connection string  (AZURE_STORAGE_CONNECTION_STRING)
      3. Account URL + DefaultAzureCredential / Service Principal env vars

    The comparison engine never needs to know how authentication works.
    """

    def __init__(
        self,
        account_url: Optional[str] = None,
        container_name: Optional[str] = None,
        connection_string: Optional[str] = None,
        credential=None,
    ) -> None:
        """
        Parameters
        ----------
        account_url:
            e.g. https://<account>.dfs.core.windows.net
            Falls back to AZURE_DATALAKE_ACCOUNT_URL.
        container_name:
            Default container. Falls back to AZURE_DATALAKE_CONTAINER.
        connection_string:
            Full storage connection string. Falls back to
            AZURE_STORAGE_CONNECTION_STRING.
        credential:
            Optional azure-identity credential. If omitted, the connector
            tries DefaultAzureCredential or Service Principal env vars.
        """
        self._account_url = account_url or os.getenv("AZURE_DATALAKE_ACCOUNT_URL")
        self._container_name = container_name or os.getenv("AZURE_DATALAKE_CONTAINER")
        self._connection_string = connection_string or os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self._credential = credential

        if not self._connection_string and not self._account_url:
            raise ValueError(
                "Provide either a connection string "
                "(AZURE_STORAGE_CONNECTION_STRING) or an account URL "
                "(AZURE_DATALAKE_ACCOUNT_URL)."
            )
        if not self._container_name:
            raise ValueError(
                "Container name is required "
                "(constructor argument or AZURE_DATALAKE_CONTAINER)."
            )

        self._service_client: Optional[DataLakeServiceClient] = None
        self._file_system_client = None
        logger.debug(
            "AzureDataLakeConnector initialised | container=%s",
            self._container_name,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Create the Data Lake service + file-system clients."""
        if self._service_client is not None:
            logger.debug("Already connected – skipping connect()")
            return

        try:
            if self._connection_string:
                self._service_client = DataLakeServiceClient.from_connection_string(
                    self._connection_string
                )
            else:
                credential = self._credential or self._build_credential()
                self._service_client = DataLakeServiceClient(
                    account_url=self._account_url,
                    credential=credential,
                )

            self._file_system_client = self._service_client.get_file_system_client(
                self._container_name
            )
            # Lightweight validation – list a single path
            next(self._file_system_client.get_paths(path="/", max_results=1), None)
            logger.info(
                "Successfully connected to ADLS container '%s'",
                self._container_name,
            )
        except Exception as exc:
            self._service_client = None
            self._file_system_client = None
            logger.exception("Failed to connect to Azure Data Lake")
            raise ConnectionError(f"Unable to connect to ADLS: {exc}") from exc

    def disconnect(self) -> None:
        """Release clients (ADLS SDK is mostly stateless; this is for symmetry)."""
        self._file_system_client = None
        self._service_client = None
        logger.info("Disconnected from Azure Data Lake")

    def test_connection(self) -> bool:
        """Return True if the container is reachable."""
        try:
            self.connect()
            next(self._file_system_client.get_paths(path="/", max_results=1), None)
            logger.info("ADLS connection test succeeded")
            return True
        except Exception as exc:
            logger.error("ADLS connection test failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_credential(self):
        """Prefer Service Principal when env vars are present, else DefaultAzureCredential."""
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")

        if tenant_id and client_id and client_secret:
            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
        return DefaultAzureCredential()

    def _ensure_connected(self):
        if self._file_system_client is None:
            self.connect()
        if self._file_system_client is None:
            raise ConnectionError("ADLS file-system client is not available")
        return self._file_system_client

    def _download_file(self, path: str) -> bytes:
        """Download a single file into memory."""
        fs = self._ensure_connected()
        file_client = fs.get_file_client(path)
        download = file_client.download_file()
        return download.readall()

    def _resolve_paths(self, path: str) -> list[str]:
        """
        If `path` is a directory, return all file paths under it.
        If it is a single file, return a one-element list.
        """
        fs = self._ensure_connected()
        paths = []
        for item in fs.get_paths(path=path.lstrip("/"), recursive=True):
            if not item.is_directory:
                paths.append(item.name)
        if not paths:
            # Maybe the path itself is a file
            try:
                file_client = fs.get_file_client(path)
                if file_client.exists():
                    paths = [path.lstrip("/")]
            except Exception:
                pass
        return paths

    def _read_single_file(self, path: str) -> pd.DataFrame:
        """Read one Parquet or CSV file into a DataFrame."""
        data = self._download_file(path)
        lower = path.lower()

        if lower.endswith(".parquet") or lower.endswith(".pq"):
            return pd.read_parquet(BytesIO(data))
        if lower.endswith(".csv"):
            return pd.read_csv(BytesIO(data))
        if lower.endswith(".json") or lower.endswith(".jsonl"):
            return pd.read_json(BytesIO(data), lines=lower.endswith(".jsonl"))

        raise ValueError(
            f"Unsupported file type for path '{path}'. "
            "Supported: .parquet, .csv, .json, .jsonl"
        )

    # ------------------------------------------------------------------
    # Public data API (same surface as the other connectors)
    # ------------------------------------------------------------------
    def read_table(self, table_name: str) -> pd.DataFrame:
        """
        Read data from a path inside the container.

        `table_name` is treated as a relative path, e.g.:
            - "raw/customers/customers.parquet"
            - "raw/customers/"          (all supported files under the folder)
        """
        if not table_name or not table_name.strip():
            raise ValueError("table_name (path) must be a non-empty string")

        logger.info("Reading ADLS path '%s'", table_name)
        file_paths = self._resolve_paths(table_name)

        if not file_paths:
            raise FileNotFoundError(
                f"No files found under path '{table_name}' "
                f"in container '{self._container_name}'"
            )

        frames = []
        for fp in file_paths:
            try:
                frames.append(self._read_single_file(fp))
            except ValueError as exc:
                logger.warning("Skipping unsupported file %s: %s", fp, exc)

        if not frames:
            raise RuntimeError(f"No readable files under '{table_name}'")

        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        logger.info("Successfully read ADLS path '%s' – shape=%s", table_name, df.shape)
        return df

    def read_query(self, query: str) -> pd.DataFrame:
        """
        Not applicable for pure object storage in this POC.

        For future extension you could integrate DuckDB / Polars over the
        downloaded files. For now we raise a clear error.
        """
        raise NotImplementedError(
            "read_query is not supported on Azure Data Lake in this POC. "
            "Use read_table with a file or folder path instead."
        )

    def get_schema(self, table_name: str) -> pd.DataFrame:
        """
        Infer schema by reading a small sample of the first file under the path.
        Returns the same column shape used by the other connectors:
            column_name, data_type, is_nullable, character_maximum_length
        """
        if not table_name or not table_name.strip():
            raise ValueError("table_name (path) must be a non-empty string")

        logger.info("Inferring schema for ADLS path '%s'", table_name)
        file_paths = self._resolve_paths(table_name)
        if not file_paths:
            raise FileNotFoundError(f"No files found under '{table_name}'")

        # Read only the first file (or a row sample for large files)
        sample_df = self._read_single_file(file_paths[0])
        # Limit rows for schema inference if the file is huge
        if len(sample_df) > 1000:
            sample_df = sample_df.head(1000)

        rows = []
        for col in sample_df.columns:
            dtype = str(sample_df[col].dtype)
            nullable = bool(sample_df[col].isna().any())
            rows.append(
                {
                    "column_name": col,
                    "data_type": dtype,
                    "is_nullable": "YES" if nullable else "NO",
                    "character_maximum_length": None,
                }
            )

        schema_df = pd.DataFrame(rows)
        logger.info(
            "Schema for '%s' inferred – %d columns",
            table_name,
            len(schema_df),
        )
        return schema_df

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "AzureDataLakeConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()