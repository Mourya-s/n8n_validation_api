"""
Pydantic models for the Data Migration Comparison Service.

Defines request / response contracts used by the FastAPI layer and the
comparison engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Platform(str, Enum):
    AZURE_STORAGE = "azure_storage"
    DATABRICKS = "databricks"


class ComparisonStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Nested configuration models
# ---------------------------------------------------------------------------
class SourceConfig(BaseModel):

    platform: Platform = Field(
        default=Platform.AZURE_STORAGE,
        description="Source platform identifier",
    )

    table: str = Field(
        ...,
        min_length=1,
        description="Source CSV file path inside Azure Storage",
    )

    query: Optional[str] = Field(
        default=None,
        description="Reserved for future use.",
    )

    model_config = {"extra": "forbid"}


class TargetConfig(BaseModel):

    platform: Platform = Field(
        default=Platform.DATABRICKS,
        description="Target platform identifier",
    )

    table: str = Field(
        ...,
        min_length=1,
        description="Target Databricks table name",
    )

    query: Optional[str] = Field(
        default=None,
        description="Optional Databricks SQL query.",
    )

    model_config = {"extra": "forbid"}


class ComparisonOptions(BaseModel):

    primary_keys: List[str] = Field(
        default_factory=list,
        description="Primary key column(s).",
    )

    ignore_columns: List[str] = Field(
        default_factory=list,
        description="Columns ignored during comparison.",
    )

    compare_schema: bool = Field(
        default=True,
        description="Enable schema comparison",
    )

    compare_values: bool = Field(
        default=True,
        description="Enable value comparison",
    )

    compare_duplicates: bool = Field(
        default=True,
        description="Enable duplicate detection",
    )

    case_sensitive: bool = Field(
        default=True,
        description="Case-sensitive comparison",
    )

    trim_strings: bool = Field(
        default=True,
        description="Trim leading/trailing spaces before comparison",
    )

    numeric_tolerance: float = Field(
        default=0.0,
        ge=0.0,
        description="Numeric comparison tolerance",
    )

    sample_size: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Maximum mismatch samples returned",
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class CompareRequest(BaseModel):

    source_platform: Platform = Field(
        default=Platform.AZURE_STORAGE,
        description="Source platform",
    )

    target_platform: Platform = Field(
        default=Platform.DATABRICKS,
        description="Target platform",
    )

    source_table: str = Field(
        ...,
        min_length=1,
        description="Azure Storage CSV path (example: n8ndirectory/day.csv)",
    )

    target_table: str = Field(
        ...,
        min_length=1,
        description="Databricks table name",
    )

    source_query: Optional[str] = Field(
        default=None,
        description="Reserved for future use.",
    )

    target_query: Optional[str] = Field(
        default=None,
        description="Optional Databricks SQL query.",
    )

    primary_keys: List[str] = Field(
        default_factory=list,
        description="Primary key column(s)",
    )

    ignore_columns: List[str] = Field(
        default_factory=list,
        description="Columns ignored during comparison",
    )

    compare_schema: bool = Field(default=True)
    compare_values: bool = Field(default=True)
    compare_duplicates: bool = Field(default=True)
    case_sensitive: bool = Field(default=True)

    trim_strings: bool = Field(
        default=True,
        description="Trim whitespace before comparison",
    )

    numeric_tolerance: float = Field(
        default=0.0,
        ge=0.0,
    )

    sample_size: int = Field(
        default=20,
        ge=1,
        le=500,
    )

    @classmethod
    def from_configs(
        cls,
        source: SourceConfig,
        target: TargetConfig,
        options: Optional[ComparisonOptions] = None,
    ) -> "CompareRequest":
        opts = options or ComparisonOptions()
        return cls(
            source_platform=source.platform,
            target_platform=target.platform,
            source_table=source.table,
            target_table=target.table,
            source_query=source.query,
            target_query=target.query,
            primary_keys=opts.primary_keys,
            ignore_columns=opts.ignore_columns,
            compare_schema=opts.compare_schema,
            compare_values=opts.compare_values,
            compare_duplicates=opts.compare_duplicates,
            case_sensitive=opts.case_sensitive,
            trim_strings=opts.trim_strings,
            numeric_tolerance=opts.numeric_tolerance,
            sample_size=opts.sample_size,
        )

    @field_validator("primary_keys", "ignore_columns", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "source_platform": "azure_storage",
                "target_platform": "databricks",
                "source_table": "n8ndirectory/day.csv",
                "target_table": "for_n8n_catalog.for_n8n_scheme.day",
                "primary_keys": ["Numeric"],
                "ignore_columns": [],
                "compare_schema": True,
                "compare_values": True,
                "compare_duplicates": True,
                "case_sensitive": False,
                "trim_strings": True,
                "numeric_tolerance": 0.01,
                "sample_size": 20,
            }
        },
    }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ComparisonResult(BaseModel):

    status: ComparisonStatus = Field(
        ...,
        description="Overall comparison outcome",
    )

    execution_time: float = Field(
        ...,
        alias="execution_time_seconds",
        ge=0.0,
    )

    row_count_source: int = Field(..., ge=0)
    row_count_target: int = Field(..., ge=0)

    matched_rows: int = Field(..., ge=0)
    missing_rows: int = Field(..., ge=0)
    extra_rows: int = Field(..., ge=0)

    duplicate_rows: Dict[str, Any] = Field(
        default_factory=dict,
    )

    schema_match: bool

    column_differences: List[Dict[str, Any]] = Field(
        default_factory=list,
    )

    sample_mismatches: List[Dict[str, Any]] = Field(
        default_factory=list,
    )

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class HealthResponse(BaseModel):

    status: str = Field(
        default="healthy",
        examples=["healthy"],
    )

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
ComparisonRequest = CompareRequest
ComparisonResponse = ComparisonResult