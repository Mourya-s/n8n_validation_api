"""
Pydantic models for the Data Migration Comparison Service.

Defines request / response contracts used by the FastAPI layer and the
comparison engine.

This file contains two families of models:

1. Existing CSV-vs-Databricks row comparison models (CompareRequest /
   ComparisonResult / etc.) - UNCHANGED from the original implementation.

2. New Databricks catalog-to-catalog validation models, added to support
   CatalogValidator in comparison_engine.py. These are intentionally kept
   separate (different enum, different result shape) rather than
   shoehorned into the existing ComparisonStatus / ComparisonResult
   models, since a catalog validation run produces a tree of results
   (catalog -> schemas -> tables -> columns) rather than a single flat
   comparison.
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


# ===========================================================================
# NEW: Databricks catalog-to-catalog validation models
# ===========================================================================
class ValidationStatus(str, Enum):
    """
    Status for a single validation stage or an aggregated object
    (column / table / schema / catalog).

    Distinct from ComparisonStatus (PASS/WARN/FAIL) because the catalog
    validator needs to separate genuine validation failures from
    technical errors (permission denied, connection dropped, etc.) and
    from stages that were intentionally skipped by configuration.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class DataCompareMode(str, Enum):
    """
    Controls how expensive stage 15 (actual row-level data comparison) is.
    Everything runs as push-down SQL against Databricks; no full-table
    collect() / toPandas() is ever performed regardless of mode.
    """

    COUNT_ONLY = "COUNT_ONLY"   # row count only, skip null/distinct/minmax/data
    STATISTICS = "STATISTICS"   # null/distinct/minmax, skip row-level data compare
    HASH = "HASH"                # key + row-hash based row compare (pushed down)
    FULL = "FULL"                 # key-based anti-join, returns sample mismatched rows

    @classmethod
    def default(cls) -> "DataCompareMode":
        # Safe default for large datasets: aggregate statistics, no
        # row-level comparison.
        return cls.STATISTICS


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class CatalogValidationRequest(BaseModel):

    source_catalog: str = Field(..., min_length=1)
    target_catalog: str = Field(..., min_length=1)

    schemas: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict validation to these schemas. If omitted, all "
            "schemas common to both catalogs are validated."
        ),
    )

    tables: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict validation to these table names (applies within "
            "every validated schema). If omitted, all common tables are "
            "validated."
        ),
    )

    ignore_columns: List[str] = Field(default_factory=list)

    case_sensitive_columns: bool = Field(
        default=False,
        description="Case-sensitive column name comparison.",
    )

    validate_column_order: bool = Field(
        default=True,
        description="If False, column order differences do not fail a table.",
    )

    validate_nullable: bool = Field(default=True)

    primary_keys: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Optional map of 'schema.table' -> key column list, used for "
            "row-level data comparison (HASH / FULL modes). Tables without "
            "an entry fall back to a safe COUNT_ONLY-style comparison for "
            "stage 15."
        ),
    )

    data_compare_mode: DataCompareMode = Field(
        default_factory=DataCompareMode.default
    )

    max_sample_rows: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Max sample mismatched rows returned per table in FULL mode.",
    )

    @field_validator("schemas", "tables", "ignore_columns", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return [value]
        return list(value)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Result building blocks
# ---------------------------------------------------------------------------
class ColumnValidationResult(BaseModel):

    column: str
    status: ValidationStatus

    source_data_type: Optional[str] = None
    target_data_type: Optional[str] = None
    data_type_status: Optional[ValidationStatus] = None

    source_nullable: Optional[bool] = None
    target_nullable: Optional[bool] = None
    nullable_status: Optional[ValidationStatus] = None

    source_null_count: Optional[int] = None
    target_null_count: Optional[int] = None
    null_count_status: Optional[ValidationStatus] = None

    source_distinct_count: Optional[int] = None
    target_distinct_count: Optional[int] = None
    distinct_count_status: Optional[ValidationStatus] = None

    source_min: Optional[Any] = None
    source_max: Optional[Any] = None
    target_min: Optional[Any] = None
    target_max: Optional[Any] = None
    min_max_status: Optional[ValidationStatus] = None

    error: Optional[str] = None

    model_config = {"extra": "ignore"}


class RowMismatchDetail(BaseModel):
    """
    Per-row, per-column detail for one changed row (matching key, differing
    value) surfaced by stage 15 in HASH/FULL mode. One instance is produced
    per mismatched column within a row - a row with 3 differing columns
    yields 3 RowMismatchDetail entries sharing the same key/row hashes.
    """

    schema_name: str
    table: str

    primary_key: Dict[str, Any] = Field(default_factory=dict)
    mismatch_column: str

    source_value: Optional[Any] = None
    target_value: Optional[Any] = None

    source_row_hash: Optional[Any] = None
    target_row_hash: Optional[Any] = None

    model_config = {"extra": "ignore"}


class RowHashMismatch(BaseModel):
    """
    One primary-key's outcome from the row-hash comparison stage: either a
    mismatched whole-row hash (key present on both sides, hashes differ),
    or a key present on only one side. Never exposes row position/order -
    the primary key is always the identity used.
    """

    primary_key: str
    source_hash: str
    target_hash: str
    status: str  # MISMATCH | MISSING_IN_TARGET | MISSING_IN_SOURCE

    model_config = {"extra": "ignore"}


class DataValidationResult(BaseModel):
    """Result of stage 15 (actual row-level data comparison)."""

    mode: DataCompareMode
    status: ValidationStatus

    source_only_rows: Optional[int] = None
    target_only_rows: Optional[int] = None
    changed_rows: Optional[int] = None

    key_columns: List[str] = Field(default_factory=list)
    sample_source_only: List[Dict[str, Any]] = Field(default_factory=list)
    sample_target_only: List[Dict[str, Any]] = Field(default_factory=list)
    sample_changed: List[Dict[str, Any]] = Field(default_factory=list)
    sample_changed_detail: List[RowMismatchDetail] = Field(default_factory=list)

    # Row-hash comparison stage (separate mechanism from the EXCEPT/hash-join
    # diff above) - pushed-down, per-key whole-row hash comparison. Primary
    # mechanism for detecting row-level mismatches when a key is configured.
    row_hash_mismatches: List[RowHashMismatch] = Field(default_factory=list)
    row_hash_mismatch_count: int = 0
    row_hash_mismatch_percentage: float = 0.0

    note: Optional[str] = None
    error: Optional[str] = None

    model_config = {"extra": "ignore"}


class TableValidationResult(BaseModel):

    schema_name: str
    table: str
    status: ValidationStatus = ValidationStatus.SKIPPED

    exists_in_source: bool = True
    exists_in_target: bool = True

    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    columns_status: ValidationStatus = ValidationStatus.SKIPPED

    column_order_status: ValidationStatus = ValidationStatus.SKIPPED
    source_column_order: List[str] = Field(default_factory=list)
    target_column_order: List[str] = Field(default_factory=list)

    row_count_source: Optional[int] = None
    row_count_target: Optional[int] = None
    row_count_difference: Optional[int] = None
    row_count_status: ValidationStatus = ValidationStatus.SKIPPED

    columns: List[ColumnValidationResult] = Field(default_factory=list)
    data_types_status: ValidationStatus = ValidationStatus.SKIPPED
    nullable_status: ValidationStatus = ValidationStatus.SKIPPED
    null_counts_status: ValidationStatus = ValidationStatus.SKIPPED
    distinct_counts_status: ValidationStatus = ValidationStatus.SKIPPED
    min_max_status: ValidationStatus = ValidationStatus.SKIPPED

    data: Optional[DataValidationResult] = None

    error: Optional[str] = None

    model_config = {"extra": "ignore"}


class SchemaValidationResult(BaseModel):

    schema_name: str
    status: ValidationStatus

    exists_in_source: bool = True
    exists_in_target: bool = True

    missing_tables: List[str] = Field(default_factory=list)
    extra_tables: List[str] = Field(default_factory=list)

    tables: List[TableValidationResult] = Field(default_factory=list)

    error: Optional[str] = None

    model_config = {"extra": "ignore"}


class ValidationSummary(BaseModel):

    total_schemas: int = 0
    passed_schemas: int = 0
    failed_schemas: int = 0

    total_tables: int = 0
    passed_tables: int = 0
    failed_tables: int = 0
    error_tables: int = 0
    missing_tables: int = 0
    extra_tables: int = 0

    model_config = {"extra": "ignore"}


class CatalogValidationResponse(BaseModel):

    source_catalog: str
    target_catalog: str
    status: ValidationStatus

    validation_timestamp: Optional[str] = Field(
        default=None,
        description="UTC ISO-8601 timestamp when this validation run started.",
    )

    execution_time_seconds: float = Field(default=0.0, ge=0.0)

    missing_schemas: List[str] = Field(default_factory=list)
    extra_schemas: List[str] = Field(default_factory=list)

    summary: ValidationSummary = Field(default_factory=ValidationSummary)

    schemas: List[SchemaValidationResult] = Field(default_factory=list)

    error: Optional[str] = None

    model_config = {"extra": "ignore"}