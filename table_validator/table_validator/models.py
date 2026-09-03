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
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from table_validator.config.schema import ValidationType


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


class ValidationTier(int, Enum):
    """
    How far the tiered fail-fast funnel got for one table (Databricks ->
    Databricks path only, see CatalogValidator). Each tier only runs if
    every cheaper tier before it failed to conclusively answer "are these
    tables equal" - a table's tier_reached tells a reader exactly how much
    work was (and, just as importantly, was NOT) done to reach its
    verdict, which the previous "everything always runs" pipeline had no
    way to express.

    Tier 3 (partition/bucket fingerprinting) is deferred to a follow-up;
    a mismatch at Tier 2 goes straight to Tier 4 over the whole table.
    """

    SCHEMA_BLOCKED = 0  # Tier 0 found a BLOCKING schema diff; aborted, no further tier ran
    SCHEMA_ONLY = 1     # Tier 0 was the final word (e.g. ROW validation disabled)
    STATISTICAL = 2     # Tier 1 was the final word (mismatch, or max_tier capped here)
    FINGERPRINT = 3     # Tier 2 was the final word (whole-table fingerprint matched)
    ROW_HASH = 4        # Tier 4 ran (per-key row-hash diff)
    COLUMN_DIFF = 5     # Tier 5 ran (column-level diff of mismatched keys)


class HashCanonicalizationSpec(BaseModel):
    """
    Canonicalization rules shared by every hashing tier (Tier 2 whole-table
    fingerprint, Tier 4 per-key row hash) so they can never disagree with
    each other about what "the same row" hashes to.

    Defaults reproduce the pre-existing, empirically-verified hash
    expression byte-for-byte (see databricks_connector.get_row_hashes'
    history and table_validator/CLAUDE.md's note on sha2()/hashlib.sha256
    cross-dialect equivalence) - nothing changes until a caller
    deliberately constructs a non-default spec. Not yet exposed via CLI/
    wizard: these knobs are correctness-sensitive and need dedicated
    verification against real data before users can toggle them.
    """

    null_sentinel: str = "\x01NULL\x01"
    trim_strings: bool = False
    case_sensitive: bool = True
    float_rounding_decimals: Optional[int] = None
    normalize_negative_zero: bool = False
    unicode_nfc_normalize: bool = False

    model_config = {"extra": "forbid"}


class PartitionPromptContext(BaseModel):
    """
    Passed to CatalogValidator's optional partition_prompt callback when a
    table is both large (row count over request.partition_threshold) and
    has a confirmed mismatch (Tier 1 and/or Tier 2) - i.e. exactly the
    case where an unpartitioned Tier 4 row-hash diff would be expensive
    and a bucketed comparison is worth offering. The callback returns the
    chosen partition column name, or None to decline (falls back to
    today's unpartitioned Tier 4 over the whole table).

    Kept as a plain data-carrier with no behavior, so the validator (pure
    decision logic, no I/O) and the CLI (the only place that actually
    prompts a human) stay cleanly separated - the validator only ever
    calls this callback and reads its return value.
    """

    schema_name: str
    table: str
    row_count: int
    candidate_columns: List[str]

    model_config = {"extra": "forbid"}


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

    schema_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of source-catalog schema name -> target-catalog "
            "schema name, for when the user has explicitly named a source "
            "and target schema that don't share the same name. An explicit "
            "pair like this is compared directly, bypassing name-based "
            "schema matching entirely (unlike `schemas`, which still "
            "requires the name to appear in the intersection). Unmapped "
            "schemas are matched by identical name as usual."
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

    table_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of source-catalog table name -> target-catalog "
            "table name, for when the user has explicitly named a source "
            "and target table that don't share the same name - an explicit "
            "pair like this is compared directly, bypassing name-based "
            "table matching entirely (unlike `tables`, which still requires "
            "the name to appear in the intersection). Unmapped tables are "
            "matched by identical name as usual."
        ),
    )

    ignore_columns: List[str] = Field(default_factory=list)

    only_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist: if set, common_cols is further restricted "
            "to just these column names (case-insensitive) before any "
            "check runs - every other common column is excluded from name/"
            "type/nullable/statistics/row-hash checks entirely, the same "
            "as ignore_columns' exclusion. A name here that isn't actually "
            "a common column is silently absent from the effective set, "
            "matching this class's other restriction fields (schemas/ "
            "tables). If a column appears in both only_columns and "
            "ignore_columns, ignore_columns wins - it is always excluded, "
            "applied after the allowlist restriction."
        ),
    )

    ignore_datatype_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Columns (case-insensitive) whose data_type_status should be "
            "reported as SKIPPED rather than PASS/FAIL, and excluded from "
            "Tier 0's BLOCKING cross-family-type-change classification - "
            "a real type difference on one of these columns never fails "
            "the table or aborts the schema stage. The column's other "
            "checks (nullable, null/distinct/min-max statistics, row-"
            "hash) still run normally."
        ),
    )

    column_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of source-table column name -> target-table "
            "column name, for when an individual column was renamed "
            "between source and target (e.g. source has 'cust_id', "
            "target has 'customer_id'). An explicit pair like this is "
            "treated as fully equivalent through the entire pipeline - "
            "schema/type/nullable checks, null/distinct/min-max "
            "statistics, whole-table fingerprint, row-hash diff, and "
            "column-level mismatch detail - bypassing name-based column "
            "matching entirely for that pair (unlike ignore_columns/"
            "only_columns, which still require the name to appear in the "
            "intersection). Unmapped columns are matched by identical "
            "name as usual. Resolved before only_columns/ignore_columns/"
            "ignore_datatype_columns apply, so those three act on the "
            "resolved/canonical (target-side) name. A mapped name that "
            "doesn't actually exist on either side produces a clear "
            "error rather than a silent no-op. Known limitation: a "
            "column configured as a primary key, or used as a Tier 3 "
            "partition/bucket column, must not also appear here."
        ),
    )

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

    max_tier: ValidationTier = Field(
        default=ValidationTier.COLUMN_DIFF,
        description=(
            "Ceiling for the tiered fail-fast funnel (Tier 0 schema -> "
            "Tier 1 statistics -> Tier 2 fingerprint -> Tier 4 row-hash -> "
            "Tier 5 column diff). STATISTICAL stops after Tier 1 even if "
            "it matches (the --mode=stats CLI flag); COLUMN_DIFF (default) "
            "lets the funnel run all the way to a column-level diff if "
            "cheaper tiers can't already prove the tables equal. Only "
            "takes effect when ROW validation is enabled - it is inert "
            "otherwise, same as data_compare_mode was."
        ),
    )

    max_sample_rows: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Max sample mismatched rows returned per table in FULL mode.",
    )

    partition_threshold: int = Field(
        default=1_000_000,
        ge=1,
        description=(
            "Row-count threshold above which a table with a confirmed "
            "mismatch (Tier 1 and/or Tier 2) is offered for partitioned "
            "Tier 4 row-hash comparison via the partition_prompt callback, "
            "instead of an unpartitioned whole-table scan. Below this "
            "threshold, or when partition_prompt is not supplied/declines, "
            "Tier 4 always runs unpartitioned as before."
        ),
    )

    enabled_validations: Set[ValidationType] = Field(
        default_factory=lambda: {
            ValidationType.CATALOG,
            ValidationType.SCHEMA,
            ValidationType.COLUMN,
            ValidationType.ROW,
        },
        description=(
            "Which validation types actually count toward a table's/"
            "catalog's overall status and appear in the report. "
            "CATALOG existence and SCHEMA/table discovery always execute "
            "regardless of this setting (discovery is a prerequisite for "
            "COLUMN/ROW checks - there is nothing to compare rows of "
            "without first knowing which tables exist), but their "
            "missing/extra findings are excluded from the status rollup "
            "and report when deselected. COLUMN (name/type/order/"
            "nullable/statistics) and ROW (row count/row-hash) checks are "
            "skipped outright when deselected."
        ),
    )

    @field_validator(
        "schemas", "tables", "ignore_columns", "only_columns",
        "ignore_datatype_columns", mode="before",
    )
    @classmethod
    def _ensure_list(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return [value]
        return list(value)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Azure Blob CSV -> single Databricks table validation request
#
# Reuses the exact same result shape (CatalogValidationResponse, wrapping
# one SchemaValidationResult with one TableValidationResult) as the
# catalog-to-catalog path above, so report_generator.py needs no changes -
# it just sees "one schema, one table" instead of many.
# ---------------------------------------------------------------------------
class CsvTableValidationRequest(BaseModel):

    source_blob_path: str = Field(
        ..., min_length=1,
        description="Path to the source CSV inside the Azure Storage container, e.g. 'validation/customers.csv'.",
    )

    target_catalog: str = Field(..., min_length=1)
    target_schema: str = Field(..., min_length=1)
    target_table: str = Field(..., min_length=1)

    primary_key: List[str] = Field(
        default_factory=list,
        description=(
            "Primary/business key column(s), used for row-hash and row-level "
            "data comparison. If omitted, comparison falls back to a "
            "synthetic row-number match (CSV file order vs. a Databricks "
            "ROW_NUMBER() over the same column order) - best-effort only, "
            "not a substitute for a real key."
        ),
    )

    ignore_columns: List[str] = Field(default_factory=list)

    case_sensitive_columns: bool = Field(default=False)
    validate_column_order: bool = Field(default=True)

    data_compare_mode: DataCompareMode = Field(
        default_factory=DataCompareMode.default
    )

    max_sample_rows: int = Field(default=50, ge=1, le=1000)

    @field_validator("primary_key", "ignore_columns", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return [value]
        return list(value)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Azure SQL Database -> Databricks catalog validation request
#
# Multi-table, matched by name: every table in the Azure SQL database
# (optionally restricted to `schemas`/`tables`) is matched against a
# like-named table in the target Databricks catalog/schema, mirroring
# CatalogValidationRequest's schema/table matching. Returns the same
# CatalogValidationResponse shape as CatalogValidator, so
# report_generator.py needs no changes.
# ---------------------------------------------------------------------------
class AzureSqlValidationRequest(BaseModel):

    target_catalog: str = Field(..., min_length=1)

    schemas: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict validation to these Azure SQL schemas (matched to "
            "same-named Databricks schemas, or remapped via schema_map). "
            "If omitted, all schemas common to both sides are validated."
        ),
    )

    schema_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of Azure SQL schema name -> Databricks schema "
            "name, for when the two sides use different schema names for "
            "the same logical migration target (e.g. Azure SQL's default "
            "'dbo' vs a purpose-named Databricks schema). Unmapped "
            "schemas are matched by identical name as usual."
        ),
    )

    tables: Optional[List[str]] = Field(
        default=None,
        description="Restrict validation to these table names. If omitted, all common tables are validated.",
    )

    table_map: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of Azure SQL table name -> Databricks table "
            "name, for when the user has explicitly named a source and "
            "target table that don't share the same name - an explicit "
            "pair like this is compared directly, bypassing name-based "
            "table matching entirely (unlike `tables`, which still "
            "requires the name to appear in the intersection). Unmapped "
            "tables are matched by identical name as usual."
        ),
    )

    ignore_columns: List[str] = Field(default_factory=list)

    case_sensitive_columns: bool = Field(default=False)
    validate_column_order: bool = Field(default=True)

    primary_keys: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Map of 'schema.table' -> key column list, used for row-hash "
            "and row-level data comparison. Tables without an entry get "
            "schema/row-count/statistics validation only."
        ),
    )

    data_compare_mode: DataCompareMode = Field(
        default_factory=DataCompareMode.default
    )

    max_sample_rows: int = Field(default=50, ge=1, le=1000)

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

    # Populated only when column_map actually renamed this column (source
    # name differs from the target/canonical name shown above) - mirrors
    # TableValidationResult.source_table_name's "only set when it
    # differs" convention.
    source_column: Optional[str] = None

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

    # Populated only when column_map actually renamed this column -
    # mismatch_column stays the canonical/target-side name (existing
    # report rendering, existing tests, unaffected).
    source_mismatch_column: Optional[str] = None

    source_value: Optional[Any] = None
    target_value: Optional[Any] = None

    source_row_hash: Optional[Any] = None
    target_row_hash: Optional[Any] = None

    # Populated by reports/excel_report.py's classification pass (a
    # report-generation-time enrichment, not something the validator
    # itself computes) - one of validators.mismatch_classifier's category
    # labels (NULL_MISMATCH, STRING_TRUNCATION, ...), or None if
    # classification hasn't run yet (e.g. for a caller inspecting the raw
    # CatalogValidationResponse straight from CatalogValidator, before
    # any report was generated).
    mismatch_category: Optional[str] = None

    verified: bool = Field(
        default=True,
        description=(
            "True when this row was pinpointed via a real configured "
            "primary key (Tier 5's standard path). False when derived "
            "from the ROW_NUMBER() fallback used when no primary key is "
            "configured - 'row N' on the source and target are only the "
            "same logical record if both sides otherwise contain the "
            "same row set; treat these rows as best-effort, not a "
            "confirmed per-record diff."
        ),
    )

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
    status: str  # MISMATCH | MISSING_IN_TARGET | MISSING_IN_SOURCE | DUPLICATE_KEY

    partition_bucket: Optional[str] = Field(
        default=None,
        description=(
            "Which partition bucket this mismatch was found in, when the "
            "table was compared via partitioned Tier 4 (see "
            "TableValidationResult.partition_column). None for an "
            "unpartitioned (whole-table) row-hash comparison."
        ),
    )

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

    # Tier 2 whole-table fingerprint (Databricks -> Databricks tiered
    # funnel only). fingerprint_status PASS means Tier 4/5 were skipped
    # because the fingerprint already proved the tables equal.
    fingerprint_status: Optional[ValidationStatus] = None
    source_fingerprint: Optional[str] = None
    target_fingerprint: Optional[str] = None

    # Tier 4: keys that appear more than once on one side (row-hash diff
    # cannot classify a duplicated key the same way as a unique one).
    duplicate_keys: List[str] = Field(default_factory=list)

    note: Optional[str] = None
    error: Optional[str] = None

    model_config = {"extra": "ignore"}


class TableValidationResult(BaseModel):

    schema_name: str
    table: str
    status: ValidationStatus = ValidationStatus.SKIPPED

    # Populated by CatalogValidator/AzureSqlValidator whenever the
    # source-side schema/table name differs from the target-side name
    # shown above (e.g. an explicit schema_map/table_map pair) - lets a
    # future report enhancement show both names. schema_name/table above
    # always reflect the TARGET-side name (the reporting convention).
    source_schema_name: Optional[str] = None
    source_table_name: Optional[str] = None

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

    # Tiered fail-fast funnel bookkeeping (Databricks -> Databricks path
    # only). tier_reached tells a reader exactly how much work was done
    # to reach this table's verdict; schema_blocking distinguishes an
    # aborted table (no row-level tier ever ran) from one that merely has
    # a non-blocking schema note alongside a real row-level result.
    tier_reached: ValidationTier = ValidationTier.SCHEMA_ONLY
    tier_stop_reason: Optional[str] = None
    schema_blocking: bool = False

    # Partitioned Tier 4 bookkeeping (Databricks -> Databricks path only).
    # Describes HOW Tier 4 was scoped for a large, confirmed-mismatched
    # table - orthogonal to tier_reached, which still just says how far
    # the funnel went (ROW_HASH/COLUMN_DIFF), partitioned or not.
    partitioned: bool = False
    partition_column: Optional[str] = None
    partition_buckets_total: Optional[int] = None
    partition_buckets_culprit: Optional[int] = None
    partition_skip_reason: Optional[str] = Field(
        default=None,
        description=(
            "Why a large, confirmed-mismatched table was NOT partitioned "
            "(e.g. 'below partition_threshold', 'no partition_prompt "
            "configured', 'user declined', '--yes flag / non-interactive "
            "run'). None when the table was too small to be offered "
            "partitioning at all, or when it was partitioned successfully."
        ),
    )

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
