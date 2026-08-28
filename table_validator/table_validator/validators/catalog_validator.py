"""
Databricks catalog-to-catalog validator.

Recursive Databricks catalog-to-catalog validation: catalog -> schemas ->
tables -> columns -> data, per the 15-stage validation sequence. All
comparisons are pushed down to Databricks SQL via DatabricksConnector; this
class never loads a full table into pandas.

Responsible for the comparison/decision logic only (PASS/FAIL/ERROR/
SKIPPED). All data retrieval is delegated to DatabricksConnector - this
class never talks to Databricks directly.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from table_validator.config.schema import ValidationType
from table_validator.connectors.databricks_connector import DatabricksConnector
from table_validator.models import (
    CatalogValidationRequest,
    CatalogValidationResponse,
    ColumnValidationResult,
    DataCompareMode,
    DataValidationResult,
    PartitionPromptContext,
    RowHashMismatch,
    RowMismatchDetail,
    SchemaValidationResult,
    TableValidationResult,
    ValidationStatus,
    ValidationSummary,
    ValidationTier,
)

PartitionPromptCallback = Callable[[PartitionPromptContext], Optional[str]]

logger = logging.getLogger(__name__)

# Type families for Tier 0's BLOCKING/NON-BLOCKING classification. Grouped
# by prefix match against the raw Databricks type string (e.g.
# "decimal(10,2)" -> matches the "decimal" family). A change *within* a
# family (e.g. int -> bigint) is a safe widening, reported but
# NON-BLOCKING; a change *across* families (e.g. string -> int) is
# BLOCKING, since the two sides may not even be comparable.
_TYPE_FAMILIES: List[Tuple[str, Tuple[str, ...]]] = [
    ("integer", ("tinyint", "smallint", "int", "bigint")),
    ("floating", ("float", "double", "decimal", "numeric")),
    ("string", ("string", "varchar", "char")),
    ("datetime", ("date", "timestamp")),
    ("boolean", ("boolean",)),
    ("binary", ("binary",)),
]


def _type_family(data_type: str) -> Optional[str]:
    dt = (data_type or "").strip().lower()
    for family, prefixes in _TYPE_FAMILIES:
        if any(dt.startswith(p) for p in prefixes):
            return family
    return None


class CatalogValidator:
    """
    Recursive Databricks catalog-to-catalog validator.

    Usage:
        validator = CatalogValidator(databricks_connector)
        result = validator.compare_catalogs(request)

        # Optionally, to offer partitioned Tier 4 on large mismatched
        # tables (see _validate_table's Tier 3 branch):
        validator = CatalogValidator(databricks_connector, partition_prompt=my_callback)
    """

    def __init__(
        self,
        databricks_connector: DatabricksConnector,
        partition_prompt: Optional[PartitionPromptCallback] = None,
    ) -> None:
        self.databricks = databricks_connector
        self.partition_prompt = partition_prompt
        logger.debug("CatalogValidator initialised")

    @staticmethod
    def _lookup_primary_key(
        request: "CatalogValidationRequest",
        schema_name: str,
        table_name: str,
    ) -> Optional[List[str]]:
        """
        Look up request.primary_keys (keyed by "schema.table" or bare
        table name, as typed by the user into config.yaml/the wizard) for
        the given schema_name/table_name.

        schema_name/table_name here come from whatever casing Databricks'
        information_schema actually returns, matched case-insensitively
        against the catalog in compare_schemas/compare_tables - which
        does not necessarily match the exact casing the user configured.
        An exact-match dict lookup would silently miss a configured key
        and fall through to the much more expensive ROW_NUMBER() fallback
        with no indication why, so this normalizes both sides to
        lowercase before comparing.
        """
        lowered = {k.lower(): v for k, v in request.primary_keys.items()}
        key_lookup = f"{schema_name}.{table_name}".lower()
        return lowered.get(key_lookup) or lowered.get(table_name.lower())

    # ------------------------------------------------------------------
    # Stage 1 + top-level orchestration
    # ------------------------------------------------------------------
    def compare_catalogs(
        self,
        request: CatalogValidationRequest,
    ) -> CatalogValidationResponse:

        start = time.perf_counter()
        run_timestamp = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Starting catalog validation | source=%s | target=%s",
            request.source_catalog,
            request.target_catalog,
        )

        # Stage 1: catalog exists
        try:
            source_exists = self.databricks.catalog_exists(request.source_catalog)
            target_exists = self.databricks.catalog_exists(request.target_catalog)
        except Exception as exc:
            logger.exception("Failed to verify catalog existence")
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.ERROR,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Unable to verify catalog existence: {exc}",
            )

        if not source_exists or not target_exists:
            missing = []
            if not source_exists:
                missing.append(request.source_catalog)
            if not target_exists:
                missing.append(request.target_catalog)
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.FAIL,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Catalog(s) do not exist: {', '.join(missing)}",
            )

        # Stage 2/3/4: schemas
        try:
            common_schema_pairs, missing_schemas, extra_schemas, explicit_schema_errors = (
                self._resolve_schema_pairs(request)
            )
        except Exception as exc:
            logger.exception("Failed to compare schemas")
            return CatalogValidationResponse(
                source_catalog=request.source_catalog,
                target_catalog=request.target_catalog,
                status=ValidationStatus.ERROR,
                validation_timestamp=run_timestamp,
                execution_time_seconds=round(time.perf_counter() - start, 3),
                error=f"Unable to compare schemas: {exc}",
            )

        if request.schemas:
            # Explicit schema scope: missing_schemas/extra_schemas must be
            # restricted to it too, not just common_schema_pairs -
            # otherwise an unrelated schema difference elsewhere in the
            # catalog (one the user never asked to compare) falsely fails
            # this targeted run. Filtered by the SOURCE-side name,
            # matching AzureSqlValidator's convention.
            wanted = {s.lower() for s in request.schemas}
            common_schema_pairs = [
                (src, tgt) for src, tgt in common_schema_pairs if src.lower() in wanted
            ]
            missing_schemas = [s for s in missing_schemas if s.lower() in wanted]
            extra_schemas = [s for s in extra_schemas if s.lower() in wanted]
        else:
            # No schema restriction -> comparing every schema common to
            # both catalogs. Surface scope + anything present on only one
            # side up front, so a catalog-wide run is never a silent
            # surprise (missing_schemas/extra_schemas are also still
            # carried on the final response for programmatic access).
            if missing_schemas:
                logger.warning(
                    "Schemas present in source catalog '%s' but not in target "
                    "'%s' (skipped): %s",
                    request.source_catalog, request.target_catalog, missing_schemas,
                )
            if extra_schemas:
                logger.warning(
                    "Schemas present in target catalog '%s' but not in source "
                    "'%s' (skipped): %s",
                    request.target_catalog, request.source_catalog, extra_schemas,
                )
            logger.info(
                "Found %d matching schema(s) across both catalogs - comparing all.",
                len(common_schema_pairs),
            )

        schema_results: List[SchemaValidationResult] = []

        for source_schema, target_schema in common_schema_pairs:
            schema_results.append(
                self._validate_schema(request, source_schema, target_schema)
            )

        for message in explicit_schema_errors:
            logger.error(message)
            schema_results.append(
                SchemaValidationResult(
                    schema_name="(unresolved schema mapping)",
                    status=ValidationStatus.ERROR,
                    exists_in_source=False,
                    exists_in_target=False,
                    error=message,
                )
            )

        if not request.schemas:
            total_tables = sum(len(s.tables) for s in schema_results)
            logger.info(
                "Catalog-wide comparison scope: %d schema(s), %d table(s) total. "
                "This may take a while.",
                len(common_schema_pairs), total_tables,
            )

        summary = self._build_summary(schema_results, missing_schemas, extra_schemas)
        schema_enabled = ValidationType.SCHEMA in request.enabled_validations
        overall_status = self.calculate_overall_status(
            [s.status for s in schema_results]
            + ([ValidationStatus.FAIL] if (missing_schemas and schema_enabled) else [])
        )

        execution_time = round(time.perf_counter() - start, 3)

        tier_counts: Dict[str, int] = {}
        for schema_result in schema_results:
            for table_result in schema_result.tables:
                tier_name = table_result.tier_reached.name
                tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1
        if tier_counts:
            logger.info(
                "Tier distribution across %d table(s): %s",
                sum(tier_counts.values()),
                ", ".join(f"{name}={count}" for name, count in tier_counts.items()),
            )

        logger.info(
            "Catalog validation finished | status=%s | duration=%.3fs",
            overall_status,
            execution_time,
        )

        return CatalogValidationResponse(
            source_catalog=request.source_catalog,
            target_catalog=request.target_catalog,
            status=overall_status,
            validation_timestamp=run_timestamp,
            execution_time_seconds=execution_time,
            missing_schemas=missing_schemas,
            extra_schemas=extra_schemas,
            summary=summary,
            schemas=schema_results,
        )

    # Backward/spec-friendly alias
    def validate(self, request: CatalogValidationRequest) -> CatalogValidationResponse:
        return self.compare_catalogs(request)

    # Databricks-managed system schema, present in every catalog - never a
    # real migration target, so it is excluded from validation entirely
    # (not just skipped: never counted as common/missing/extra either).
    _EXCLUDED_SCHEMAS = {"information_schema"}

    # ------------------------------------------------------------------
    # Stage 2/3/4: schema comparison
    # ------------------------------------------------------------------
    def compare_schemas(
        self,
        source_catalog: str,
        target_catalog: str,
    ) -> Tuple[List[str], List[str], List[str]]:

        source_schemas = set(self.databricks.get_schemas(source_catalog))
        target_schemas = set(self.databricks.get_schemas(target_catalog))

        source_schemas -= {
            s for s in source_schemas if s.lower() in self._EXCLUDED_SCHEMAS
        }
        target_schemas -= {
            s for s in target_schemas if s.lower() in self._EXCLUDED_SCHEMAS
        }

        common = sorted(source_schemas & target_schemas)
        missing = sorted(source_schemas - target_schemas)   # in source, not target
        extra = sorted(target_schemas - source_schemas)     # in target, not source

        return common, missing, extra

    # ------------------------------------------------------------------
    # Schema pair resolution: identical-name intersection (compare_schemas,
    # unchanged above) PLUS request.schema_map for explicitly-named pairs
    # whose names differ between source and target. Mirrors
    # AzureSqlValidator._compare_schemas' resolution logic (row_validator.py)
    # so the two paths behave consistently, with one addition: a
    # schema_map entry naming a schema that does not actually exist on
    # either side produces an explicit error message instead of a silent
    # no-op (the bug this feature exists to fix).
    # ------------------------------------------------------------------
    def _resolve_schema_pairs(
        self,
        request: CatalogValidationRequest,
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str], List[str]]:
        """
        Returns (common_pairs, missing, extra, explicit_pair_errors).

        common_pairs is a list of (source_schema_name, target_schema_name)
        - identical-name pairs from compare_schemas(), plus any
        request.schema_map pairs that resolve to real schemas on both
        sides (using each side's real casing). missing/extra are the
        identical-name leftovers, with any name that a schema_map entry
        successfully accounted for removed (a mapped schema should not
        also show up as "missing" just because it didn't equal-match by
        name). explicit_pair_errors carries a human-readable message for
        every schema_map entry that names a schema that does not exist on
        one side or the other, so it can be surfaced as a visible
        ERROR/FAIL rather than a silent skip.
        """
        common, missing, extra = self.compare_schemas(
            request.source_catalog, request.target_catalog
        )

        if not request.schema_map:
            common_pairs = [(name, name) for name in common]
            return common_pairs, missing, extra, []

        common_pairs = [(name, name) for name in common]
        missing_set = set(missing)
        extra_set = set(extra)
        explicit_pair_errors: List[str] = []

        # Resolve against the full (unfiltered) schema lists on both
        # sides, not just common/missing/extra, since a schema_map's
        # source name might already be one of the "common" identical
        # names (a no-op remap) or might name a schema only found via
        # case-insensitive lookup.
        try:
            all_source_schemas = self.databricks.get_schemas(request.source_catalog)
        except Exception:
            all_source_schemas = []
        try:
            all_target_schemas = self.databricks.get_schemas(request.target_catalog)
        except Exception:
            all_target_schemas = []

        source_by_lower = {s.lower(): s for s in all_source_schemas}
        target_by_lower = {s.lower(): s for s in all_target_schemas}

        existing_pairs_lower = {(s.lower(), t.lower()) for s, t in common_pairs}

        for src_name, tgt_name in request.schema_map.items():
            actual_source = source_by_lower.get(src_name.lower())
            actual_target = target_by_lower.get(tgt_name.lower())

            if actual_source is None or actual_target is None:
                missing_side = []
                if actual_source is None:
                    missing_side.append(
                        f"schema '{src_name}' does not exist in source catalog "
                        f"'{request.source_catalog}'"
                    )
                if actual_target is None:
                    missing_side.append(
                        f"schema '{tgt_name}' does not exist in target catalog "
                        f"'{request.target_catalog}'"
                    )
                explicit_pair_errors.append(
                    f"Configured schema mapping '{src_name}' -> '{tgt_name}': "
                    + "; ".join(missing_side)
                )
                continue

            pair_key = (actual_source.lower(), actual_target.lower())
            if pair_key in existing_pairs_lower:
                continue

            common_pairs.append((actual_source, actual_target))
            existing_pairs_lower.add(pair_key)
            missing_set.discard(actual_source)
            extra_set.discard(actual_target)

        common_pairs.sort(key=lambda pair: pair[0])
        return (
            common_pairs,
            sorted(missing_set),
            sorted(extra_set),
            explicit_pair_errors,
        )

    def _validate_schema(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
    ) -> SchemaValidationResult:

        try:
            common_table_pairs, missing_tables, extra_tables, explicit_pair_errors = (
                self._resolve_table_pairs(request, source_schema, target_schema)
            )
        except Exception as exc:
            logger.exception(
                "Failed to compare tables for schema '%s' -> '%s'",
                source_schema, target_schema,
            )
            return SchemaValidationResult(
                schema_name=target_schema,
                status=ValidationStatus.ERROR,
                error=f"Unable to compare tables: {exc}",
            )

        if request.tables:
            # Explicit table scope: missing_tables/extra_tables must be
            # restricted to it too, not just common_table_pairs -
            # otherwise an unrelated table difference elsewhere in the
            # same schema (one the user never asked to compare) falsely
            # fails this targeted run. Filtered by the SOURCE-side name,
            # matching AzureSqlValidator._validate_schema's convention.
            wanted = {t.lower() for t in request.tables}
            common_table_pairs = [
                (src, tgt) for src, tgt in common_table_pairs if src.lower() in wanted
            ]
            missing_tables = [t for t in missing_tables if t.lower() in wanted]
            extra_tables = [t for t in extra_tables if t.lower() in wanted]
        else:
            # No table restriction -> comparing every table common to this
            # schema on both sides. Surface anything present on only one
            # side rather than silently skipping it.
            if missing_tables:
                logger.warning(
                    "Tables present in source schema '%s.%s' but not in "
                    "target (skipped): %s",
                    request.source_catalog, source_schema, missing_tables,
                )
            if extra_tables:
                logger.warning(
                    "Tables present in target schema '%s.%s' but not in "
                    "source (skipped): %s",
                    request.target_catalog, target_schema, extra_tables,
                )

        table_results: List[TableValidationResult] = []

        for source_table, target_table in common_table_pairs:
            table_results.append(
                self._validate_table(request, source_schema, target_schema, source_table, target_table)
            )

        for message in explicit_pair_errors:
            table_results.append(
                TableValidationResult(
                    schema_name=target_schema,
                    table="(unresolved table mapping)",
                    status=ValidationStatus.ERROR,
                    exists_in_source=False,
                    exists_in_target=False,
                    error=message,
                )
            )

        statuses = [t.status for t in table_results]
        if missing_tables and ValidationType.SCHEMA in request.enabled_validations:
            statuses.append(ValidationStatus.FAIL)

        status = self.calculate_overall_status(statuses)

        return SchemaValidationResult(
            schema_name=target_schema,
            status=status,
            missing_tables=missing_tables,
            extra_tables=extra_tables,
            tables=table_results,
        )

    # ------------------------------------------------------------------
    # Table pair resolution: identical-name intersection (compare_tables,
    # unchanged above) PLUS request.table_map for explicitly-named pairs
    # whose names differ between source and target. Mirrors
    # AzureSqlValidator._compare_tables' resolution logic (row_validator.py),
    # with the same "explicit pair references a name that doesn't exist"
    # error surfacing as _resolve_schema_pairs above.
    # ------------------------------------------------------------------
    def _resolve_table_pairs(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
    ) -> Tuple[List[Tuple[str, str]], List[str], List[str], List[str]]:
        """
        Returns (common_pairs, missing, extra, explicit_pair_errors), same
        shape/semantics as _resolve_schema_pairs but one level down (table
        names within an already-resolved source_schema/target_schema
        pair).
        """
        common, missing, extra = self.compare_tables(
            request.source_catalog, request.target_catalog, source_schema
        )

        # NOTE: compare_tables() above assumes the same schema name on
        # both sides. When source_schema != target_schema (a schema_map
        # pair), that identical-name baseline is meaningless - re-derive
        # it directly from each side's own schema.
        if source_schema.lower() != target_schema.lower():
            try:
                source_tables = set(self.databricks.get_tables(request.source_catalog, source_schema))
            except Exception:
                source_tables = set()
            try:
                target_tables = set(self.databricks.get_tables(request.target_catalog, target_schema))
            except Exception:
                target_tables = set()
            common = sorted(source_tables & target_tables)
            missing = sorted(source_tables - target_tables)
            extra = sorted(target_tables - source_tables)

        if not request.table_map:
            common_pairs = [(name, name) for name in common]
            return common_pairs, missing, extra, []

        common_pairs = [(name, name) for name in common]
        missing_set = set(missing)
        extra_set = set(extra)
        explicit_pair_errors: List[str] = []

        try:
            all_source_tables = self.databricks.get_tables(request.source_catalog, source_schema)
        except Exception:
            all_source_tables = []
        try:
            all_target_tables = self.databricks.get_tables(request.target_catalog, target_schema)
        except Exception:
            all_target_tables = []

        source_by_lower = {t.lower(): t for t in all_source_tables}
        target_by_lower = {t.lower(): t for t in all_target_tables}

        existing_pairs_lower = {(s.lower(), t.lower()) for s, t in common_pairs}

        for src_name, tgt_name in request.table_map.items():
            actual_source = source_by_lower.get(src_name.lower())
            actual_target = target_by_lower.get(tgt_name.lower())

            if actual_source is None or actual_target is None:
                missing_side = []
                if actual_source is None:
                    missing_side.append(
                        f"table '{src_name}' does not exist in source schema "
                        f"'{request.source_catalog}.{source_schema}'"
                    )
                if actual_target is None:
                    missing_side.append(
                        f"table '{tgt_name}' does not exist in target schema "
                        f"'{request.target_catalog}.{target_schema}'"
                    )
                explicit_pair_errors.append(
                    f"Configured table mapping '{src_name}' -> '{tgt_name}': "
                    + "; ".join(missing_side)
                )
                continue

            pair_key = (actual_source.lower(), actual_target.lower())
            if pair_key in existing_pairs_lower:
                continue

            common_pairs.append((actual_source, actual_target))
            existing_pairs_lower.add(pair_key)
            missing_set.discard(actual_source)
            extra_set.discard(actual_target)

        common_pairs.sort(key=lambda pair: pair[0])
        return (
            common_pairs,
            sorted(missing_set),
            sorted(extra_set),
            explicit_pair_errors,
        )

    # ------------------------------------------------------------------
    # Stage 3/4: table comparison
    # ------------------------------------------------------------------
    def compare_tables(
        self,
        source_catalog: str,
        target_catalog: str,
        schema_name: str,
    ) -> Tuple[List[str], List[str], List[str]]:

        source_tables = set(self.databricks.get_tables(source_catalog, schema_name))
        target_tables = set(self.databricks.get_tables(target_catalog, schema_name))

        common = sorted(source_tables & target_tables)
        missing = sorted(source_tables - target_tables)
        extra = sorted(target_tables - source_tables)

        return common, missing, extra

    # ------------------------------------------------------------------
    # Per-table pipeline: stages 5-15
    # ------------------------------------------------------------------
    def _validate_table(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
    ) -> TableValidationResult:

        # Deliberately a plain, uncluttered progress line (unlike the
        # detailed [row-hash]/stats logging further down) - this is what
        # a user watching the console during a large catalog-wide run
        # needs to see to know the tool is progressing, not stalled.
        # Uses the TARGET-side name, matching the reporting convention
        # (TableValidationResult.schema_name/table below) - unless the
        # source-side name actually differs, in which case both are
        # logged for clarity.
        if source_schema.lower() == target_schema.lower() and source_table.lower() == target_table.lower():
            logger.info("Validating table '%s.%s' ...", target_schema, target_table)
        else:
            logger.info(
                "Validating table '%s.%s' (source) -> '%s.%s' (target) ...",
                source_schema, source_table, target_schema, target_table,
            )

        result = TableValidationResult(schema_name=target_schema, table=target_table)
        if source_schema.lower() != target_schema.lower():
            result.source_schema_name = source_schema
        if source_table.lower() != target_table.lower():
            result.source_table_name = source_table

        try:
            source_schema_df = self.databricks.get_table_schema(
                request.source_catalog, source_schema, source_table
            )
            target_schema_df = self.databricks.get_table_schema(
                request.target_catalog, target_schema, target_table
            )
        except Exception as exc:
            logger.exception(
                "Failed to retrieve column metadata for '%s.%s' -> '%s.%s'",
                source_schema, source_table, target_schema, target_table,
            )
            result.status = ValidationStatus.ERROR
            result.error = f"Unable to retrieve column metadata: {exc}"
            return result

        row_enabled = ValidationType.ROW in request.enabled_validations

        # Tier 0: schema comparison. A BLOCKING difference (missing/extra
        # column, cross-family type change, missing configured PK column)
        # aborts here - no further tier runs, no further SQL. A
        # NON-BLOCKING difference (nullable, column order) is recorded
        # but execution continues into Tier 1+.
        blocking, common_cols = self._tier0_schema(
            request, target_schema, target_table, source_schema_df, target_schema_df, result,
        )

        if blocking:
            result.tier_reached = ValidationTier.SCHEMA_BLOCKED
            result.schema_blocking = True
            logger.info(
                "[tier0-schema] table=%s.%s | BLOCKING schema difference - aborting, "
                "no further tier will run",
                target_schema, target_table,
            )
            if not common_cols:
                result.error = "No common columns between source and target"
            result.status = self.calculate_overall_status(
                [
                    result.columns_status,
                    result.column_order_status,
                    result.data_types_status,
                    result.nullable_status,
                ]
            )
            return result

        if not row_enabled:
            # ROW deselected: schema-only verdict, no row-level SQL at all.
            result.tier_reached = ValidationTier.SCHEMA_ONLY
            result.row_count_status = ValidationStatus.SKIPPED
            result.data = DataValidationResult(
                mode=request.data_compare_mode,
                status=ValidationStatus.SKIPPED,
                note="Row-level comparison skipped - 'row' validation not selected.",
            )
            result.status = self.calculate_overall_status(
                [
                    result.columns_status,
                    result.column_order_status,
                    result.data_types_status,
                    result.nullable_status,
                ]
            )
            return result

        # Tier 1: statistical profile. Only the explicit --mode=stats
        # ceiling stops the funnel here. A real statistical mismatch
        # (row count, null/distinct count, min/max) no longer stops the
        # funnel by itself - it now falls through to Tier 2+ so the
        # actual differing row(s) get surfaced in the Data Mismatches /
        # Row Hash Mismatches sheets, rather than leaving a confirmed
        # difference with no row-level detail at all.
        stats_mismatch = self._tier1_statistics(
            request, source_schema, target_schema, source_table, target_table,
            common_cols, source_schema_df, result,
        )

        stats_only_ceiling = request.max_tier == ValidationTier.STATISTICAL

        if stats_only_ceiling:
            result.tier_reached = ValidationTier.STATISTICAL
            logger.info(
                "[tier1-statistics] table=%s.%s | stopping here - --mode=stats requested",
                target_schema, target_table,
            )
            result.data = DataValidationResult(
                mode=request.data_compare_mode,
                status=ValidationStatus.SKIPPED,
                note="Statistical-only mode requested - row-level comparison not run.",
            )
            result.status = self.calculate_overall_status(
                [
                    result.columns_status,
                    result.column_order_status,
                    result.row_count_status,
                    result.data_types_status,
                    result.nullable_status,
                    result.null_counts_status,
                    result.distinct_counts_status,
                    result.min_max_status,
                ]
            )
            return result

        if stats_mismatch:
            logger.info(
                "[tier1-statistics] table=%s.%s | statistical mismatch found - "
                "proceeding to Tier 2+ to locate the exact differing row(s)",
                target_schema, target_table,
            )

        # Tier 2: whole-table fingerprint. Match -> tables are equal per
        # the fingerprint, but a confirmed Tier 1 mismatch always wins:
        # trust the cheaper, already-confirmed finding and still proceed
        # to Tier 4 rather than reporting PASS on a fingerprint that
        # happens to collide (e.g. a min/max-only mismatch on a column
        # excluded from hashing).
        fingerprint_matches = self._tier2_fingerprint(
            request, source_schema, target_schema, source_table, target_table, common_cols, result,
        )

        if fingerprint_matches and not stats_mismatch:
            result.tier_reached = ValidationTier.FINGERPRINT
            logger.info(
                "[tier2-fingerprint] table=%s.%s | stopping here - fingerprint "
                "matched, tables are equal (no row-hash SQL will run)",
                target_schema, target_table,
            )
        else:
            logger.info(
                "[tier2-fingerprint] table=%s.%s | %s - proceeding to Tier 4 row-hash diff",
                target_schema, target_table,
                "mismatch" if not fingerprint_matches else "fingerprint matched but "
                "Tier 1 already confirmed a mismatch",
            )
            if fingerprint_matches and result.data is not None:
                # The fingerprint matched but Tier 1 already proved a real
                # difference exists - don't let the fingerprint's PASS
                # verdict survive into Tier 4. Reset to SKIPPED so Tier 4's
                # own status logic (FAIL on a real mismatch, PASS only if
                # it genuinely finds none) is what actually decides this,
                # not a stale, now-overridden fingerprint result.
                result.data.status = ValidationStatus.SKIPPED
                result.data.fingerprint_status = ValidationStatus.PASS
                result.data.note = (
                    "Whole-table fingerprint matched, but a statistical "
                    "mismatch (row count, null count, distinct count, or "
                    "min/max) was already found at Tier 1 - proceeding to "
                    "row-hash comparison to locate it, since it may involve "
                    "a column excluded from the fingerprint's hash."
                )
            # Tier 3 (optional, large confirmed-mismatched tables only) +
            # Tier 4 (+ Tier 5 for any ROW_HASH_MISMATCH keys).
            self._dispatch_tier4(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
            )

        # Overall table status
        result.status = self.calculate_overall_status(
            [
                result.columns_status,
                result.column_order_status,
                result.row_count_status,
                result.data_types_status,
                result.nullable_status,
                result.null_counts_status,
                result.distinct_counts_status,
                result.min_max_status,
                result.data.status if result.data else ValidationStatus.SKIPPED,
            ]
        )

        return result

    # ------------------------------------------------------------------
    # Stage 5/6: column name comparison
    # ------------------------------------------------------------------
    def compare_columns(
        self,
        source_schema_df: pd.DataFrame,
        target_schema_df: pd.DataFrame,
        case_sensitive: bool,
        ignore: Set[str],
    ) -> Tuple[List[str], List[str], List[str]]:

        def norm(name: str) -> str:
            return name if case_sensitive else name.lower()

        src_cols = {
            norm(str(c)): str(c) for c in source_schema_df["column_name"]
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
        }
        tgt_cols = {
            norm(str(c)): str(c) for c in target_schema_df["column_name"]
            if norm(str(c)) not in ignore and str(c).lower() not in ignore
        }

        missing = sorted(set(src_cols) - set(tgt_cols))
        extra = sorted(set(tgt_cols) - set(src_cols))
        common = sorted(
            src_cols[k] for k in (set(src_cols) & set(tgt_cols))
        )

        return missing, extra, common

    # ------------------------------------------------------------------
    # Stage 7: data types (per-column, used above; exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_data_types(self, source_type: str, target_type: str) -> ValidationStatus:
        return ValidationStatus.PASS if source_type == target_type else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 8: nullable (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_nullable(self, source_nullable: bool, target_nullable: bool) -> ValidationStatus:
        return ValidationStatus.PASS if source_nullable == target_nullable else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 9: column order (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_column_order(
        self, source_order: List[str], target_order: List[str]
    ) -> ValidationStatus:
        return (
            ValidationStatus.PASS
            if [c.lower() for c in source_order] == [c.lower() for c in target_order]
            else ValidationStatus.FAIL
        )

    # ------------------------------------------------------------------
    # Stage 10: row counts (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_row_counts(self, source_count: int, target_count: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_count == target_count else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 11/12: null + distinct counts (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_null_counts(self, source_nulls: int, target_nulls: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_nulls == target_nulls else ValidationStatus.FAIL

    def compare_distinct_counts(self, source_distinct: int, target_distinct: int) -> ValidationStatus:
        return ValidationStatus.PASS if source_distinct == target_distinct else ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Stage 13: min/max (exposed for reuse/tests)
    # ------------------------------------------------------------------
    def compare_min_max(
        self, source_min: Any, source_max: Any, target_min: Any, target_max: Any
    ) -> ValidationStatus:
        return (
            ValidationStatus.PASS
            if source_min == target_min and source_max == target_max
            else ValidationStatus.FAIL
        )

    # ------------------------------------------------------------------
    # Tier 0 classification: same-family widening (NON-BLOCKING) vs.
    # cross-family type change (BLOCKING). Static/pure so it's directly
    # unit-testable without a connector mock.
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_type_family(source_type: str, target_type: str) -> ValidationStatus:
        if source_type == target_type:
            return ValidationStatus.PASS

        src_family = _type_family(source_type)
        tgt_family = _type_family(target_type)

        if src_family is not None and src_family == tgt_family:
            # Same family (e.g. int -> bigint): a real, reportable
            # difference, but not one that invalidates row-level
            # comparison downstream.
            return ValidationStatus.PASS

        return ValidationStatus.FAIL

    # ------------------------------------------------------------------
    # Tier 0: schema comparison (always runs first). Classifies every
    # difference as BLOCKING (abort the whole table - never run Tier 1+)
    # or NON-BLOCKING (record the finding, continue). A schema difference
    # must never, by itself, prevent statistical/fingerprint/row-level
    # tiers from running unless it's BLOCKING.
    # ------------------------------------------------------------------
    def _tier0_schema(
        self,
        request: CatalogValidationRequest,
        schema_name: str,
        table_name: str,
        source_schema_df: pd.DataFrame,
        target_schema_df: pd.DataFrame,
        result: TableValidationResult,
    ) -> Tuple[bool, List[str]]:
        """
        Returns (blocking, common_cols). When blocking is True, the caller
        must abort the table immediately without running any further tier.
        """
        ignore = {c.lower() for c in (request.ignore_columns or [])}
        column_enabled = ValidationType.COLUMN in request.enabled_validations

        missing_cols, extra_cols, common_cols = self.compare_columns(
            source_schema_df, target_schema_df, request.case_sensitive_columns, ignore
        )

        if column_enabled:
            result.missing_columns = missing_cols
            result.extra_columns = extra_cols
            result.columns_status = (
                ValidationStatus.FAIL if (missing_cols or extra_cols) else ValidationStatus.PASS
            )
        else:
            result.columns_status = ValidationStatus.SKIPPED

        if missing_cols or extra_cols or not common_cols:
            # Missing/extra columns are always BLOCKING, regardless of
            # whether COLUMN reporting is enabled - Tier 0's detection is
            # a correctness prerequisite for every downstream tier, same
            # as compare_columns() always running today for common_cols.
            return True, common_cols

        # Configured PK column missing from either side is also BLOCKING -
        # every later tier depends on being able to resolve a usable key.
        configured_key = self._lookup_primary_key(request, schema_name, table_name)
        if configured_key:
            common_lower = {c.lower() for c in common_cols}
            if any(k.lower() not in common_lower for k in configured_key):
                return True, common_cols

        # Per-column NON-BLOCKING checks: type family, nullable, order.
        src_by_col = {
            str(r["column_name"]).lower(): r for _, r in source_schema_df.iterrows()
        }
        tgt_by_col = {
            str(r["column_name"]).lower(): r for _, r in target_schema_df.iterrows()
        }

        if column_enabled:
            source_order = [
                c for c in source_schema_df["column_name"].tolist()
                if c.lower() in {x.lower() for x in common_cols}
            ]
            target_order = [
                c for c in target_schema_df["column_name"].tolist()
                if c.lower() in {x.lower() for x in common_cols}
            ]
            result.source_column_order = source_order
            result.target_column_order = target_order

            if request.validate_column_order:
                result.column_order_status = self.compare_column_order(source_order, target_order)
            else:
                result.column_order_status = ValidationStatus.SKIPPED

            dtype_statuses, nullable_statuses = [], []
            column_results: List[ColumnValidationResult] = []

            for col in common_cols:
                key = col.lower()
                src_row = src_by_col.get(key, {})
                tgt_row = tgt_by_col.get(key, {})

                col_result = ColumnValidationResult(column=col, status=ValidationStatus.PASS)

                src_type = str(src_row.get("data_type"))
                tgt_type = str(tgt_row.get("data_type"))
                col_result.source_data_type = src_type
                col_result.target_data_type = tgt_type
                col_result.data_type_status = self._classify_type_family(src_type, tgt_type)
                dtype_statuses.append(col_result.data_type_status)

                if request.validate_nullable:
                    src_null = bool(src_row.get("is_nullable"))
                    tgt_null = bool(tgt_row.get("is_nullable"))
                    col_result.source_nullable = src_null
                    col_result.target_nullable = tgt_null
                    col_result.nullable_status = self.compare_nullable(src_null, tgt_null)
                else:
                    col_result.nullable_status = ValidationStatus.SKIPPED
                nullable_statuses.append(col_result.nullable_status)

                col_result.status = self.calculate_overall_status(
                    [col_result.data_type_status, col_result.nullable_status]
                )
                column_results.append(col_result)

            result.columns = column_results
            result.data_types_status = self.calculate_overall_status(dtype_statuses)
            result.nullable_status = self.calculate_overall_status(nullable_statuses)
        else:
            result.column_order_status = ValidationStatus.SKIPPED
            result.data_types_status = ValidationStatus.SKIPPED
            result.nullable_status = ValidationStatus.SKIPPED

        # Cross-family type change is BLOCKING even when COLUMN reporting
        # is disabled (detection always runs; only reporting is gated).
        for col in common_cols:
            key = col.lower()
            src_type = str(src_by_col.get(key, {}).get("data_type"))
            tgt_type = str(tgt_by_col.get(key, {}).get("data_type"))
            if self._classify_type_family(src_type, tgt_type) == ValidationStatus.FAIL:
                return True, common_cols

        return False, common_cols

    # ------------------------------------------------------------------
    # Tier 1: statistical profile. One aggregate query per side (row
    # count already available via get_row_count; null/distinct/min-max
    # via get_column_statistics). Any mismatch -> stop before Tier 2.
    # ------------------------------------------------------------------
    def _tier1_statistics(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        source_schema_df: pd.DataFrame,
        result: TableValidationResult,
    ) -> bool:
        """Returns True if a statistical mismatch was found (stop the funnel here)."""
        column_enabled = ValidationType.COLUMN in request.enabled_validations
        mismatch = False

        # Row count (also serves stage "row_count_status" as before).
        try:
            src_count = self.databricks.get_row_count(
                request.source_catalog, source_schema, source_table
            )
            tgt_count = self.databricks.get_row_count(
                request.target_catalog, target_schema, target_table
            )
            result.row_count_source = src_count
            result.row_count_target = tgt_count
            result.row_count_difference = tgt_count - src_count
            result.row_count_status = self.compare_row_counts(src_count, tgt_count)
            if result.row_count_status == ValidationStatus.FAIL:
                mismatch = True
        except Exception as exc:
            logger.exception(
                "Failed to compute row counts for '%s.%s'", target_schema, target_table
            )
            result.row_count_status = ValidationStatus.ERROR
            result.error = f"Row count failed: {exc}"
            mismatch = True

        src_by_col = {
            str(r["column_name"]).lower(): r for _, r in source_schema_df.iterrows()
        }
        min_max_columns = [
            c for c in common_cols
            if self.databricks.is_min_max_eligible(
                str(src_by_col.get(c.lower(), {}).get("data_type", ""))
            )
        ]

        try:
            source_stats = self.databricks.get_column_statistics(
                request.source_catalog, source_schema, source_table,
                common_cols, min_max_columns,
            )
            target_stats = self.databricks.get_column_statistics(
                request.target_catalog, target_schema, target_table,
                common_cols, min_max_columns,
            )
            stats_error = None
        except Exception as exc:
            logger.exception(
                "Failed to compute column statistics for '%s.%s'", target_schema, target_table
            )
            source_stats, target_stats = {}, {}
            stats_error = str(exc)
            mismatch = True

        existing_by_col = {c.column: c for c in result.columns}
        null_statuses, distinct_statuses, minmax_statuses = [], [], []

        for col in common_cols:
            col_result = existing_by_col.get(col)
            if col_result is None:
                col_result = ColumnValidationResult(column=col, status=ValidationStatus.PASS)
                existing_by_col[col] = col_result

            if stats_error:
                col_result.null_count_status = ValidationStatus.ERROR
                col_result.distinct_count_status = ValidationStatus.ERROR
                col_result.error = stats_error
            else:
                s_stat = source_stats.get(col, {})
                t_stat = target_stats.get(col, {})

                col_result.source_null_count = s_stat.get("null_count")
                col_result.target_null_count = t_stat.get("null_count")
                col_result.null_count_status = self.compare_null_counts(
                    col_result.source_null_count, col_result.target_null_count
                )
                if col_result.null_count_status == ValidationStatus.FAIL:
                    mismatch = True

                col_result.source_distinct_count = s_stat.get("distinct_count")
                col_result.target_distinct_count = t_stat.get("distinct_count")
                col_result.distinct_count_status = self.compare_distinct_counts(
                    col_result.source_distinct_count, col_result.target_distinct_count
                )
                if col_result.distinct_count_status == ValidationStatus.FAIL:
                    mismatch = True

                if col in min_max_columns:
                    col_result.source_min = s_stat.get("min")
                    col_result.source_max = s_stat.get("max")
                    col_result.target_min = t_stat.get("min")
                    col_result.target_max = t_stat.get("max")
                    col_result.min_max_status = self.compare_min_max(
                        col_result.source_min, col_result.source_max,
                        col_result.target_min, col_result.target_max,
                    )
                    if col_result.min_max_status == ValidationStatus.FAIL:
                        mismatch = True
                else:
                    col_result.min_max_status = ValidationStatus.SKIPPED

            null_statuses.append(col_result.null_count_status)
            distinct_statuses.append(col_result.distinct_count_status)
            minmax_statuses.append(col_result.min_max_status)

            col_result.status = self.calculate_overall_status(
                [
                    col_result.data_type_status,
                    col_result.nullable_status,
                    col_result.null_count_status,
                    col_result.distinct_count_status,
                    col_result.min_max_status,
                ]
            )

        # null/distinct/min-max are Tier 1's own findings - part of ROW's
        # fail-fast pipeline (this method only runs when ROW is enabled),
        # not gated by COLUMN. result.columns must always reflect what
        # Tier 1 actually computed regardless of COLUMN, since the
        # Suggestions sheet explains a ROW-only FAIL by walking this list -
        # gating it on COLUMN previously left a Tier-1-only failure with no
        # explanation at all ("Unclassified") when COLUMN wasn't selected.
        result.null_counts_status = self.calculate_overall_status(null_statuses)
        result.distinct_counts_status = self.calculate_overall_status(distinct_statuses)
        result.min_max_status = self.calculate_overall_status(minmax_statuses)
        result.columns = [existing_by_col[c] for c in common_cols]

        return mismatch

    # ------------------------------------------------------------------
    # Tier 2: whole-table fingerprint. Single aggregate query per side
    # (COUNT * SUM(hash) * XOR(hash)). Match -> tables equal, stop (never
    # reach Tier 4/5). This is the tier that directly prevents the
    # ROW_NUMBER()-fallback timeout: most "actually equal" or "clearly
    # different at the schema/stats level" tables never reach it or stop
    # right here.
    # ------------------------------------------------------------------
    def _tier2_fingerprint(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        result: TableValidationResult,
    ) -> bool:
        """Returns True if the fingerprints match (stop the funnel here)."""
        value_columns = sorted(common_cols)

        try:
            source_fp = self.databricks.get_table_fingerprint(
                request.source_catalog, source_schema, source_table, value_columns,
            )
            target_fp = self.databricks.get_table_fingerprint(
                request.target_catalog, target_schema, target_table, value_columns,
            )
        except Exception as exc:
            logger.exception(
                "Failed to compute table fingerprint for '%s.%s'", target_schema, target_table
            )
            result.data = DataValidationResult(
                mode=request.data_compare_mode,
                status=ValidationStatus.ERROR,
                fingerprint_status=ValidationStatus.ERROR,
                error=f"Fingerprint comparison failed: {exc}",
            )
            return False

        matches = (
            source_fp.get("row_count") == target_fp.get("row_count")
            and source_fp.get("hash_sum") == target_fp.get("hash_sum")
            and source_fp.get("hash_xor") == target_fp.get("hash_xor")
        )

        logger.info(
            "[tier2-fingerprint] table=%s.%s | match=%s | source=%s | target=%s",
            target_schema, target_table, matches, source_fp, target_fp,
        )

        result.data = DataValidationResult(
            mode=request.data_compare_mode,
            status=ValidationStatus.PASS if matches else ValidationStatus.SKIPPED,
            fingerprint_status=ValidationStatus.PASS if matches else ValidationStatus.FAIL,
            source_fingerprint=str(source_fp),
            target_fingerprint=str(target_fp),
            note=(
                "Whole-table fingerprint matched - tables are equal, "
                "row-level comparison skipped." if matches else None
            ),
        )

        return matches

    # ------------------------------------------------------------------
    # Tier 3 candidate selection (pure, no I/O): which common columns are
    # reasonable to offer as a partition/bucket column. Kept deliberately
    # simple - no cardinality ranking - just excludes the configured
    # primary key (already uniquely identifies rows, useless as a bucket
    # dimension) and sorts alphabetically.
    # ------------------------------------------------------------------
    @staticmethod
    def _partition_candidates(
        common_cols: List[str],
        key_columns: Optional[List[str]],
    ) -> List[str]:
        key_lower = {k.lower() for k in (key_columns or [])}
        return sorted(c for c in common_cols if c.lower() not in key_lower)

    # ------------------------------------------------------------------
    # Tier 4 dispatch: decides between partitioned and unpartitioned Tier
    # 4 for a table with a confirmed mismatch (Tier 1 and/or Tier 2).
    # Partitioning is only offered when a partition_prompt callback is
    # configured AND the table is large enough (row_count over
    # request.partition_threshold) - small tables and callers that never
    # opted into the callback always get today's unpartitioned behavior,
    # unchanged.
    # ------------------------------------------------------------------
    def _dispatch_tier4(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        result: TableValidationResult,
        stats_mismatch: bool = False,
    ) -> None:
        row_count = result.row_count_source or 0
        large_enough = row_count > request.partition_threshold

        if not large_enough:
            result.partition_skip_reason = None  # too small to even offer
            self._tier4_and_5_row_level(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
            )
            return

        if self.partition_prompt is None:
            result.partition_skip_reason = "no partition_prompt configured"
            self._tier4_and_5_row_level(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
            )
            return

        # Keyed by TARGET-side name, same convention as _lookup_primary_key
        # everywhere else (this is what the user types into config as
        # "target_table.table"). common_cols is the already-agreed common
        # column list, so it's identical regardless of which side's names
        # are used for the partition candidate list/UI prompt.
        key_columns = self._lookup_primary_key(request, target_schema, target_table)
        candidates = self._partition_candidates(common_cols, key_columns)

        context = PartitionPromptContext(
            schema_name=target_schema,
            table=target_table,
            row_count=row_count,
            candidate_columns=candidates,
        )

        try:
            chosen_column = self.partition_prompt(context)
        except Exception:
            logger.exception(
                "partition_prompt callback failed for '%s.%s' - falling back "
                "to unpartitioned Tier 4",
                target_schema, target_table,
            )
            chosen_column = None

        if not chosen_column:
            result.partition_skip_reason = "user declined or non-interactive run"
            self._tier4_and_5_row_level(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
            )
            return

        self._tier3_partition_and_tier4(
            request, source_schema, target_schema, source_table, target_table,
            common_cols, result,
            chosen_column, stats_mismatch=stats_mismatch,
        )

    # ------------------------------------------------------------------
    # Tier 3: partition/bucket fingerprinting, then Tier 4 (+5) scoped to
    # only the culprit buckets whose fingerprints disagree. Matching
    # buckets are never touched by Tier 4 - this is what makes a large,
    # partly-different table cheaper to diff than a full unpartitioned
    # row-hash scan.
    # ------------------------------------------------------------------
    def _tier3_partition_and_tier4(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        result: TableValidationResult,
        bucket_column: str,
        stats_mismatch: bool = False,
    ) -> None:
        value_columns = sorted(common_cols)

        try:
            source_buckets = self.databricks.get_table_fingerprint_by_bucket(
                request.source_catalog, source_schema, source_table, value_columns, bucket_column,
            )
            target_buckets = self.databricks.get_table_fingerprint_by_bucket(
                request.target_catalog, target_schema, target_table, value_columns, bucket_column,
            )
        except Exception as exc:
            logger.exception(
                "Tier 3 bucket fingerprint failed for '%s.%s' (bucket_column='%s') - "
                "falling back to unpartitioned Tier 4",
                target_schema, target_table, bucket_column,
            )
            result.partition_skip_reason = f"bucket fingerprint failed: {exc}"
            self._tier4_and_5_row_level(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
            )
            return

        source_by_bucket = {row["bucket_value"]: row for _, row in source_buckets.iterrows()}
        target_by_bucket = {row["bucket_value"]: row for _, row in target_buckets.iterrows()}
        all_buckets = sorted(
            set(source_by_bucket) | set(target_by_bucket), key=str,
        )

        culprit_buckets = []
        for bucket_value in all_buckets:
            src = source_by_bucket.get(bucket_value)
            tgt = target_by_bucket.get(bucket_value)
            matches = (
                src is not None and tgt is not None
                and src.get("row_count") == tgt.get("row_count")
                and src.get("hash_sum") == tgt.get("hash_sum")
                and src.get("hash_xor") == tgt.get("hash_xor")
            )
            if not matches:
                culprit_buckets.append(bucket_value)

        logger.info(
            "[tier3-partition] table=%s.%s | bucket_column=%s | total_buckets=%d | "
            "culprit_buckets=%d",
            target_schema, target_table, bucket_column, len(all_buckets), len(culprit_buckets),
        )

        result.partitioned = True
        result.partition_column = bucket_column
        result.partition_buckets_total = len(all_buckets)
        result.partition_buckets_culprit = len(culprit_buckets)

        accumulate: Dict[str, Any] = {
            "mismatches": [],
            "mismatch_count": 0,
            "using_row_number_fallback": False,
        }
        for bucket_value in culprit_buckets:
            self._tier4_and_5_row_level(
                request, source_schema, target_schema, source_table, target_table,
                common_cols, result,
                stats_mismatch=stats_mismatch,
                bucket_predicate=(bucket_column, bucket_value),
                _accumulate=accumulate,
            )

        mismatches: List[RowHashMismatch] = accumulate["mismatches"]
        mismatch_count: int = accumulate["mismatch_count"]
        using_row_number_fallback: bool = accumulate["using_row_number_fallback"]
        effective_key_columns = accumulate.get("key_columns", ["row_number"])
        total_rows = result.row_count_source or 0
        mismatch_pct = (mismatch_count / total_rows) * 100 if total_rows else 0.0

        if result.data is None:
            result.data = DataValidationResult(
                mode=request.data_compare_mode,
                status=ValidationStatus.SKIPPED,
            )
        result.data.row_hash_mismatches = mismatches
        result.data.row_hash_mismatch_count = mismatch_count
        result.data.row_hash_mismatch_percentage = mismatch_pct
        result.data.key_columns = effective_key_columns
        result.data.note = (
            (result.data.note + " " if result.data.note else "")
            + f"Partitioned by '{bucket_column}': {len(culprit_buckets)} of "
            f"{len(all_buckets)} bucket(s) differed and were row-hash "
            f"compared; matching buckets were not scanned."
        )

        if mismatch_count > 0:
            result.data.status = ValidationStatus.FAIL
            result.tier_reached = ValidationTier.ROW_HASH

            if not using_row_number_fallback:
                key_columns = self._lookup_primary_key(request, target_schema, target_table)
                mismatched_keys = [m.primary_key for m in mismatches if m.status == "MISMATCH"]
                if key_columns and mismatched_keys:
                    self._tier5_column_diff(
                        request, source_schema, target_schema, source_table, target_table,
                        common_cols,
                        key_columns, mismatched_keys, result,
                    )
                    result.tier_reached = ValidationTier.COLUMN_DIFF
        elif stats_mismatch:
            # Tier 1 already confirmed a real difference, but no culprit
            # bucket's row-hash join found a mismatched key - same
            # reasoning as the unpartitioned path: don't silently
            # override a confirmed finding.
            result.data.status = ValidationStatus.FAIL
            result.data.note += (
                " Row-hash comparison of all differing buckets found no "
                "mismatched keys, but Tier 1 already confirmed a "
                "statistical difference."
            )
            result.tier_reached = ValidationTier.ROW_HASH
        else:
            result.data.status = ValidationStatus.PASS
            result.tier_reached = ValidationTier.ROW_HASH

    # ------------------------------------------------------------------
    # Tier 4/5 orchestration: row-hash diff, either over the whole table
    # (bucket_predicate=None) or scoped to a single partition bucket (see
    # _tier3_partition_and_tier4), then column-level diff for whatever
    # keys the row-hash diff flagged as mismatched.
    # ------------------------------------------------------------------
    def _tier4_and_5_row_level(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        result: TableValidationResult,
        stats_mismatch: bool = False,
        bucket_predicate: Optional[Tuple[str, Any]] = None,
        _accumulate: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Runs Tier 4 (+ Tier 5 for real mismatched keys) either over the
        whole table (bucket_predicate=None, the default/unpartitioned
        path, mutates `result` directly) or scoped to exactly one
        partition bucket. When `_accumulate` is given (only used by
        _tier3_partition_and_tier4, one call per culprit bucket), this
        call's mismatches/counts are appended into it instead of being
        written straight to `result.data`, so multiple bucket calls
        aggregate instead of each overwriting the last.
        """
        row_hash_key_columns = self._lookup_primary_key(request, target_schema, target_table)

        using_row_number_fallback = not row_hash_key_columns
        if using_row_number_fallback:
            logger.info(
                "[row-hash] no key configured for '%s.%s' - falling back to "
                "ROW_NUMBER()-based comparison (ORDER BY every common column). "
                "Best-effort only: reliable solely when both sides have the same "
                "row set.",
                target_schema, target_table,
            )

        try:
            if using_row_number_fallback:
                mismatches, mismatch_count, mismatch_pct = self._run_row_hash_stage_by_row_number(
                    request, source_schema, target_schema, source_table, target_table, common_cols,
                    bucket_predicate=bucket_predicate,
                )
                effective_key_columns = ["row_number"]
            else:
                mismatches, mismatch_count, mismatch_pct = self._run_row_hash_stage(
                    request, source_schema, target_schema, source_table, target_table,
                    common_cols, row_hash_key_columns,
                    bucket_predicate=bucket_predicate,
                )
                effective_key_columns = row_hash_key_columns

            if bucket_predicate is not None:
                bucket_label = str(bucket_predicate[1])
                for m in mismatches:
                    m.partition_bucket = bucket_label

            logger.info(
                "[row-hash] table=%s.%s | key_columns=%s | mismatch_count=%s | mismatch_pct=%.2f%%",
                target_schema, target_table, effective_key_columns, mismatch_count, mismatch_pct,
            )

            if _accumulate is not None:
                # Called once per culprit bucket by
                # _tier3_partition_and_tier4 - append rather than
                # overwrite, and let the caller decide the final status/
                # tier_reached/Tier 5 dispatch once all buckets are in.
                _accumulate["mismatches"].extend(mismatches)
                _accumulate["mismatch_count"] += mismatch_count
                _accumulate.setdefault("key_columns", effective_key_columns)
                _accumulate["using_row_number_fallback"] = using_row_number_fallback
                return

            if result.data is None:
                result.data = DataValidationResult(
                    mode=request.data_compare_mode,
                    status=ValidationStatus.SKIPPED,
                )
            result.data.row_hash_mismatches = mismatches
            result.data.row_hash_mismatch_count = mismatch_count
            result.data.row_hash_mismatch_percentage = mismatch_pct
            result.data.key_columns = effective_key_columns
            if using_row_number_fallback:
                result.data.note = (
                    "No primary key configured - row-level comparison used a "
                    "synthetic ROW_NUMBER() (ORDER BY every common column) "
                    "instead of a real key. Only reliable when both sides "
                    "contain the same set of rows in the same relative order; "
                    "the 'Data Mismatches' sheet may show best-effort "
                    "column-level detail for these rows (marked unverified), "
                    "but it cannot guarantee the same confidence as a real "
                    "primary key."
                )
            if mismatch_count > 0 and result.data.status != ValidationStatus.ERROR:
                result.data.status = ValidationStatus.FAIL
                result.tier_reached = ValidationTier.ROW_HASH

                # Tier 5: column-level diff. For a real configured key
                # this pinpoints the exact record; for the row-number
                # fallback it's best-effort (see
                # get_row_detail_for_row_numbers) and every resulting
                # detail row is tagged verified=False.
                mismatched_keys = [
                    m.primary_key for m in mismatches if m.status == "MISMATCH"
                ]
                if mismatched_keys:
                    self._tier5_column_diff(
                        request, source_schema, target_schema, source_table, target_table,
                        common_cols,
                        effective_key_columns, mismatched_keys, result,
                        using_row_number_fallback=using_row_number_fallback,
                        bucket_predicate=bucket_predicate,
                    )
                    result.tier_reached = ValidationTier.COLUMN_DIFF
            elif result.data.status == ValidationStatus.SKIPPED and mismatch_count == 0:
                if stats_mismatch:
                    # Tier 1 already confirmed a real difference, but the
                    # row-hash join found no mismatched key - the
                    # difference is likely in a value outside the hashed
                    # columns (e.g. a min/max-only finding) or invisible
                    # to a row-set comparison. Keep this a FAIL rather
                    # than silently reporting PASS on a table Tier 1
                    # already proved differs.
                    result.data.status = ValidationStatus.FAIL
                    result.data.note = (
                        (result.data.note + " " if result.data.note else "")
                        + "Row-hash comparison found no mismatched keys, but "
                        "Tier 1 already confirmed a statistical difference - "
                        "see the null/distinct/min-max columns on this table "
                        "for the specific statistic that disagrees."
                    )
                else:
                    result.data.status = ValidationStatus.PASS
                result.tier_reached = ValidationTier.ROW_HASH
        except Exception as exc:
            logger.exception(
                "Failed to run row-hash comparison for '%s.%s'", target_schema, target_table
            )
            if result.data is None:
                result.data = DataValidationResult(
                    mode=request.data_compare_mode,
                    status=ValidationStatus.ERROR,
                    key_columns=row_hash_key_columns or ["row_number"],
                    error=f"Row-hash comparison failed: {exc}",
                )
            else:
                result.data.status = ValidationStatus.ERROR
                result.data.error = f"Row-hash comparison failed: {exc}"

    # ------------------------------------------------------------------
    # Tier 5: column-level diff. Only for keys Tier 4 flagged as
    # ROW_HASH_MISMATCH - a thin re-wire of the existing bounded-sample
    # fetch (_changed_row_detail), so it's never a full-table pull.
    # ------------------------------------------------------------------
    def _tier5_column_diff(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_cols: List[str],
        key_columns: List[str],
        mismatched_keys: List[str],
        result: TableValidationResult,
        using_row_number_fallback: bool = False,
        bucket_predicate: Optional[Tuple[str, Any]] = None,
    ) -> None:
        # NOTE: DatabricksConnector.get_row_detail_for_keys/
        # get_row_detail_for_row_numbers each take a single schema/table
        # pair used to qualify BOTH source_catalog and target_catalog
        # (they assume the schema/table name is shared) - an asymmetric
        # schema_map/table_map pair therefore re-queries both sides under
        # the TARGET-side name, matching the reporting convention used
        # everywhere else in this class. Widening those connector methods
        # to accept independent source/target schema+table names is out
        # of scope for this change; today this only matters when the
        # source-side name doesn't actually exist under the target's own
        # catalog, which would already have been caught as a BLOCKING
        # Tier 0 schema difference or a missing-table pair error before
        # Tier 5 could ever run.
        if using_row_number_fallback:
            # No real key to exclude - every common column is a value
            # column. This MUST match the column list/order used to
            # compute the original ROW_NUMBER() in
            # _run_row_hash_stage_by_row_number, or the re-fetched row
            # numbers here won't line up with the mismatch's stored
            # "row_number" values.
            value_columns = sorted(common_cols)
        else:
            value_columns = sorted(
                c for c in common_cols if c.lower() not in {k.lower() for k in key_columns}
            )
        if not value_columns:
            return

        # mismatched_keys are the "|"-joined display keys from
        # compare_row_hashes; for a single-column key this is directly
        # usable as a literal value list. Multi-column keys aren't safely
        # reconstructible from the "|"-joined display string - skip Tier 5
        # rather than risk fetching the wrong rows. (Deferred: carry
        # structured key tuples through Tier 4 instead of display strings.)
        # Row-number fallback is always a single synthetic key, so this
        # guard never applies to it.
        if not using_row_number_fallback and len(key_columns) != 1:
            return

        try:
            if using_row_number_fallback:
                detail = self.databricks.get_row_detail_for_row_numbers(
                    source_catalog=request.source_catalog,
                    target_catalog=request.target_catalog,
                    schema=target_schema,
                    table=target_table,
                    order_by_columns=value_columns,
                    row_numbers=[int(k) for k in mismatched_keys],
                    value_columns=value_columns,
                    limit_samples=request.max_sample_rows,
                    bucket_predicate=bucket_predicate,
                )
            else:
                detail = self.databricks.get_row_detail_for_keys(
                    source_catalog=request.source_catalog,
                    target_catalog=request.target_catalog,
                    schema=target_schema,
                    table=target_table,
                    key_column=key_columns[0],
                    key_values=mismatched_keys,
                    value_columns=value_columns,
                    limit_samples=request.max_sample_rows,
                )
        except Exception as exc:
            logger.exception(
                "Tier 5 column diff failed for '%s.%s'", target_schema, target_table
            )
            if result.data is not None:
                result.data.error = f"Column-level diff failed: {exc}"
            return

        sample_changed_detail: List[RowMismatchDetail] = []
        for row in detail:
            for col in row["mismatched_columns"]:
                sample_changed_detail.append(
                    RowMismatchDetail(
                        schema_name=target_schema,
                        table=target_table,
                        primary_key=row["key"],
                        mismatch_column=col,
                        source_value=row["source_values"].get(col),
                        target_value=row["target_values"].get(col),
                        source_row_hash=row["source_row_hash"],
                        target_row_hash=row["target_row_hash"],
                        verified=not using_row_number_fallback,
                    )
                )

        if result.data is not None:
            result.data.sample_changed_detail = sample_changed_detail

    # ------------------------------------------------------------------
    # Stage 15: actual data comparison
    # ------------------------------------------------------------------
    def compare_data(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_columns: List[str],
    ) -> DataValidationResult:

        mode = request.data_compare_mode
        key_columns = self._lookup_primary_key(request, target_schema, target_table)

        logger.info(
            "[compare_data] table=%s.%s | mode=%s | resolved_key_columns=%s",
            target_schema, target_table, mode.value, key_columns,
        )

        if mode == DataCompareMode.COUNT_ONLY:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note="COUNT_ONLY mode: row-level data comparison skipped by configuration.",
            )

        if mode == DataCompareMode.STATISTICS:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note=(
                    "STATISTICS mode (default): row count / null / distinct / "
                    "min-max already validated above; row-level comparison skipped "
                    "for cost. Use HASH or FULL to enable it."
                ),
            )

        if not key_columns:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.SKIPPED,
                note=(
                    f"No primary/business key configured for "
                    f"'{target_schema}.{target_table}' - row-level data comparison "
                    "requires a key and was skipped. Configure request.primary_keys "
                    "to enable it."
                ),
            )

        missing_keys = [k for k in key_columns if k.lower() not in {c.lower() for c in common_columns}]
        if missing_keys:
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.ERROR,
                key_columns=key_columns,
                error=f"Configured key column(s) not found as common columns: {missing_keys}",
            )

        value_columns = [
            c for c in common_columns if c.lower() not in {k.lower() for k in key_columns}
        ]

        source_fqtn = f"{request.source_catalog}.{source_schema}.{source_table}"
        target_fqtn = f"{request.target_catalog}.{target_schema}.{target_table}"

        try:
            diff = self.databricks.key_based_row_diff(
                source_fqtn=source_fqtn,
                target_fqtn=target_fqtn,
                key_columns=key_columns,
                # HASH mode: only need counts, not full samples of value diffs;
                # FULL mode: return samples too. Either way this is pushed down.
                value_columns=value_columns,
                limit_samples=request.max_sample_rows if mode == DataCompareMode.FULL else 5,
            )
        except Exception as exc:
            logger.exception(
                "Failed to run key-based data comparison for '%s.%s'",
                target_schema, target_table,
            )
            return DataValidationResult(
                mode=mode,
                status=ValidationStatus.ERROR,
                key_columns=key_columns,
                error=f"Data comparison failed: {exc}",
            )

        has_diff = (
            diff["source_only_rows"] > 0
            or diff["target_only_rows"] > 0
            or diff["changed_rows"] > 0
        )

        logger.info(
            "[data-mismatch] table=%s.%s | mode=%s | source_only=%d | target_only=%d | "
            "changed_rows=%d | sample_changed_detail_rows=%d",
            target_schema, target_table, mode.value,
            diff["source_only_rows"], diff["target_only_rows"], diff["changed_rows"],
            len(diff.get("sample_changed_detail", [])),
        )

        sample_changed_detail: List[RowMismatchDetail] = []
        if mode == DataCompareMode.FULL:
            for row in diff.get("sample_changed_detail", []):
                for col in row["mismatched_columns"]:
                    sample_changed_detail.append(
                        RowMismatchDetail(
                            schema_name=target_schema,
                            table=target_table,
                            primary_key=row["key"],
                            mismatch_column=col,
                            source_value=row["source_values"].get(col),
                            target_value=row["target_values"].get(col),
                            source_row_hash=row["source_row_hash"],
                            target_row_hash=row["target_row_hash"],
                        )
                    )

        return DataValidationResult(
            mode=mode,
            status=ValidationStatus.FAIL if has_diff else ValidationStatus.PASS,
            key_columns=key_columns,
            source_only_rows=diff["source_only_rows"],
            target_only_rows=diff["target_only_rows"],
            changed_rows=diff["changed_rows"],
            sample_source_only=(
                diff["sample_source_only"] if mode == DataCompareMode.FULL else []
            ),
            sample_target_only=(
                diff["sample_target_only"] if mode == DataCompareMode.FULL else []
            ),
            sample_changed=(
                diff["sample_changed"] if mode == DataCompareMode.FULL else []
            ),
            sample_changed_detail=sample_changed_detail,
        )

    # ------------------------------------------------------------------
    # Stage 15b: row-hash comparison
    #
    # Separate mechanism from key_based_row_diff/_changed_row_detail above
    # (which only run in HASH/FULL mode): this is a single pushed-down
    # whole-row hash per side, joined by primary key in Python (never by
    # row position/order), and is the primary way to detect row-level
    # mismatches whenever a primary key is configured - independent of
    # data_compare_mode.
    # ------------------------------------------------------------------
    def _run_row_hash_stage(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_columns: List[str],
        key_columns: List[str],
        bucket_predicate: Optional[Tuple[str, Any]] = None,
    ) -> Tuple[List[RowHashMismatch], int, float]:

        value_columns = sorted(
            c for c in common_columns if c.lower() not in {k.lower() for k in key_columns}
        )

        logger.info(
            "[row-hash] fetching hashes | table=%s.%s | key_columns=%s | value_columns=%s"
            "%s",
            target_schema, target_table, key_columns, value_columns,
            f" | bucket={bucket_predicate}" if bucket_predicate else "",
        )

        source_hashes = self.databricks.get_row_hashes(
            request.source_catalog, source_schema, source_table, value_columns, key_columns,
            bucket_predicate=bucket_predicate,
        )
        target_hashes = self.databricks.get_row_hashes(
            request.target_catalog, target_schema, target_table, value_columns, key_columns,
            bucket_predicate=bucket_predicate,
        )

        logger.info(
            "[row-hash] fetched | table=%s.%s | source_rows=%d | target_rows=%d",
            target_schema, target_table, len(source_hashes), len(target_hashes),
        )

        return self.compare_row_hashes(source_hashes, target_hashes, key_columns)

    def _run_row_hash_stage_by_row_number(
        self,
        request: CatalogValidationRequest,
        source_schema: str,
        target_schema: str,
        source_table: str,
        target_table: str,
        common_columns: List[str],
        bucket_predicate: Optional[Tuple[str, Any]] = None,
    ) -> Tuple[List[RowHashMismatch], int, float]:
        """
        Fallback used when no primary key is configured for the table:
        both sides get a synthetic ROW_NUMBER() (ORDER BY every common
        column) instead of a real key. See
        DatabricksConnector.get_row_hashes_by_row_number for the caveat
        about what this can and cannot detect.
        """
        value_columns = sorted(common_columns)

        logger.info(
            "[row-hash] fetching row-number hashes | table=%s.%s | value_columns=%s%s",
            target_schema, target_table, value_columns,
            f" | bucket={bucket_predicate}" if bucket_predicate else "",
        )

        source_hashes = self.databricks.get_row_hashes_by_row_number(
            request.source_catalog, source_schema, source_table, value_columns,
            bucket_predicate=bucket_predicate,
        )
        target_hashes = self.databricks.get_row_hashes_by_row_number(
            request.target_catalog, target_schema, target_table, value_columns,
            bucket_predicate=bucket_predicate,
        )

        logger.info(
            "[row-hash] fetched row-number hashes | table=%s.%s | source_rows=%d | target_rows=%d",
            target_schema, target_table, len(source_hashes), len(target_hashes),
        )

        return self.compare_row_hashes(source_hashes, target_hashes, ["row_number"])

    @staticmethod
    def compare_row_hashes(
        source_hashes: pd.DataFrame,
        target_hashes: pd.DataFrame,
        primary_key_cols: Sequence[str],
    ) -> Tuple[List[RowHashMismatch], int, float]:
        """
        Join two per-key row-hash sets by primary key (never row position,
        via a sort-then-merge join over both key sets) and classify every
        key as matching, MISMATCH (key on both sides, hash differs),
        MISSING_IN_TARGET, or MISSING_IN_SOURCE. A key appearing more than
        once on either side (the configured "key" isn't actually unique)
        is classified separately as DUPLICATE_KEY rather than silently
        collapsed to its last occurrence.

        Returns (mismatches, mismatch_count, mismatch_percentage) where
        mismatch_percentage is mismatch_count / total_compared_keys * 100
        and total_compared_keys is the union of keys seen on either side.
        """

        def _display_key(row: pd.Series) -> str:
            return "|".join(str(row[k]) for k in primary_key_cols)

        def _key_tuple(row: pd.Series) -> tuple:
            return tuple(row[k] for k in primary_key_cols)

        def _group_by_key(df: pd.DataFrame) -> Dict[tuple, List[pd.Series]]:
            grouped: Dict[tuple, List[pd.Series]] = {}
            for _, row in df.iterrows():
                grouped.setdefault(_key_tuple(row), []).append(row)
            return grouped

        source_by_key = _group_by_key(source_hashes)
        target_by_key = _group_by_key(target_hashes)

        # Sort-then-merge join: both key sets are sorted once, then walked
        # with two cursors, so memory stays proportional to the number of
        # distinct keys rather than requiring a hash-join structure sized
        # to the larger side.
        all_keys = sorted(set(source_by_key) | set(target_by_key))

        mismatches: List[RowHashMismatch] = []

        for key_tuple in all_keys:
            src_rows = source_by_key.get(key_tuple, [])
            tgt_rows = target_by_key.get(key_tuple, [])

            if len(src_rows) > 1 or len(tgt_rows) > 1:
                display = _display_key(src_rows[0] if src_rows else tgt_rows[0])
                mismatches.append(
                    RowHashMismatch(
                        primary_key=display,
                        source_hash=str(src_rows[0]["row_hash"]) if src_rows else "",
                        target_hash=str(tgt_rows[0]["row_hash"]) if tgt_rows else "",
                        status="DUPLICATE_KEY",
                    )
                )
                continue

            src_row = src_rows[0] if src_rows else None
            tgt_row = tgt_rows[0] if tgt_rows else None

            if src_row is not None and tgt_row is None:
                mismatches.append(
                    RowHashMismatch(
                        primary_key=_display_key(src_row),
                        source_hash=str(src_row["row_hash"]),
                        target_hash="",
                        status="MISSING_IN_TARGET",
                    )
                )
            elif src_row is None and tgt_row is not None:
                mismatches.append(
                    RowHashMismatch(
                        primary_key=_display_key(tgt_row),
                        source_hash="",
                        target_hash=str(tgt_row["row_hash"]),
                        status="MISSING_IN_SOURCE",
                    )
                )
            elif src_row is not None and tgt_row is not None:
                if src_row["row_hash"] != tgt_row["row_hash"]:
                    mismatches.append(
                        RowHashMismatch(
                            primary_key=_display_key(src_row),
                            source_hash=str(src_row["row_hash"]),
                            target_hash=str(tgt_row["row_hash"]),
                            status="MISMATCH",
                        )
                    )

        total_compared_keys = len(all_keys)
        mismatch_count = len(mismatches)
        mismatch_percentage = (
            (mismatch_count / total_compared_keys) * 100 if total_compared_keys else 0.0
        )

        return mismatches, mismatch_count, mismatch_percentage

    # ------------------------------------------------------------------
    # Stage 16/17: overall status aggregation (programmatic, never hardcoded)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_overall_status(
        statuses: List[Optional[ValidationStatus]],
    ) -> ValidationStatus:

        clean = [s for s in statuses if s is not None]

        if not clean:
            return ValidationStatus.SKIPPED

        if any(s == ValidationStatus.ERROR for s in clean):
            return ValidationStatus.ERROR

        if any(s == ValidationStatus.FAIL for s in clean):
            return ValidationStatus.FAIL

        if all(s == ValidationStatus.SKIPPED for s in clean):
            return ValidationStatus.SKIPPED

        return ValidationStatus.PASS

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        schema_results: List[SchemaValidationResult],
        missing_schemas: List[str],
        extra_schemas: List[str],
    ) -> ValidationSummary:

        summary = ValidationSummary()

        summary.total_schemas = len(schema_results) + len(missing_schemas)
        summary.failed_schemas = sum(
            1 for s in schema_results if s.status in (ValidationStatus.FAIL, ValidationStatus.ERROR)
        ) + len(missing_schemas)
        summary.passed_schemas = summary.total_schemas - summary.failed_schemas

        for schema_result in schema_results:
            summary.total_tables += len(schema_result.tables)
            summary.missing_tables += len(schema_result.missing_tables)
            summary.extra_tables += len(schema_result.extra_tables)

            for table in schema_result.tables:
                if table.status == ValidationStatus.PASS:
                    summary.passed_tables += 1
                elif table.status == ValidationStatus.ERROR:
                    summary.error_tables += 1
                    summary.failed_tables += 1
                else:
                    summary.failed_tables += 1

        return summary


# ----------------------------------------------------------------------
# Module-level convenience function (per spec section 22 / app.py usage)
# ----------------------------------------------------------------------
def validate_catalogs(
    databricks_connector: DatabricksConnector,
    source_catalog: str,
    target_catalog: str,
    **kwargs: Any,
) -> CatalogValidationResponse:
    """
    Thin convenience wrapper so callers (app.py, CLI, scripts) don't need
    to construct CatalogValidationRequest / CatalogValidator by hand for
    the common case.
    """
    request = CatalogValidationRequest(
        source_catalog=source_catalog,
        target_catalog=target_catalog,
        **kwargs,
    )
    validator = CatalogValidator(databricks_connector)
    return validator.compare_catalogs(request)
