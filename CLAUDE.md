# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from the `comparison-service/` directory.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally (reload driven by RELOAD env var, default off)
uvicorn app:app --reload

# Run the full test suite
pytest

# Run a single test
pytest test_api.py::test_row_count_mismatch_detected

# Run the catalog-to-catalog validator from the CLI (no HTTP server needed)
python app.py validate-catalogs --source-catalog A --target-catalog B \
    [--schemas s1,s2] [--tables t1,t2] [--no-column-order] \
    [--mode COUNT_ONLY|STATISTICS|HASH|FULL] [--csv out.csv] [--excel out.xlsx] \
    [--primary-keys 'schema.table=col1,col2;other_table=col']
```

There is no lint/format tooling configured in this repo (no ruff/black/flake8 config present).

Config is via a `.env` file (loaded by `python-dotenv` in `app.py`) or real environment variables — required keys: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`, plus `AZURE_STORAGE_ACCOUNT` / `AZURE_STORAGE_KEY` / `AZURE_CONTAINER` for the Azure CSV path (`AZURE_STORAGE_ACCOUNT`/`AZURE_CONTAINER` default to `n8nstorages`/`n8ncontainer` if unset).

## Architecture

FastAPI POC service, designed to be called from **n8n workflows**, that validates data migrations in two independent modes. `app.py` is the only entrypoint (HTTP via uvicorn, or CLI via `python app.py validate-catalogs ...`); it wires connectors → engine/validator via FastAPI `Depends`, and does no comparison logic itself.

**Strict layering, enforced across all files**: connectors do I/O only and never decide pass/fail; the engine/validator classes contain all comparison/decision logic and never talk to a data source directly; `report_generator.py` only knows how to render an already-built result object to CSV/Excel.

### Two independent comparison paths (do not conflate them)

1. **Row-level CSV-vs-Databricks comparison** (`ComparisonEngine` in `comparison_engine.py`, `POST /compare`) — original/legacy path. Loads a full CSV from Azure Blob Storage (`AzureConnector.read_csv`) and a full Databricks table into pandas, then compares schema, duplicates, missing/extra rows, nulls, and dtypes in-memory. `ComparisonRequest`/`ComparisonResponse` in `models.py` (aliased from `CompareRequest`/`ComparisonResult` for backward compatibility). Note `_compare_values` is currently a stub that always reports zero mismatches.

2. **Databricks catalog-to-catalog validation** (`CatalogValidator` in `comparison_engine.py`, `POST /validate-catalogs` and `POST /validate-catalogs/report`) — the actively-developed path. Recursively walks catalog → schemas → tables → columns → data as a fixed ~15-stage sequence (see numbered stage comments in `comparison_engine.py` and `_validate_table`). Every check is pushed down as SQL via `DatabricksConnector` (`SHOW CATALOGS`/`SHOW SCHEMAS`/`information_schema.columns`, aggregate `SUM`/`COUNT DISTINCT`/`MIN`/`MAX` in one query per table, key-based `EXCEPT`/hash-join row diffs) — **no full table is ever loaded into pandas or collected**, since catalogs can be arbitrarily large. Status aggregation is never hardcoded per-stage; it always flows bottom-up through `CatalogValidator.calculate_overall_status` (ERROR > FAIL > SKIPPED > PASS precedence).

   - `DataCompareMode` (`COUNT_ONLY` / `STATISTICS` / `HASH` / `FULL`) controls how expensive stage 15 (actual row-level data diff) is. Default is `STATISTICS`, which relies on the row count/null/distinct/min-max checks already done above and explicitly skips row-level diffing. `HASH`/`FULL` require a configured primary/business key per table (`request.primary_keys`, keyed by `"schema.table"` or bare table name) or that table's data stage is safely skipped rather than erroring. The CLI's `--primary-keys 'schema.table=col1,col2;other_table=col'` flag builds this map; the HTTP API takes it directly as JSON.
   - Only in `FULL` mode, `DatabricksConnector.key_based_row_diff` additionally calls `_changed_row_detail`, which re-fetches the full key+value columns (plus a whole-row `hash()`) for the sampled changed keys from both sides, diffs them column-by-column in Python, and returns one entry per row with its mismatched column names, before/after values, and both row hashes. This becomes `DataValidationResult.sample_changed_detail: List[RowMismatchDetail]` (`models.py`) — the source for the Excel "Data Mismatches" sheet. `HASH` mode still only returns counts (no detail fetch) since it's meant to be cheaper than `FULL`.
   - `CatalogValidator._EXCLUDED_SCHEMAS` (currently `{"information_schema"}`) is filtered out in `compare_schemas` before common/missing/extra are computed, so Databricks' built-in system schema is never validated and never reported as a false missing/extra schema difference.
   - `ValidationStatus` (PASS/FAIL/ERROR/SKIPPED) is intentionally a different enum from `ComparisonStatus` (PASS/WARN/FAIL) used by the row-level path — ERROR (technical failure, e.g. permission denied) is kept distinct from FAIL (real validation mismatch) so one bad table doesn't get conflated with a genuine data difference, and doesn't abort validation of the rest of the catalog.
   - **Row-hash comparison stage** (`CatalogValidator._run_row_hash_stage`/`compare_row_hashes`, `DatabricksConnector.get_row_hashes`) — a separate, cheaper mechanism from `key_based_row_diff` above, and the *primary* way row-level mismatches get detected: it runs whenever a primary key is configured for a table, **independent of `data_compare_mode`** (so it fires even under the default `STATISTICS` mode, unlike `compare_data`'s HASH/FULL-gated diff). `get_row_hashes` issues one pushed-down query per side — `SELECT <pk_cols>, sha2(concat_ws('||', COALESCE(CAST(col AS STRING), '\x01NULL\x01'), ...), 256) AS row_hash FROM ...` over a fixed, alphabetically-sorted business-column list (PK excluded) so both sides hash identically; the `\x01NULL\x01` sentinel keeps real NULLs from colliding with an empty-string value. `compare_row_hashes` (pure logic, no I/O) joins both hash sets **by primary key, never row position**, and classifies every key as `MISMATCH` (hash differs), `MISSING_IN_TARGET`, or `MISSING_IN_SOURCE`, plus a `mismatch_percentage` over the union of keys seen on either side. Results land on the same `DataValidationResult` as `compare_data` (`row_hash_mismatches: List[RowHashMismatch]`, `row_hash_mismatch_count`, `row_hash_mismatch_percentage` in `models.py`) — if `compare_data` itself was SKIPPED (e.g. STATISTICS mode), this stage still overrides that table's `data.status` to PASS/FAIL based on its own findings.
   - `report_generator.py` has two output paths sharing the same row-builders: `generate_csv_report` writes a flat "Table Validation"-shaped `.csv`; `generate_excel_report` writes a 5-sheet `.xlsx` — **Summary** (overall status, timestamp/duration, total/passed/failed/error/skipped table counts, pass %), **Table Validation** (one row per table: Source/Target Schema+Table, Overall Status, Schema Match, Column Order, row counts, per-check statuses, Mismatch Count/%, Row Hash Mismatch Count/%, Validation Timestamp, Duration — sorted by schema, then FAIL/ERROR-before-PASS/SKIPPED, then table, and grouped into a collapsible Excel outline per schema), **Column Validation** (one row per column per table), **Data Mismatches** (one row per mismatched column per changed row, from `sample_changed_detail` — only populated in `FULL` mode with a key configured), and **Row Hash Mismatches** (one row per `RowHashMismatch` — Row #/Source Schema/Source Table/Primary Key/Source Hash/Target Hash/Mismatch Status, where Row # is a sequential number assigned after sorting — populated whenever a key is configured, any mode). All sheets get frozen header rows, autofit column widths, `auto_filter`, and PASS=green/FAIL=red/ERROR=orange/SKIPPED=gray fills via `STATUS_FILLS`/`STATUS_FONTS` (the Row Hash Mismatches sheet maps its non-`ValidationStatus` string statuses onto the same fills via `_ROW_HASH_STATUS_FILL_MAP`). "Schema Match" reuses `columns_status` (column-name match, not a separate concept). `POST /validate-catalogs/report?format=csv|excel` (default csv) and the CLI's `--csv`/`--excel` flags (either or both) select the format.

### Connectors (`azure_connector.py`, `databricks_connector.py`)

- `AzureConnector` — reads CSV blobs from Azure Storage into pandas; used only by the legacy `/compare` path.
- `DatabricksConnector` — SQL Warehouse connector (`databricks-sql-connector`), lazily connects on first use, all identifiers are backtick-quoted via `_quote_ident`/`_qualify` to build safe fully-qualified names. Exposes both low-level metadata methods (`get_schemas`, `get_tables`, `get_table_schema`, `get_row_count`, `get_column_statistics`), the push-down `key_based_row_diff` used for HASH/FULL mode, and `get_row_hashes` (single per-side `sha2`/`concat_ws` query) for the always-on row-hash comparison stage.

### Known repo quirks

- `comparison-service/CatalogValidator.py` is **not** the catalog validator — despite its name it contains an unused, unwired `AzureDataLakeConnector` (ADLS Gen2) class. Nothing imports it. The real `CatalogValidator` class lives in `comparison_engine.py`.
- `comparison-service/test_api.py` is the pytest suite for `CatalogValidator` (all Databricks calls mocked via `unittest.mock.MagicMock`, no live Databricks needed) — despite the name, it does not test `app.py`'s HTTP layer directly.
