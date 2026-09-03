# table-validator

Cross-platform data migration validator. Compares a source table against a
target Databricks table/catalog and reports whether the migration is
correct: matching schema, row counts, column statistics, and (where a
difference is found) the exact row/column that changed. Produces a
multi-sheet Excel report and a pass/fail summary.

Currently supported sources (target is always Databricks):

- Databricks catalog &rarr; Databricks catalog
- Azure Blob Storage (CSV / Excel / Parquet) &rarr; Databricks table
- Azure SQL Database &rarr; Databricks catalog
- Azure Synapse SQL pool (dedicated or serverless) &rarr; Databricks catalog

More source platforms can be added behind the same connector/validator
interfaces described below.

## Install

```bash
pip install table-validator
```

From a local checkout (the directory containing `pyproject.toml`):

```bash
pip install .
```

For local development (editable install, so code changes take effect
without reinstalling):

```bash
pip install -e ".[dev]"
```

Either way, this installs the `tablevalidator` command on your PATH.

## CLI usage

```bash
tablevalidator info
```

Prints what the tool does and the commands below, in the order you'd
normally run them.

```bash
tablevalidator configure
```

Interactive wizard that walks you through:

1. Azure Storage account + container (optional, skip if you don't have a Blob source) and account key
2. Azure SQL server + database (optional, skip if you don't have a SQL source) and username/password
3. Azure Synapse SQL pool endpoint + database (optional, skip if you don't have a Synapse source), then either a SQL login (username/password) or a Microsoft Entra ID service principal (tenant ID + client ID + client secret)
4. Databricks workspace URL, SQL Warehouse HTTP path, and personal access token
5. Source table (catalog / schema / table)
6. Target table (catalog / schema / table)
7. Which validations to run (catalog / schema / column / row)

```bash
tablevalidator validate
```

Runs the comparison using the saved configuration and writes
`validation_report.xlsx` in the current directory, printing a pass/fail
summary to the console. Exit code is `0` if the overall result is PASS,
non-zero otherwise (useful in CI). Useful flags:

```bash
tablevalidator validate --config-path /path/to/config.yaml --output /path/to/report.xlsx
```

```bash
tablevalidator open
```

Opens the most recently generated report in your default spreadsheet app.

## Notebook usage (zero config, inside Databricks)

From inside a Databricks notebook cell, `validate_tables()` compares two
tables using the notebook's own already-authenticated Spark session - no
workspace URL, personal access token, or SQL Warehouse HTTP path to set
up. This is a second, independent way to use the package alongside the
CLI above (pick whichever fits: the CLI for a scheduled/scripted run
against a SQL Warehouse, this for an ad-hoc in-notebook check) - both run
the exact same full-depth comparison engine.

```python
%pip install table-validator

from table_validator import validate_tables

result = validate_tables(
    "catalog1.schema1.table1",
    "catalog1.schema1.table2",
)
print(result)  # compact summary
```
```
Overall status: FAIL
Tables: 1 total, 0 passed, 1 failed, 0 error, 0 skipped
```

`result` also exposes each of the Excel report's own sheets as a
plain-text table - print only the one you want:

```python
print(result.table_validation)    # one row per table
print(result.column_validation)   # one row per column
print(result.data_mismatches)     # one row per mismatched cell (FULL mode)
print(result.row_hash_mismatches) # one row per mismatched primary key
print(result.mismatch_categories) # root-cause breakdown (NULL_MISMATCH, etc.)
print(result.suggestions)         # plain-English fix suggestions
```
```
Source Schema Source Table Overall Status  Row Count (Src)  Row Count (Tgt)  ...
         sch1           t1           FAIL                5              100  ...
```

Each of these is a small `ResultTable` object - `.headers`/`.rows` for
programmatic access, or `.to_dataframe()` if you want a real pandas
`DataFrame` to filter/sort/export yourself. `result.response` is the raw
`CatalogValidationResponse`, for full programmatic access beyond the
sheet breakdown.

Optional keyword arguments to `validate_tables()`: `primary_key` (a real
key for row-level comparison instead of the ROW_NUMBER() fallback, single-
table mode only), `ignore_columns`, `only_columns`, `column_map` (for a
renamed column), `ignore_datatype_columns` (skip a real type mismatch on
these columns rather than failing the table on it). Outside a Databricks
notebook (e.g. local development against a real Spark session), install
the `spark` extra: `pip install "table-validator[spark]"` - inside an
actual Databricks notebook, pyspark and an active session already exist,
so the bare `%pip install table-validator` above is sufficient.

### Schema-wide sweep (no single table named)

Leave the table off both `source`/`target` (just `"catalog.schema"`) to
compare every identically-named table in that schema in one call - the
same auto-discovery the CLI's own blank-table config triggers, with zero
further setup:

```python
result = validate_tables("catalog1.bronze", "catalog2.silver")
print(result)                  # lists every matched table's status
print(result.table_validation) # one row per matched table
```

If some tables were renamed between source and target, pass `table_map`
(source name &rarr; target name) - unmapped tables are still matched by
identical name as usual:

```python
result = validate_tables(
    "catalog1.bronze", "catalog2.silver",
    table_map={"cust": "customers", "ord": "orders"},
)
```

`primary_key` isn't valid in this mode (a single key can't apply to every
table in the sweep) - compare one table at a time
(`"catalog.schema.table"` on both sides) if you need row-level detail via
a real key.

## Quickstart (Python API)

Everything the CLI does is available as a library, built from the same
public API exported by `table_validator/__init__.py`:

```python
from table_validator import (
    load_config,
    CatalogValidator,
    CatalogValidationRequest,
    DatabricksConnector,
)
from table_validator.auth.databricks_auth import get_databricks_token

# Non-secret settings from ~/.table_validator/config.yaml
# (see `tablevalidator configure`); secrets from ~/.table_validator/.env.
config = load_config()
token = get_databricks_token(config)

# DatabricksConnector wants a bare hostname, not the full workspace URL
host = config.databricks.workspace_url.replace("https://", "").split("/")[0]

databricks = DatabricksConnector(
    host=host,
    token=token,
    http_path=config.databricks.http_path,
)

validator = CatalogValidator(databricks)

request = CatalogValidationRequest(
    source_catalog="source_catalog_name",
    target_catalog="target_catalog_name",
    schemas=["sales"],          # optional: restrict scope
    primary_keys={"sales.orders": ["order_id"]},  # optional: enables row-level diffing
)

result = validator.compare_catalogs(request)

print(result.status)  # PASS / FAIL / ERROR / SKIPPED
```

Other public entry points exported from `table_validator`:

- `AzureCsvValidator` / `AzureSqlValidator` — the Blob-CSV and Azure-SQL
  equivalents of `CatalogValidator`, returning the same
  `CatalogValidationResponse` shape.
- `BlobCatalogValidator` — validates every file in an Azure Blob container
  against like-named Databricks tables.
- `AzureConnector` / `AzureSqlConnector` — the Azure-side connectors, for
  building your own validation flow against a connector directly.
- `ValidatorConfig`, `default_config`, `save_config`, `require_config`,
  `ConfigNotFoundError` — the same config load/save layer the CLI wizard
  uses, if you want to construct or persist configuration programmatically.
- `validate_tables` / `SparkConnector` — the notebook-native entry point
  described above (lazily imported: importing `table_validator` itself
  never requires `pyspark` to be installed).

## Where config and secrets are stored

Everything lives outside the repo, under your home directory:

- `~/.table_validator/config.yaml` — non-secret configuration (table
  references, workspace URL, which validations are enabled). Safe to
  inspect or version-control separately if you want.
- `~/.table_validator/.env` — credentials (Azure Storage key, Azure SQL
  username/password, Databricks personal access token), written in
  plaintext and restricted to owner read/write only (`chmod 600`,
  best-effort on Windows since NTFS doesn't map POSIX permission bits).
  **Never commit this file or add it inside the project repo** — it isn't,
  by construction, since it's written under your home directory rather
  than the working directory.

Optional: `DATABRICKS_RETRY_TIMEOUT_SECONDS` (a plain environment variable,
not part of `config.yaml`) raises the CloudFetch HTTP retry timeout above
its 300-second default. On a slow or unstable network, downloading a
large row-hash result set can legitimately take longer than that, causing
a `Retry request would exceed Retry policy max retry duration` failure
even though the query itself succeeded. Set it higher (e.g. `900` for 15
minutes) if you hit this on large tables.

A future version will replace manual credential entry with Azure CLI /
Service Principal auth and Databricks CLI / OAuth login, without changing
the config file format or any command usage above.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
