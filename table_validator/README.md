# table-validator

CLI tool for validating data migrations between Azure (Blob Storage / SQL Database) and Databricks Delta Lake catalogs. Validates schema, column, row-count/statistics, and row-level data differences, and produces a multi-sheet Excel report.

## Install

From the project directory (contains `pyproject.toml`):

```bash
pip install .
```

For local development (editable install, so code changes take effect without reinstalling):

```bash
pip install -e .
```

This installs the `tablevalidator` command on your PATH.

## First run

```bash
tablevalidator configure
```

Walks you through an interactive wizard:

1. Azure Storage account + container (optional, skip if you don't have a Blob source) and account key
2. Azure SQL server + database (optional, skip if you don't have a SQL source) and username/password
3. Databricks workspace URL, SQL Warehouse HTTP path, and personal access token
4. Source table (catalog / schema / table)
5. Target table (catalog / schema / table)
6. Which validations to run (catalog / schema / column / row)

Then run the validation:

```bash
tablevalidator validate
```

This produces `validation_report.xlsx` in the current directory and prints a pass/fail summary to the console. Exit code is `0` if the overall result is PASS, non-zero otherwise (useful in CI).

Useful flags:

```bash
tablevalidator validate --config-path /path/to/config.yaml --output /path/to/report.xlsx
```

## Where config and secrets are stored

Everything lives outside the repo, under your home directory:

- `~/.table_validator/config.yaml` — non-secret configuration (table references, workspace URL, which validations are enabled). Safe to inspect or version-control separately if you want.
- `~/.table_validator/.env` — credentials (Azure Storage key, Azure SQL username/password, Databricks personal access token), written in plaintext and restricted to owner read/write only (`chmod 600`, best-effort on Windows since NTFS doesn't map POSIX permission bits). **Never commit this file or add it inside the project repo** — it isn't, by construction, since it's written under your home directory rather than the working directory.

A future version will replace manual credential entry with Azure CLI / Service Principal auth and Databricks CLI / OAuth login, without changing the config file format or any command usage above.

## Development

```bash
pip install -e ".[dev]"
pytest
```
