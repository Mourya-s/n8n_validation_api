"""
Regression test for the hard requirement behind the notebook-native API:
`import table_validator` (and accessing every OTHER __all__ name) must
keep working with zero pyspark installed - CLI-only users must never be
forced to install pyspark just to use the package at all, even though
SparkConnector/validate_tables now live in the same __all__ list.

Run in a subprocess with pyspark import simulated as unavailable, rather
than monkeypatching sys.modules/builtins.__import__ in-process, since
that kind of global tampering would otherwise leak into every other test
module that imports table_validator afterward in the same test session.
"""

from __future__ import annotations

import subprocess
import sys


_SCRIPT = """
import builtins
import sys

real_import = builtins.__import__

def fake_import(name, *args, **kwargs):
    if name == "pyspark" or name.startswith("pyspark."):
        raise ImportError("No module named pyspark (simulated)")
    return real_import(name, *args, **kwargs)

builtins.__import__ = fake_import

import table_validator
assert table_validator.CatalogValidator is not None
assert table_validator.DatabricksConnector is not None

# Accessing the lazy names themselves must not require pyspark either -
# notebook.py/spark_connector.py only import pyspark inside
# SparkConnector.__init__, when actually constructing without an
# explicit spark= session.
validate_tables = table_validator.validate_tables
SparkConnector = table_validator.SparkConnector
assert callable(validate_tables)
assert SparkConnector is not None

# Actually trying to use SparkConnector with no session AND no pyspark
# must fail clearly at that point, not earlier.
try:
    SparkConnector()
    raise SystemExit("expected ImportError, none was raised")
except ImportError:
    pass

print("OK")
"""


def test_import_table_validator_succeeds_without_pyspark():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout
