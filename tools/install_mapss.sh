#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" -c '
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        "MAPSS 1.1.2 requires Python 3.10-3.12; "
        f"this interpreter is Python {sys.version_info.major}.{sys.version_info.minor}."
    )
'

"$PYTHON_BIN" -c '
from importlib.metadata import PackageNotFoundError, version
try:
    installed = version("transformers")
except PackageNotFoundError:
    pass
else:
    major_minor = tuple(int(part) for part in installed.split(".")[:2])
    if major_minor >= (4, 53):
        raise SystemExit(
            "MAPSS 1.1.2 requires transformers<4.53, but this environment has "
            f"transformers {installed}. Refusing to modify the environment; "
            "install MAPSS in a dedicated compatible virtual environment."
        )
'

"$PYTHON_BIN" -m pip install "mapss-measures==1.1.2"
"$PYTHON_BIN" -m pip check

echo "MAPSS 1.1.2 is ready. The default backbone downloads on first use."
