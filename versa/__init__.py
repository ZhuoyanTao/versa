"""VERSA package initialization.

Optional metric modules are loaded lazily so lightweight helpers such as
reporting and aggregation do not import model-backed dependencies.
"""

import os
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

from versa.metric_registry import load_metric_symbol, metric_symbol_names

try:
    __version__ = version("versa-speech-audio-toolkit")
except PackageNotFoundError:
    __version__ = "1.1.0"

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path.cwd() / "versa_cache" / "numba_cache")
)


def __getattr__(name):
    if name in metric_symbol_names():
        symbol = load_metric_symbol(name)
        globals()[name] = symbol
        return symbol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(metric_symbol_names()))
