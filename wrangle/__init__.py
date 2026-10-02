"""Native Polars data preparation for scientific researchers and their agents."""
from importlib import import_module

from ._api import inspect, prepare
from ._core import WrangleError

__version__ = "1.0.1"
__all__ = ["inspect", "prepare", "WrangleError"]


def __getattr__(name):
    # The historical operation names remain directly discoverable without
    # importing every operation before a researcher can inspect one file.
    if name == "utils":
        return import_module(".utils", __name__)
    if name == "shuffle":
        name = "array_random_shuffle"
    prefix = name.split("_", 1)[0]
    if prefix in {"df", "col", "array", "dic"}:
        try:
            return getattr(import_module(f".{prefix}.{name}", __name__), name)
        except (ModuleNotFoundError, AttributeError) as error:
            raise AttributeError(f"wrangle has no operation {name!r}") from error
    raise AttributeError(f"wrangle has no operation {name!r}")
