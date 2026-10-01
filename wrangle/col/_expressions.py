"""Native expression helpers shared by the column operations."""
import math
import polars as pl
from .._core import WrangleError, columns, frame, require_columns


def clean(data, name):
    """Represent floating NaN as null; preserve every other value."""
    dtype = frame(data).collect_schema()[name]
    value = pl.col(name)
    return value.fill_nan(None) if dtype.is_float() else value


def numeric(data, name, *, float64=False):
    require_columns(data, [name])
    dtype = frame(data).collect_schema()[name]
    if not dtype.is_numeric():
        raise WrangleError("NON_NUMERIC_COLUMN", f"Column {name!r} must be numeric.")
    value = clean(data, name)
    if dtype.is_float():
        invalid = frame(data).select((value.is_not_null() & ~value.is_finite()).any()).collect().item()
        if invalid:
            raise WrangleError("NON_FINITE_VALUE", f"Column {name!r} contains infinity.")
    if float64:
        # Runtime import avoids cycles between the public legacy namespaces.
        from ..df._expressions import require_float_exactness
        require_float_exactness(data, [name])
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WrangleError("INVALID_PARAMETER", f"{name} must be a finite positive number.")


def seed_value(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise WrangleError("INVALID_SEED", "seed must be an integer between 0 and 2**64 - 1.")


def temporary(data, stem="__wrangle_row"):
    names = columns(data)
    while stem in names:
        stem += "_"
    return stem


def levels(data, name):
    require_columns(data, [name])
    return frame(data).select(clean(data, name).drop_nulls().unique().sort()).collect().to_series().to_list()


def names_available(data, names, removed=()):
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise WrangleError("INVALID_COLUMN_NAMES", "Output column names must be nonempty, unique strings.")
    collisions = set(names) & (set(columns(data)) - set(removed))
    if collisions:
        raise WrangleError("COLUMN_COLLISION", f"Output columns already exist: {sorted(collisions)!r}.")
