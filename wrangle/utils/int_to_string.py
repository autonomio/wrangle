"""Native zero-padded character lists."""
from .._core import operation
import polars as pl
from .._core import as_series, WrangleError


@operation(returns=('series',), recipe='never')
def int_to_chars(x):
    """Return a List(String) Series; require non-null nonnegative integers."""
    values = as_series(x)
    if not values.dtype.is_integer() or values.null_count() or (len(values) and values.min() < 0):
        raise WrangleError("INVALID_INPUT", "Character encoding requires non-null nonnegative integers.")
    if values.is_empty():
        return pl.Series(values.name, [], dtype=pl.List(pl.String))
    plan = values.to_frame()
    width = plan.select(pl.col(values.name).cast(pl.String).str.len_chars().max()).item()
    return plan.select(pl.col(values.name).cast(pl.String).str.pad_start(width, "0").str.split("").alias(values.name)).to_series()
