"""Seeded weighted sampling with replacement using native cumulative probability joins."""
from .._core import operation
import polars as pl
from .._core import frame, as_series, WrangleError
from ._native import rows, restore, positive_int, checked_seed


@operation(returns=('series', 'table'), recipe='conditional')
def array_random_weighted(x, weights, size, *, seed=0):
    """Sample size rows from explicit nonnegative weights, preserving input column types and row alignment."""
    positive_int(size, "size", allow_zero=True)
    checked_seed(seed)
    if isinstance(weights, str):
        raise WrangleError("EXPLICIT_WEIGHTS_REQUIRED", "Declare one weight per source observation; legacy random distribution names had ambiguous weighting semantics.")
    try:
        weights = as_series(weights).rename("__wr_weight") if isinstance(weights, (pl.Series, pl.DataFrame, pl.LazyFrame)) else pl.Series("__wr_weight", weights, strict=False)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise WrangleError("INVALID_WEIGHTS", "Provide numeric weights aligned with source observations.") from error
    count = rows(x)
    if len(weights) != count or not count or not weights.dtype.is_numeric():
        raise WrangleError("INVALID_WEIGHTS", "Weights must be numeric and aligned with nonempty source observations.")
    weights = weights.cast(pl.Float64)
    checked = weights.to_frame().select((pl.col("__wr_weight").is_null() | ~pl.col("__wr_weight").is_finite() | (pl.col("__wr_weight") < 0)).any(), pl.col("__wr_weight").sum().alias("total"), pl.col("__wr_weight").sum().is_finite().alias("finite_total"))
    if checked.item(0, 0) or checked.item(0, 1) <= 0 or not checked.item(0, 2):
        raise WrangleError("INVALID_WEIGHTS", "Weights must be finite, nonnegative and have a positive total.")
    source = frame(x)
    names = source.collect_schema().names()
    if any(name.startswith("__wr_") for name in names):
        raise WrangleError("RESERVED_COLUMN", "Weighted sampling reserves __wr_ column names.")
    cdf = source.with_columns(weights).filter(pl.col("__wr_weight") > 0).with_columns((pl.col("__wr_weight") / pl.col("__wr_weight").sum()).cum_sum().alias("__wr_cdf")).drop("__wr_weight").with_columns(pl.when(pl.int_range(0, pl.len()) == pl.len() - 1).then(1.0).otherwise(pl.col("__wr_cdf")).alias("__wr_cdf"))
    # 53 hash bits avoid Float64 rounding to the upper probability boundary.
    draws = pl.LazyFrame().select(pl.int_range(0, size).alias("__wr_draw")).with_columns(((pl.col("__wr_draw").hash(seed=seed) // 2048).cast(pl.Float64) / 2**53).alias("__wr_uniform")).sort("__wr_uniform")
    result = draws.join_asof(cdf, left_on="__wr_uniform", right_on="__wr_cdf", strategy="forward", check_sortedness=False).sort("__wr_draw").select(names)
    return restore(result, x)
