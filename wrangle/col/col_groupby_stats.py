from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean, numeric


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_groupby_stats(data, col, y):
    """Return observed n, sum, mean, and sample std per category.

    Missing outcomes are excluded; n=0 gives null sum/mean/std. Missing category
    forms its own group. Values retain full precision. Sample std uses ddof=1.
    """
    require_columns(data, [col, y])
    if col in {"n", "sum", "mean", "std"}:
        raise WrangleError("COLUMN_COLLISION", "Category name conflicts with a statistic output name.")
    dtype = frame(data).collect_schema()[y]
    value = clean(data, y).cast(pl.Float64) if dtype == pl.Boolean else numeric(data, y, float64=True)
    total = value.cast(pl.Int128).sum() if dtype.is_integer() else value.sum()
    plan = frame(data).with_columns(clean(data, col)).group_by(col, maintain_order=True).agg(
        value.count().alias("n"),
        pl.when(value.count() > 0).then(total).otherwise(None).alias("sum"),
        value.mean().alias("mean"), value.std(ddof=1).alias("std")
    )
    return finish(plan, data)
