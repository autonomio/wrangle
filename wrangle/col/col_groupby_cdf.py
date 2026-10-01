from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_groupby_cdf(data, col, metric_col, ascending=False):
    """Return P(metric <= value) within each group in a CDF column.

    Missing metric observations are excluded. The cumulative probability is
    computed on ascending support, independently of requested display order.
    """
    require_columns(data, [col, metric_col])
    if col == metric_col or "CDF" in {col, metric_col}:
        raise WrangleError("INVALID_PARAMETER", "Group, metric, and output names must differ.")
    plan = frame(data).with_columns(clean(data, col), clean(data, metric_col)).filter(
        pl.col(metric_col).is_not_null()
    ).group_by([col, metric_col], maintain_order=True).agg(pl.len().alias("__n")).sort([col, metric_col]).with_columns(
        (pl.col("__n").cum_sum().over(col) / pl.col("__n").sum().over(col)).alias("CDF")
    ).drop("__n").sort([col, metric_col], descending=[False, not ascending], nulls_last=True)
    return finish(plan, data)
