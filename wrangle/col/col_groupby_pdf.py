from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_groupby_pdf(data, col, metric_col, ascending=False):
    """Return grouped empirical probability masses in a PDF column.

    The historical name PDF is retained; discrete observations produce a PMF,
    not a continuous density. Missing metric observations are excluded. Masses
    sum to 1 within every observed group. Sorting changes display order only.
    """
    require_columns(data, [col, metric_col])
    if col == metric_col or "PDF" in {col, metric_col}:
        raise WrangleError("INVALID_PARAMETER", "Group, metric, and output names must differ.")
    plan = frame(data).with_columns(clean(data, col), clean(data, metric_col)).filter(
        pl.col(metric_col).is_not_null()
    ).group_by([col, metric_col], maintain_order=True).agg(pl.len().alias("__n")).with_columns(
        (pl.col("__n") / pl.col("__n").sum().over(col)).alias("PDF")
    ).drop("__n").sort([col, metric_col], descending=[False, not ascending], nulls_last=True)
    return finish(plan, data)
