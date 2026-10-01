"""Native grouped aggregation without arbitrary Python callbacks."""
from .._core import operation
import polars as pl
from polars.dataframe.group_by import GroupBy
from polars.lazyframe.group_by import LazyGroupBy
from .._core import WrangleError, require_native
from ..array._native import checked_seed


@operation(returns=('table',), recipe='never', aggregates=True)
def groupby_func(data, func, *, seed=0):
    """Aggregate a Polars GroupBy/LazyGroupBy using a named native reducer or pl.Expr. Entropy is normalized nonnegative mass entropy (nats)."""
    if not isinstance(data, (GroupBy, LazyGroupBy)):
        raise WrangleError("INVALID_INPUT", "Use a native Polars GroupBy/LazyGroupBy.")
    is_lazy = isinstance(data, LazyGroupBy)
    if not is_lazy:
        grouped = data.df.lazy().group_by(*data.by, **data.named_by, maintain_order=data.maintain_order)
        if getattr(data, "predicates", None):
            grouped = grouped.having(data.predicates)
    else:
        grouped = data
    if isinstance(func, pl.Expr):
        require_native(func)
        plan = grouped.agg(func)
        require_native(plan)
        return plan if is_lazy else plan.collect()
    if callable(func):
        raise WrangleError("NATIVE_EXPRESSION_REQUIRED", "Use a Polars expression; Python callbacks are not executed.")
    column = pl.all()
    basic = {"median", "mean", "first", "last", "std", "max", "min", "sum"}
    if func in basic:
        expression = getattr(column, func)()
    elif func == "random":
        expression = column.sample(n=1, seed=checked_seed(seed)).first()
    elif func == "freq":
        expression = column.mode().sort().first()
    elif func == "string":
        expression = column.cast(pl.String).str.join(" ")
    elif func == "entropy":
        prefix = "__wr_entropy_invalid_"
        try:
            checks = grouped.agg((column.is_null().any() | (~column.is_finite()).any() | (column < 0).any() | (column.sum() <= 0)).name.prefix(prefix))
            require_native(checks)
            checks = checks.collect()
        except pl.exceptions.PolarsError as error:
            raise WrangleError("INVALID_ENTROPY", "Entropy requires finite nonnegative numeric masses with positive group totals.") from error
        flags = [name for name in checks.columns if name.startswith(prefix)]
        if flags and checks.select(pl.any_horizontal(flags).any()).item():
            raise WrangleError("INVALID_ENTROPY", "Entropy requires finite nonnegative numeric masses with positive group totals.")
        expression = column.entropy(base=2.718281828459045, normalize=True)

    else:
        raise WrangleError("INVALID_AGGREGATION", "Declare a supported native aggregation.")
    plan = grouped.agg(expression)
    require_native(plan)
    return plan if is_lazy else plan.collect()
