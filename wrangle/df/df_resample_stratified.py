from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, reject_destructive, WrangleError
from ._expressions import positive_n, seed_value, temp_name


@operation(returns=('table',), recipe='yes')
def df_resample_stratified(data, strat_col, n=None, destructive=False, *, seed=0):
    """Select the same seeded number of observations per stratum, without replacement."""
    reject_destructive(destructive)
    require_columns(data, [strat_col])
    positive_n(n, allow_none=True)
    seed_value(seed)
    minimum = frame(data).group_by(strat_col).len().select(pl.col("len").min()).collect().item()
    if minimum is None:
        return finish(frame(data), data)
    if n is None:
        n = minimum
    if n > minimum:
        raise WrangleError("INSUFFICIENT_SAMPLES", "n exceeds the smallest stratum; choose a smaller n.", {"requested": n, "available_per_stratum": minimum})
    row = temp_name(data)
    plan = frame(data).with_row_index(row).sort(pl.col(row).hash(seed=seed), pl.col(row)).filter(pl.col(row).cum_count().over(strat_col) <= n).sort(row).drop(row)
    return finish(plan, data)
