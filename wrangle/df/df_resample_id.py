from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, reject_destructive
from ._expressions import positive_n, seed_value, temp_name


@operation(returns=('table',), recipe='yes')
def df_resample_id(data, id_col, n=None, destructive=False, *, seed=0):
    """Keep the first row per ID, then select a seeded sample in source order."""
    reject_destructive(destructive)
    require_columns(data, [id_col])
    positive_n(n, allow_none=True)
    seed_value(seed)
    row = temp_name(data)
    plan = frame(data).with_row_index(row).unique(subset=[id_col], keep="first", maintain_order=True)
    if n is not None:
        available = plan.select(pl.len()).collect().item()
        if n > available:
            from .._core import WrangleError
            raise WrangleError("INSUFFICIENT_SAMPLES", "n exceeds the number of distinct IDs.", {"requested": n, "available": available})
        plan = plan.sort(pl.col(row).hash(seed=seed), pl.col(row)).head(n)
    return finish(plan.sort(row).drop(row), data)
