from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, reject_destructive
from ..col.col_corr_ols import col_corr_ols


@operation(returns=('table', 'tuple'), recipe='conditional')
def df_corr_ols(data, y, destructive=False, *, return_mapping=False):
    """Rank categorical columns by group means of a numeric target; target leakage is explicit."""
    reject_destructive(destructive)
    require_columns(data, [y])
    plan = frame(data)
    mappings = {}
    for name, dtype in plan.collect_schema().items():
        if name != y and (dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum)):
            plan, mapping = col_corr_ols(plan, name, y, return_mapping=True)
            mappings[name] = mapping
    result = finish(frame(plan), data)
    return (result, mappings) if return_mapping else result
