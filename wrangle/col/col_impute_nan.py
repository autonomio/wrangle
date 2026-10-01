from .._core import operation
from .._core import WrangleError, as_series
from ._expressions import clean, numeric, seed_value


@operation(returns=('series',), recipe='never')
def col_impute_nan(data, impute_mode="mean_by_std", *, seed=0):
    """Return a Series with missing values filled using observed values only.

    mean/median require numbers. mode/common choose the smallest tied mode.
    mean_by_std draws seeded continuous uniform values from mean +/- sample std;
    a singleton uses its observed value. Existing observations remain unchanged.
    All-missing input raises. Fit imputation parameters on training data only.
    """
    if impute_mode not in {"mean", "median", "mode", "common", "mean_by_std"}:
        raise WrangleError("INVALID_MODE", "Unknown imputation mode.")
    seed_value(seed)
    series = as_series(data)
    name = series.name or "value"
    table = series.rename(name).to_frame()
    value = clean(table, name)
    if table.select(value.count()).item() == 0:
        raise WrangleError("NO_OBSERVATIONS", "Cannot impute a column without observed values.")
    if impute_mode not in {"mode", "common"}:
        numeric(table, name, float64=True)
    # Imported when called to keep the legacy package namespaces acyclic.
    from ..df._expressions import impute_expr
    result = impute_expr(name, table.schema[name], impute_mode, seed=seed)
    return table.select(result.alias(series.name)).to_series()
