from .._core import operation
from .._core import frame, finish, reject_destructive
from ._expressions import fraction, numeric_only, correlation
from .df_corr_pearson import df_corr_pearson


@operation(returns=('table',), recipe='yes')
def df_drop_weak(data, y, min_correlation=.2, method='pearson', drop_object=True, destructive=False):
    """Keep the target and features meeting absolute pairwise correlation; remove undefined correlations."""
    reject_destructive(destructive)
    fraction(min_correlation, "min_correlation")
    if method not in {"pearson", "spearman", "kendall"}:
        from .._core import WrangleError
        raise WrangleError("INVALID_OPTION", "method must be pearson, spearman, or kendall.")
    schema = numeric_only(data, [y])
    coefficients = df_corr_pearson(data, y) if method == "pearson" else None
    selected = []
    for name, dtype in schema.items():
        if name == y:
            selected.append(name)
        elif dtype.is_numeric():
            value = coefficients[name][0] if coefficients is not None else correlation(data, name, y, method)
            if value is not None and abs(value) >= min_correlation:
                selected.append(name)
        elif not drop_object:
            selected.append(name)
    return finish(frame(data).select(selected), data)
