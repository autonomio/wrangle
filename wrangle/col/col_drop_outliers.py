from .._core import operation
from .._core import reject_destructive, WrangleError, finish, frame
from ._expressions import numeric, positive


@operation(returns=('table',), recipe='yes')
def col_drop_outliers(data, col, mode="zscore", threshold=3, destructive=False):
    """Exclude missing values and rows outside explicit z-score or IQR fences.

    zscore uses population standard deviation; iqr uses linearly interpolated
    quartiles and [Q1 - threshold*IQR, Q3 + threshold*IQR]. Boundary values stay.
    Constant observed columns stay. Input order and other columns are preserved.
    """
    reject_destructive(destructive)
    if mode not in {"zscore", "iqr"}:
        raise WrangleError("INVALID_MODE", "mode must be 'zscore' or 'iqr'.")
    positive(threshold, "threshold")
    value = numeric(data, col, float64=True)
    if mode == "zscore":
        std = value.std(ddof=0)
        keep = (std == 0) | ((value - value.mean()).abs() <= threshold * std)
    else:
        q1, q3 = value.quantile(0.25, interpolation="linear"), value.quantile(0.75, interpolation="linear")
        width = q3 - q1
        keep = value.is_between(q1 - threshold * width, q3 + threshold * width, closed="both")
    return finish(frame(data).filter(value.is_not_null() & keep), data)
