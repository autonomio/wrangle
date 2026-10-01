from .._core import operation
import math
import polars as pl
from .._core import reject_destructive, WrangleError, finish, frame, require_columns
from ._expressions import clean, numeric


@operation(returns=('table',), recipe='yes')
def col_to_binary(data, col, func="median", destructive=False):
    """Encode a column using an explicit threshold or deterministic categories.

    mean/median/mode compare >= a fitted statistic; integer func is a literal
    threshold; float func is a quantile in [0,1]. cat_string produces sorted
    dense codes from 0; cat_numeric/cat_int produce five quantile bins from 0.
    Missing values stay null. 'none' passes through. Fit on training data only.
    """
    reject_destructive(destructive)
    require_columns(data, [col])
    if not isinstance(func, (str, int, float)):
        raise WrangleError("INVALID_MODE", "func must be a supported mode, integer threshold, or float quantile.")
    value = clean(data, col)
    if func == "none":
        return finish(frame(data), data)
    if func == "cat_string":
        result = value.rank("dense").cast(pl.Int64) - 1
    elif func in {"cat_numeric", "cat_int"}:
        value = numeric(data, col, float64=True)
        # Dense codes avoid empty intervals after repeated quantile boundaries.
        bins = value.qcut(5, labels=[str(i) for i in range(5)], allow_duplicates=True).cast(pl.String).cast(pl.Int64)
        result = bins.rank("dense").cast(pl.Int64) - 1
    elif isinstance(func, bool):
        raise WrangleError("INVALID_PARAMETER", "Boolean func is not a threshold.")
    elif isinstance(func, int):
        result = numeric(data, col) >= func
    elif isinstance(func, float):
        if not math.isfinite(func) or not 0 <= func <= 1:
            raise WrangleError("INVALID_PARAMETER", "A float func must be a quantile in [0, 1].")
        value = numeric(data, col, float64=True)
        result = value >= value.quantile(func, interpolation="linear")
    elif func in {"mean", "median", "mode"}:
        value = numeric(data, col, float64=True)
        threshold = value.drop_nulls().mode().sort().first() if func == "mode" else getattr(value, func)()
        result = value >= threshold
    else:
        raise WrangleError("INVALID_MODE", "Unsupported binary conversion mode.")
    return finish(frame(data).with_columns(result.alias(col)), data)
