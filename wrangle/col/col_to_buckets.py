from .._core import operation
import polars as pl
from .._core import WrangleError, frame, require_columns
from ._expressions import numeric


@operation(returns=('series',), recipe='never')
def col_to_buckets(data, col, cuts=5, rounding=False):
    """Return a Series of equal-width 'left to right' labels; missing stays null.

    Intervals are right-closed; the first interval includes the minimum.
    Constant observations receive a single 'value to value' label. rounding
    rounds displayed boundaries to whole numbers only, not the assigned bins.
    """
    require_columns(data, [col])
    if isinstance(cuts, bool) or not isinstance(cuts, int) or cuts < 1:
        raise WrangleError("INVALID_PARAMETER", "cuts must be a positive integer.")
    value = numeric(data, col, float64=True).cast(pl.Float64)
    lo, hi = value.min(), value.max()
    width = (hi - lo) / cuts
    index = ((value - lo) / width).ceil().cast(pl.Int64, strict=False).sub(1).clip(0, cuts - 1)
    left, right = lo + index * width, lo + (index + 1) * width
    left = pl.when(hi == lo).then(lo).otherwise(left)
    right = pl.when(hi == lo).then(hi).otherwise(right)
    if rounding:
        left, right = left.round(0).cast(pl.Int64), right.round(0).cast(pl.Int64)
    label = pl.concat_str(left.cast(pl.String), pl.lit(" to "), right.cast(pl.String))
    result = pl.when(value.is_not_null()).then(label).otherwise(None).alias(col)
    return frame(data).select(result).collect().to_series()
