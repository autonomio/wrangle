from .._core import operation
import math
import polars as pl
from .._core import WrangleError, as_series
from ._expressions import numeric


@operation(returns=('series',), recipe='never')
def col_rescale_max(values, scale=1, to_int=False):
    """Multiply numeric observations so their maximum equals scale; retain missing.

    This is maximum scaling, not min-max normalization. A zero maximum raises
    unless all observed values are zero, which remain zero. A negative maximum
    raises because multiplication would reverse the ordering. to_int uses ceiling.
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise WrangleError("INVALID_PARAMETER", "scale must be finite and positive.")
    series = as_series(values)
    table = series.rename(series.name or "value").to_frame()
    name = table.columns[0]
    value = numeric(table, name, float64=True).cast(pl.Float64)
    maximum = table.select(value.max()).item()
    if maximum is None:
        raise WrangleError("NO_OBSERVATIONS", "Cannot scale a column without observed values.")
    if maximum < 0:
        raise WrangleError("INVALID_DOMAIN", "Maximum must be positive, or every observed value must be zero.")
    if maximum == 0:
        if table.select((value != 0).any()).item():
            raise WrangleError("ZERO_DENOMINATOR", "Maximum is zero with nonzero observations.")
        result = value
    elif maximum < 2.0**-1022:
        # Native division uses a reciprocal; shift subnormals into normal range.
        shift = 2.0**512
        result = value * shift / (maximum * shift) * scale
    else:
        result = value / maximum * scale
    if to_int:
        result = result.ceil().cast(pl.Int64)
    return table.select(result.alias(series.name)).to_series()
