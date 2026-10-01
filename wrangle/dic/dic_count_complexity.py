"""Native product of parameter-grid dimensions."""
from .._core import operation
import polars as pl
from .._core import WrangleError
from ..array._native import rows


@operation(returns=('scalar',), recipe='never')
def dic_count_complexity(dic):
    """Return the Cartesian grid size (empty dictionary = 1); reject products outside UInt64 rather than overflow."""
    if not isinstance(dic, dict):
        raise WrangleError("INVALID_INPUT", "Use a dictionary of Polars values or Python lists.")
    if not dic:
        return 1
    sizes = pl.Series("size", [rows(value) for value in dic.values()], dtype=pl.UInt64)
    if sizes.min() == 0:
        return 0
    # Native multiplication with a preflight logarithmic overflow check.
    if sizes.to_frame().select(pl.col("size").cast(pl.Float64).log().sum()).item() >= 44.3614195558365:
        raise WrangleError("RESOURCE_LIMIT", "Parameter-grid complexity exceeds UInt64 capacity.")
    return sizes.to_frame().select(pl.col("size").product()).item()
