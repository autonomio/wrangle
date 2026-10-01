"""Immutable deterministic parameter-grid sampling."""
from .._core import operation
from .._core import frame, WrangleError
from ..array._native import positive_int, rows, shuffled, restore, checked_seed


@operation(returns=('dictionary',), recipe='never')
def dic_resample_values(params, n, *, seed=0):
    """Return a new dictionary with n rows sampled without replacement from each value; undersized values fail."""
    positive_int(n, "n", allow_zero=True)
    checked_seed(seed)
    if not isinstance(params, dict):
        raise WrangleError("INVALID_INPUT", "Use a dictionary of Polars values or Python lists.")
    sizes = {name: rows(values) for name, values in params.items()}
    small = [name for name, size in sizes.items() if size < n]
    if small:
        raise WrangleError("INSUFFICIENT_ROWS", "Every parameter must contain at least n values.", {"parameters": small, "rows": sizes})
    return {name: restore(frame(shuffled(values, seed)).head(n), values) for name, values in params.items()}
