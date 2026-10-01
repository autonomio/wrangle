"""Explicit one-hot label encoding using native Polars expressions."""
from .._core import operation
import polars as pl
from .._core import as_series, WrangleError
from ._native import positive_int


@operation(returns=('table',), recipe='yes')
def array_to_multilabel(y, *, classes=None):
    """Encode nonnegative integer labels into columns class_0...; classes fixes the mapping across batches."""
    values = as_series(y)
    if not values.dtype.is_integer() or values.null_count() or values.is_empty() or values.min() < 0:
        raise WrangleError("INVALID_LABELS", "One-hot labels must be non-null nonnegative integers.")
    required = values.max() + 1
    count = required if classes is None else positive_int(classes, "classes")
    if count < required:
        raise WrangleError("UNKNOWN_CATEGORY", "A label exceeds the declared class count.")
    if count > 10000:
        raise WrangleError("RESOURCE_LIMIT", "One-hot class count exceeds 10000; use categorical labels.")
    return values.to_frame().select([(pl.col(values.name) == i).cast(pl.UInt8).alias(f"class_{i}") for i in range(count)])
