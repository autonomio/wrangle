"""Finite-number recognition via native Polars casting."""
from .._core import operation
import polars as pl


@operation(returns=('scalar',), recipe='never')
def is_number(value):
    """Return whether one scalar parses as a finite number; null, NaN and infinity are false."""
    if isinstance(value, (list, tuple, dict, pl.DataFrame, pl.Series, pl.LazyFrame)):
        return False
    try:
        return pl.Series("value", [value]).to_frame().select(pl.col("value").cast(pl.Float64, strict=False).is_finite().fill_null(False)).item()
    except (TypeError, ValueError, pl.exceptions.PolarsError):
        return False
