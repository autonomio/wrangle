"""Native first-character extraction; nulls retain their missingness."""
from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns


@operation(returns=('series', 'table'), recipe='conditional')
def value_starts_with(data, col):
    """Return the first character as a Series (or one-column lazy table), without inferred categorical codes."""
    require_columns(data, col)
    result = finish(frame(data).select(pl.col(col).cast(pl.String).str.slice(0, 1).alias(col)), data)
    return result if isinstance(data, pl.LazyFrame) else result.to_series()


def _category_starts_with(data, col):
    return finish(frame(data).select(pl.col(col).cast(pl.String).str.slice(0, 1).unique().sort().alias(col)), data)
