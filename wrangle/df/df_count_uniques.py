from .._core import operation
import polars as pl
from .._core import frame, finish, columns


@operation(returns=('table',), recipe='yes', aggregates=True)
def df_count_uniques(data):
    """Return one row of distinct counts, including null as a value."""
    return finish(frame(data).select(pl.col(name).n_unique() for name in columns(data)), data)
