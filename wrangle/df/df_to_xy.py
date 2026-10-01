from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns


@operation(returns=('tuple',), recipe='never')
def df_to_xy(data, y_col):
    """Separate features and target: eager target Series, lazy target one-column plan."""
    require_columns(data, [y_col])
    x = finish(frame(data).drop(y_col), data)
    y = frame(data).select(y_col)
    return x, y if isinstance(data, pl.LazyFrame) else y.collect().to_series()
