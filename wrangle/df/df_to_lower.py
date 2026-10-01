from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns
from ._expressions import names


@operation(returns=('table',), recipe='yes')
def df_to_lower(data, cols=None):
    """Lowercase selected string columns; preserve numeric and null values."""
    schema = frame(data).collect_schema()
    selected = names(cols, default=schema)
    require_columns(data, selected)
    return finish(frame(data).with_columns(pl.col(name).str.to_lowercase() for name in selected if schema[name] == pl.String), data)
