from .._core import operation
import polars as pl
from .._core import frame, finish, WrangleError


@operation(returns=('table',), recipe='yes')
def df_fill_empty(data, fill_with):
    """Strip string columns and replace blank strings; preserve other dtypes."""
    if fill_with is not None and not isinstance(fill_with, str):
        raise WrangleError("INVALID_OPTION", "fill_with must be a string or None.")
    schema = frame(data).collect_schema()
    expressions = []
    for name, dtype in schema.items():
        if dtype == pl.String:
            value = pl.col(name).str.strip_chars()
            expressions.append(pl.when(value == "").then(pl.lit(fill_with)).otherwise(value).alias(name))
    return finish(frame(data).with_columns(expressions), data)
