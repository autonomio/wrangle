from .._core import operation
import polars as pl
from .._core import frame, finish
from ._expressions import missing


@operation(returns=('table',), recipe='yes', aggregates=True)
def df_find_nan(data):
    """Report null/NaN counts, missing fraction and complete fraction by column."""
    schema = frame(data).collect_schema()
    if not schema:
        out = pl.DataFrame(schema={"column": pl.String, "missing": pl.UInt32, "missing_fraction": pl.Float64, "no_nans": pl.Boolean, "quality": pl.Float64}).lazy()
    else:
        out = pl.concat([frame(data).select(pl.lit(name).alias("column"), missing(name, dtype).sum().alias("missing"), missing(name, dtype).mean().alias("missing_fraction")).with_columns((pl.col("missing") == 0).alias("no_nans"), (1 - pl.col("missing_fraction")).alias("quality")) for name, dtype in schema.items()])
    return finish(out, data)
