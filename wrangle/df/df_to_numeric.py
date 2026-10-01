from .._core import operation
from .._core import reject_destructive
import polars as pl
from .._core import frame, finish


@operation(returns=('table',), recipe='yes')
def df_to_numeric(data, destructive=False):
    """Cast fully numeric string columns; leave mixed strings and nulls intact."""
    reject_destructive(destructive)
    schema = frame(data).collect_schema()
    selected = [name for name, dtype in schema.items() if dtype == pl.String]
    if not selected:
        return finish(frame(data), data)
    tests = []
    for name in selected:
        value = pl.col(name).str.strip_chars()
        tests.extend([value.is_not_null().any().alias(f"{name}__observed"), (value.is_not_null() & value.cast(pl.Float64, strict=False).is_null()).any().alias(f"{name}__bad"), (value.is_not_null() & value.cast(pl.Int64, strict=False).is_null()).any().alias(f"{name}__float")])
    results = frame(data).select(tests).collect().row(0, named=True)
    expressions = [pl.col(name).str.strip_chars().cast(pl.Float64 if results[f"{name}__float"] else pl.Int64).alias(name) for name in selected if results[f"{name}__observed"] and not results[f"{name}__bad"]]
    return finish(frame(data).with_columns(expressions), data)
