from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, reject_destructive
from ._expressions import clean, names, positive_n


@operation(returns=('table', 'tuple'), recipe='conditional')
def df_to_multiclass(data, max_uniques=30, ignore_y=None, destructive=False, *, return_mapping=False):
    """Encode low-cardinality columns in sorted order; missing=-1; optionally return maps."""
    reject_destructive(destructive)
    positive_n(max_uniques)
    ignored = names(ignore_y)
    require_columns(data, ignored)
    schema = frame(data).collect_schema()
    selected = [name for name in schema if name not in ignored]
    maps, expressions = {}, []
    for name in selected:
        values = frame(data).select(clean(name, schema[name]).unique().sort(nulls_last=True).alias("value")).collect()
        if values.height <= max_uniques:
            observed = values.drop_nulls().with_row_index("code").with_columns(pl.col("code").cast(pl.Int64)).select("value", "code")
            maps[name] = observed
            expressions.append(clean(name, schema[name]).replace_strict(observed["value"], observed["code"], default=-1, return_dtype=pl.Int64).alias(name))
    result = finish(frame(data).with_columns(expressions), data)
    return (result, maps) if return_mapping else result
