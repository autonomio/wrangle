from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, WrangleError, reject_destructive
from ._expressions import clean, names, positive_n


@operation(returns=('table', 'tuple'), recipe='conditional')
def df_to_multilabel(data, max_uniques=30, ignore_y=None, destructive=False, *, return_mapping=False):
    """One-hot low-cardinality columns as name__code, with an explicit missing category."""
    reject_destructive(destructive)
    positive_n(max_uniques)
    ignored = names(ignore_y)
    require_columns(data, ignored)
    schema = frame(data).collect_schema()
    maps, expressions = {}, []
    generated = set()
    for name, dtype in schema.items():
        if name in ignored:
            expressions.append(pl.col(name))
            continue
        values = frame(data).select(clean(name, dtype).unique().sort(nulls_last=True).alias("value")).collect()
        if values.height > max_uniques:
            expressions.append(pl.col(name))
            continue
        output_names = [f"{name}__{i}" for i in range(values.height)]
        for output in output_names:
            if output in schema or output in generated:
                raise WrangleError("COLUMN_COLLISION", "Encoded column names collide with existing names.", {"column": output})
            generated.add(output)
        maps[name] = values.with_columns(pl.Series("column", output_names, dtype=pl.String))
        value = clean(name, dtype)
        for category, output in zip(values["value"], output_names):
            expression = value.is_null() if category is None else value.eq(pl.lit(category)).fill_null(False)
            expressions.append(expression.cast(pl.UInt8).alias(output))
    result = finish(frame(data).select(expressions), data)
    return (result, maps) if return_mapping else result
