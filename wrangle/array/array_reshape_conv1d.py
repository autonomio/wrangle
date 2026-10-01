"""Native nested-list representation of Conv1D preparation."""
from .._core import operation
import polars as pl
from .._core import frame, finish, WrangleError


@operation(returns=('series', 'table'), recipe='conditional')
def array_reshape_conv1d(x):
    """Return tensor: one list of singleton feature lists per row; retain lazy input as a one-column lazy table."""
    plan = frame(x)
    names = plan.collect_schema().names()
    if not names:
        raise WrangleError("EMPTY_DATA", "At least one feature column is required.")
    schema = plan.collect_schema()
    if any(not dtype.is_numeric() for dtype in schema.dtypes()):
        raise WrangleError("NON_NUMERIC_COLUMN", "Tensor features must have declared numeric types; encode labels explicitly.")
    dtype = plan.select(pl.concat_list(names).alias("tensor")).collect_schema()["tensor"].inner
    checks = []
    for name in names:
        source = pl.col(name)
        restored = source.cast(dtype).cast(schema[name], strict=False)
        checks.append(source.is_not_null() & (restored.is_null() | (restored != source)))
    if plan.select(pl.any_horizontal(checks).any()).collect().item():
        raise WrangleError("LOSSY_CAST", "Combining tensor features would lose numeric precision; declare a common exact dtype first.")
    result = plan.select(pl.concat_list(names).list.eval(pl.concat_list(pl.element())).alias("tensor"))
    out = finish(result, x)
    return out if isinstance(x, pl.LazyFrame) else out.to_series()
