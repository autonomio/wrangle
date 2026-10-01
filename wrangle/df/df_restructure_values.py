from .._core import operation
import polars as pl
from .._core import frame, finish, columns, WrangleError, reject_destructive


@operation(returns=('table',), recipe='yes')
def df_restructure_values(data, structure='list', destructive=False):
    """Couple each value to its header: string lists, typed structs, or strings."""
    reject_destructive(destructive)
    if structure not in {"list", "str", "tuple", "dict"}:
        raise WrangleError("INVALID_OPTION", "structure must be list, str, tuple, or dict.")
    expressions = []
    for name in columns(data):
        if structure == 'list':
            expression = pl.concat_list(pl.lit(name), pl.col(name).cast(pl.String))
        elif structure == 'str':
            expression = pl.concat_str(pl.lit(name), pl.col(name).cast(pl.String), separator=' ')
        elif structure == 'dict':
            expression = pl.struct(pl.col(name).alias(name))
        else:
            expression = pl.struct(pl.lit(name).alias('column'), pl.col(name).alias('value'))
        expressions.append(expression.alias(name))
    return finish(frame(data).select(expressions), data)
