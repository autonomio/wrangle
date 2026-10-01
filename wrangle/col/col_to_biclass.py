from .._core import operation
import polars as pl
from .._core import reject_destructive, WrangleError, finish, frame, require_columns
from ._expressions import clean, levels


@operation(returns=('table',), recipe='yes')
def col_to_biclass(data, col, true_value, destructive=False):
    """Map the named true class to 1 and the other observed class to 0.

    More than two observed classes or an absent true class raises. Missing values
    remain null. The supplied table is never modified.
    """
    reject_destructive(destructive)
    require_columns(data, [col])
    categories = levels(data, col)
    if len(categories) > 2:
        raise WrangleError("NON_BINARY_CLASSES", "Biclass encoding requires at most two observed classes.")
    if true_value not in categories:
        raise WrangleError("UNKNOWN_TRUE_CLASS", "true_value is not an observed class.")
    return finish(frame(data).with_columns((clean(data, col) == pl.lit(true_value)).cast(pl.Int8).alias(col)), data)
