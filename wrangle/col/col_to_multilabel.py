from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean, levels, names_available


@operation(returns=('table',), recipe='yes')
def col_to_multilabel(data, col, colnames=None, extended_colname=False, extended_separator="_"):
    """Replace one categorical column with deterministic one-hot columns.

    Categories sort by value. colnames, when provided, follows that exact order.
    Missing category yields null indicators. Existing column names cannot be
    overwritten. Input row order is preserved, with indicators appended.
    """
    require_columns(data, [col])
    categories = levels(data, col)
    names = [str(category) for category in categories] if colnames is None else list(colnames)
    if len(names) != len(categories):
        raise WrangleError("INVALID_COLUMN_NAMES", "colnames must provide exactly one name per observed category.")
    if not isinstance(extended_separator, str):
        raise WrangleError("INVALID_PARAMETER", "extended_separator must be a string.")
    if extended_colname:
        names = [col + extended_separator + str(name) for name in names]
    names_available(data, names, removed=[col])
    value = clean(data, col)
    expressions = [(value == pl.lit(category)).cast(pl.Int8).alias(name) for category, name in zip(categories, names)]
    if not categories:
        raise WrangleError("NO_OBSERVATIONS", "Cannot encode a column without observed categories.")
    plan = frame(data).select(pl.exclude(col), *expressions)
    return finish(plan, data)
