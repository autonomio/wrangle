from .._core import operation
import string
import polars as pl
from .._core import reject_destructive, WrangleError, finish, frame, require_columns
from ._expressions import names_available


@operation(returns=('table',), recipe='yes')
def col_to_split(data, col, col_names=None, sep=" ", destructive=False):
    """Replace a string column with equal-length literal-delimiter components.

    All observed rows must split into the same number of fields; missing rows
    yield missing fields. Explicit names define width for all-missing input.
    Default names use col_a...col_z, then col_27 onwards. Source order stays.
    """
    reject_destructive(destructive)
    require_columns(data, [col])
    if not isinstance(sep, str) or not sep:
        raise WrangleError("INVALID_PARAMETER", "sep must be a nonempty literal delimiter.")
    dtype = frame(data).collect_schema()[col]
    if dtype != pl.String:
        raise WrangleError("NON_STRING_COLUMN", "Split requires a String column.")
    split = pl.col(col).str.split(sep)
    widths = frame(data).select(split.list.len().drop_nulls().unique()).collect().to_series().to_list()
    if len(widths) > 1:
        raise WrangleError("INCONSISTENT_FIELDS", "Observed rows split into different numbers of fields.")
    if widths:
        width = widths[0]
    elif col_names is not None:
        width = len(col_names)
    else:
        raise WrangleError("NO_OBSERVATIONS", "All-missing input requires explicit col_names.")
    names = list(col_names) if col_names is not None else [f"{col}_{string.ascii_lowercase[i] if i < 26 else i + 1}" for i in range(width)]
    if len(names) != width or width < 1:
        raise WrangleError("INVALID_COLUMN_NAMES", "Provide one output name per split field.")
    names_available(data, names, removed=[col])
    return finish(frame(data).select(pl.exclude(col), *(split.list.get(i, null_on_oob=True).alias(name) for i, name in enumerate(names))), data)
