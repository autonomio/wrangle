from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean


def _replacement(value, dtype, name):
    """Validate a scalar fill against the declared field without losing precision."""
    if value is None:
        return pl.lit(None, dtype=dtype)
    if dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        value = str(value)
    literal = pl.lit(value)
    source_dtype = pl.select(literal).dtypes[0]
    if dtype == pl.Null:
        return literal
    if dtype == pl.Boolean and not (isinstance(value, bool) or isinstance(value, int) and value in {0, 1}):
        raise WrangleError("INVALID_FILL_VALUE", f"Column {name!r} requires a boolean fill or 0/1.")
    if dtype.is_numeric() and not source_dtype.is_numeric():
        raise WrangleError("INVALID_FILL_VALUE", f"Column {name!r} requires a numeric fill.")
    if dtype.is_temporal() and dtype.base_type() != source_dtype.base_type():
        raise WrangleError("INVALID_FILL_VALUE", f"Column {name!r} requires a fill with the same temporal type.")
    typed = literal.cast(dtype, strict=True)
    try:
        unchanged = pl.select(literal.eq_missing(typed.cast(source_dtype, strict=True))).item()
    except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
        raise WrangleError("INVALID_FILL_VALUE", f"Fill is incompatible with column {name!r} of type {dtype}.") from error
    if not unchanged:
        raise WrangleError("LOSSY_FILL", f"Fill cannot be represented exactly in column {name!r} of type {dtype}.")
    return typed


@operation(returns=('table',), recipe='yes')
def col_fill_nan(data, cols, fill_with=0):
    """Fill null and floating NaN in selected columns, preserving the whole table.

    String/categorical fields receive the string representation of fill_with.
    Declared dtypes, categorical domains, rows, and column order are preserved;
    incompatible or lossy fills raise. Untyped Null fields infer the explicit
    replacement's dtype. fill_with=None preserves nulls and normalizes NaN.
    """
    names = [cols] if isinstance(cols, str) else list(cols)
    require_columns(data, names)
    schema = frame(data).collect_schema()
    expressions = [clean(data, name).fill_null(_replacement(fill_with, schema[name], name)).alias(name) for name in names]
    return finish(frame(data).with_columns(expressions), data)
