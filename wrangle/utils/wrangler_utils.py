"""Native legacy preparation helpers with explicit selections."""
from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, WrangleError
from .value_starts_with import value_starts_with


@operation(returns=('scalar',), recipe='never')
def max_category(data, max_categories):
    """Resolve an explicitly requested category cap; auto is floor(n/50), with a minimum of one."""
    count = frame(data).select(pl.len()).collect().item()
    if max_categories == "auto":
        return max(1, count // 50)
    if max_categories == "max" or max_categories is None:
        return count + 1
    if isinstance(max_categories, int) and not isinstance(max_categories, bool) and max_categories >= 1:
        return max_categories
    raise WrangleError("INVALID_ARGUMENT", "Declare a positive category cap, auto, or max.")


@operation(returns=('table',), recipe='yes')
def filling_nans(data, fill_columns, fill_with):
    """Fill explicitly selected fields using the native column helper."""
    from ..col.col_fill_nan import col_fill_nan
    for name in ([fill_columns] if isinstance(fill_columns, str) else fill_columns):
        data = col_fill_nan(data, name, fill_with)
    return data


@operation(returns=('table',), recipe='yes')
def imputing_nans(data, impute_columns, impute_mode):
    """Impute explicitly selected fields using the native column helper."""
    from ..df.df_impute_nan import df_impute_nan
    names = [impute_columns] if isinstance(impute_columns, str) else list(impute_columns)
    return df_impute_nan(data, cols=names, impute_mode=impute_mode)


@operation(returns=('series', 'table'), recipe='conditional')
def starts_with_output(data, col):
    """Extract first characters without inferred category codes."""
    return value_starts_with(data, col)


@operation(returns=('table',), recipe='yes')
def string_contains_to_binary(data, col_that_contains, col_contains_strings):
    """Return literal substring indicators; null stays null. Multiple source fields prefix indicator names."""
    names = [col_that_contains] if isinstance(col_that_contains, str) else list(col_that_contains)
    patterns = [col_contains_strings] if isinstance(col_contains_strings, str) else list(col_contains_strings)
    require_columns(data, names)
    if not patterns or len(patterns) != len(set(patterns)):
        raise WrangleError("INVALID_ARGUMENT", "Provide distinct substring indicators.")
    expressions = [pl.col(c).cast(pl.String).str.contains(pattern, literal=True).cast(pl.UInt8).alias(pattern if len(names) == 1 else f"{c}__{pattern}") for c in names for pattern in patterns]
    return finish(frame(data).select(expressions), data)
