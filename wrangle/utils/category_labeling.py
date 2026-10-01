"""Explicit legacy string-category preparation, with no automatic column deletion."""
from .._core import operation
import polars as pl
from .._core import frame, finish, WrangleError, require_columns
from .wrangler_utils import string_contains_to_binary


@operation(returns=('table',), recipe='yes')
def to_category_labels(data, max_categories, starts_with_col=None, col_that_contains=None, col_contains_strings=(), *, categories=None):
    """Encode strings only with a declared categories mapping; explicit starts-with and substring selections are retained. Never drop high-cardinality fields."""
    plan = frame(data)
    categories = categories or {}
    if not isinstance(max_categories, int) or isinstance(max_categories, bool) or max_categories < 1:
        raise WrangleError("INVALID_ARGUMENT", "Declare a positive maximum category count.")
    if starts_with_col is not None:
        require_columns(plan, starts_with_col)
        plan = plan.with_columns(pl.col(starts_with_col).cast(pl.String).str.slice(0, 1))
    if col_that_contains is not None:
        indicators = string_contains_to_binary(plan, col_that_contains, col_contains_strings)
        names = indicators.collect_schema().names()
        if set(names) & set(plan.collect_schema().names()):
            raise WrangleError("DUPLICATE_COLUMNS", "Indicator names collide with existing fields.")
        # A row-index join preserves identity, including null source values.
        scratch = "__wr_row"
        available = set(plan.collect_schema().names()) | set(names)
        while scratch in available:
            scratch += "_"
        plan = plan.with_row_index(scratch).join(indicators.with_row_index(scratch), on=scratch, how="left", validate="1:1", maintain_order="left").drop(scratch)
    for name, declared in categories.items():
        require_columns(plan, name)
        if len(declared) > max_categories or len(set(declared)) != len(declared):
            raise WrangleError("INVALID_CATEGORIES", "Declared categories must be unique and within the cap.")
        invalid = plan.filter(pl.col(name).is_not_null() & ~pl.col(name).is_in(declared)).select(pl.len()).collect().item()
        if invalid:
            raise WrangleError("UNKNOWN_CATEGORY", "Observed values are absent from the declared category mapping.", {"column": name, "rows": invalid})
        plan = plan.with_columns(pl.col(name).replace_strict(declared, list(range(len(declared))), default=None, return_dtype=pl.UInt32))
    return finish(plan, data)
