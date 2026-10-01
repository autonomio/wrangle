from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean, numeric


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_corr_category(data, col, outcome, rounding=2, warning_threshold=40):
    """Report binary outcome percentage, observed count, and low-sample flag per category.

    Missing categories form their own group; missing outcomes are excluded from n.
    Outcome values must be 0 or 1. Counts and warnings remain separate from labels.
    """
    require_columns(data, [col, outcome])
    if {col, outcome} & {"n", "low_sample"}:
        raise WrangleError("COLUMN_COLLISION", "Input names conflict with n or low_sample output names.")
    if col == outcome:
        raise WrangleError("INVALID_PARAMETER", "Category and outcome must be different columns.")
    if isinstance(rounding, bool) or not isinstance(rounding, int) or not 0 <= rounding <= 15:
        raise WrangleError("INVALID_PARAMETER", "rounding must be an integer between 0 and 15.")
    if isinstance(warning_threshold, bool) or not isinstance(warning_threshold, int) or warning_threshold < 0:
        raise WrangleError("INVALID_PARAMETER", "warning_threshold must be a nonnegative integer.")
    dtype = frame(data).collect_schema()[outcome]
    value = clean(data, outcome).cast(pl.Float64) if dtype == pl.Boolean else numeric(data, outcome)
    if frame(data).select((value.is_not_null() & ~value.is_in([0, 1])).any()).collect().item():
        raise WrangleError("NON_BINARY_OUTCOME", f"Column {outcome!r} must contain only 0, 1, or missing values.")
    plan = frame(data).with_columns(clean(data, col)).group_by(col, maintain_order=True).agg(
        (value.mean() * 100).round(rounding).alias(outcome), value.count().alias("n")
    ).with_columns((pl.col("n") < warning_threshold).alias("low_sample"))
    return finish(plan, data)
