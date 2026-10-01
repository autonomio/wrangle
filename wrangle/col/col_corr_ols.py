from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, reject_destructive, require_columns
from ._expressions import clean, numeric, temporary


@operation(returns=('table', 'tuple'), recipe='conditional')
def col_corr_ols(data, x, y, destructive=False, *, return_mapping=False):
    """Encode categories by their OLS coefficient, equal to the observed group mean.

    This is dummy-only OLS without an intercept. Codes start at 1; equal means
    receive equal codes. Missing x stays missing. Missing y is excluded when
    fitting; a category with no observed y raises. Fit on training data only.
    return_mapping=True returns (encoded_table, mapping), with mapping columns
    category/coefficient/code. destructive=True raises IMMUTABLE_INPUT.
    """
    reject_destructive(destructive)
    require_columns(data, [x, y])
    if x == y:
        raise WrangleError("INVALID_PARAMETER", "x and y must be different columns.")
    value = numeric(data, y, float64=True)
    mapping = frame(data).select(
        clean(data, x).alias("category"), value.alias("observed")
    ).filter(pl.col("category").is_not_null()).group_by("category").agg(
        pl.col("observed").mean().alias("coefficient")
    ).sort("category").with_columns(
        pl.col("coefficient").rank("dense").cast(pl.Int64).alias("code")
    ).collect()
    if mapping["coefficient"].null_count():
        raise WrangleError("NO_OBSERVATIONS", "Every category needs at least one observed outcome.")
    code_name = temporary(data, "__wrangle_code")
    plan = frame(data).with_columns(clean(data, x)).join(
        mapping.select("category", pl.col("code").alias(code_name)).lazy(),
        left_on=x, right_on="category", how="left", validate="m:1", maintain_order="left"
    ).with_columns(pl.col(code_name).alias(x)).drop(code_name)
    result = finish(plan, data)
    return (result, mapping) if return_mapping else result
