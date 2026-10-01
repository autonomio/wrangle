from .._core import operation
import polars as pl
from .._core import frame, finish, WrangleError
from ._expressions import clean, fraction, numeric_only, require_float_exactness


@operation(returns=('table',), recipe='yes')
def df_add_scorecol(data, metric, win_threshold=.66, lose_threshold=.33, win_points=2, lose_points=-2, tie_points=0, score_col_name='score'):
    """Score a metric using its quantiles; missing metrics retain null scores."""
    fraction(win_threshold, "win_threshold")
    fraction(lose_threshold, "lose_threshold")
    if lose_threshold > win_threshold:
        raise WrangleError("INVALID_OPTION", "lose_threshold must not exceed win_threshold.")
    if score_col_name in frame(data).collect_schema():
        raise WrangleError("COLUMN_COLLISION", "Score output column already exists.", {"column": score_col_name})
    schema = numeric_only(data, [metric])
    require_float_exactness(data, [metric])
    value = clean(metric, schema[metric])
    score = pl.when(value.is_null()).then(None).when(value >= value.quantile(win_threshold, interpolation="linear")).then(win_points).when(value <= value.quantile(lose_threshold, interpolation="linear")).then(lose_points).otherwise(tie_points).alias(score_col_name)
    return finish(frame(data).with_columns(score), data)
