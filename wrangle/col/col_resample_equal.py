from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean, seed_value, temporary


@operation(returns=('table',), recipe='yes')
def col_resample_equal(data, col, sample_size, *, seed=0):
    """Sample sample_size observations without replacement from each category.

    The seed controls selection; selected rows retain source order and identity.
    Missing values form one category. Insufficient categories raise explicitly.
    """
    require_columns(data, [col])
    seed_value(seed)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise WrangleError("INVALID_PARAMETER", "sample_size must be a nonnegative integer.")
    counts = frame(data).with_columns(clean(data, col)).group_by(col).len().collect()
    if counts.filter(pl.col("len") < sample_size).height:
        raise WrangleError("INSUFFICIENT_SAMPLES", "At least one category has fewer rows than sample_size.")
    row = temporary(data)
    plan = frame(data).with_row_index(row).filter(
        pl.col(row).shuffle(seed=seed).rank("ordinal").over(clean(data, col)) <= sample_size
    ).drop(row)
    return finish(plan, data)
