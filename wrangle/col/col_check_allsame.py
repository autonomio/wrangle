from .._core import operation
from .._core import frame, require_columns
from ._expressions import clean


@operation(returns=('scalar',), recipe='never')
def col_check_allsame(data, col):
    """Return whether a column has exactly one distinct value, counting missingness.

    Empty columns return False. Floating NaN and null represent one missing value.
    """
    require_columns(data, [col])
    return frame(data).select(clean(data, col).n_unique() == 1).collect().item()
