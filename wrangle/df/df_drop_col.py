from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish, require_columns
from ._expressions import names


@operation(returns=('table',), recipe='yes')
def df_drop_col(data, cols, destructive=False):
    """Remove declared columns; unknown columns fail explicitly."""
    reject_destructive(destructive)
    cols = names(cols)
    require_columns(data, cols)
    return finish(frame(data).drop(cols), data)
