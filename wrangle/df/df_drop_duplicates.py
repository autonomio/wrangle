from .._core import operation
from .._core import frame, finish, require_columns


@operation(returns=('table',), recipe='yes')
def df_drop_duplicates(data, subset=None):
    """Keep the first observation for each duplicate key, in source order."""
    if subset is not None:
        require_columns(data, subset)
    return finish(frame(data).unique(subset=subset, keep="first", maintain_order=True), data)
