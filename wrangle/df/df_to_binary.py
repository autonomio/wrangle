from .._core import operation
from ..col.col_to_binary import col_to_binary


@operation(returns=('table',), recipe='yes')
def df_to_binary(data, y, destructive=False, *, func='median'):
    """Threshold a selected column through the native column preparation operation."""
    return col_to_binary(data, y, func=func, destructive=destructive)
