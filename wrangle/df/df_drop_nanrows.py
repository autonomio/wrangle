from .._core import operation
from .._core import reject_destructive
import polars as pl
from .._core import frame, finish
from ._expressions import missing


@operation(returns=('table',), recipe='yes')
def df_drop_nanrows(data, destructive=False):
    """Exclude rows containing any null or floating-point NaN."""
    reject_destructive(destructive)
    schema = frame(data).collect_schema()
    if not schema:
        return finish(frame(data), data)
    return finish(frame(data).filter(~pl.any_horizontal(missing(name, dtype) for name, dtype in schema.items())), data)
