from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish
from ._expressions import missing, fraction


@operation(returns=('table',), recipe='yes')
def df_drop_nancols(data, threshold=0, destructive=False):
    """Keep columns whose missing fraction is at most threshold (0..1)."""
    reject_destructive(destructive)
    fraction(threshold)
    schema = frame(data).collect_schema()
    if not schema:
        return finish(frame(data), data)
    rates = frame(data).select(missing(name, dtype).mean().alias(name) for name, dtype in schema.items()).collect().row(0, named=True)
    return finish(frame(data).select(name for name, rate in rates.items() if rate is None or rate <= threshold), data)
