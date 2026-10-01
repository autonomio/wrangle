from .._core import operation
from .._core import frame, columns
from ._expressions import positive_n


@operation(returns=('dictionary',), recipe='never', aggregates=True)
def df_print_values(data, n=5):
    """Return most frequent values by column; agent-visible data replaces print side effects."""
    positive_n(n)
    return {name: frame(data).group_by(name, maintain_order=True).len().sort('len', descending=True, maintain_order=True).head(n).collect() for name in columns(data)}
