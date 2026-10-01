from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish, numeric_columns, require_columns
from ._expressions import names, clean, domain


@operation(returns=('table',), recipe='yes')
def df_rescale_sqrt(data, retain_cols=None, destructive=False):
    """Square-root numeric columns; negative observed values fail explicitly."""
    reject_destructive(destructive)
    retained = names(retain_cols)
    require_columns(data, retained)
    selected = [name for name in numeric_columns(data) if name not in retained]
    domain(data, selected, 0)
    schema = frame(data).collect_schema()
    return finish(frame(data).with_columns(clean(name, schema[name]).sqrt().alias(name) for name in selected), data)
