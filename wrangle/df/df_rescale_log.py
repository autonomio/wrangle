from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish, numeric_columns, require_columns
from ._expressions import names, clean, domain


@operation(returns=('table',), recipe='yes')
def df_rescale_log(data, retain_cols=None, destructive=False):
    """Apply log1p to numeric columns; values <= -1 fail explicitly."""
    reject_destructive(destructive)
    retained = names(retain_cols)
    require_columns(data, retained)
    selected = [name for name in numeric_columns(data) if name not in retained]
    domain(data, selected, -1, strict=True)
    schema = frame(data).collect_schema()
    return finish(frame(data).with_columns(clean(name, schema[name]).log1p().alias(name) for name in selected), data)
