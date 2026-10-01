from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish, columns, require_columns, WrangleError
from ._expressions import names


@operation(returns=('table',), recipe='yes')
def df_rename_cols(data, exclude=None, prefix='C', destructive=False):
    """Assign sequential names while preserving excluded columns and order."""
    reject_destructive(destructive)
    excluded = names(exclude)
    require_columns(data, excluded)
    if not isinstance(prefix, str):
        raise WrangleError("INVALID_OPTION", "prefix must be a string.")
    selected = [name for name in columns(data) if name not in excluded]
    mapping = {name: f"{prefix}{i}" for i, name in enumerate(selected)}
    output = [mapping.get(name, name) for name in columns(data)]
    if len(set(output)) != len(output):
        raise WrangleError("COLUMN_COLLISION", "Generated names collide with excluded columns.")
    return finish(frame(data).rename(mapping), data)
