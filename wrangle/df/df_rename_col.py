from .._core import operation
from .._core import reject_destructive
from .._core import frame, finish, require_columns, columns, WrangleError


@operation(returns=('table',), recipe='yes')
def df_rename_col(data, col, rename_to, destructive=False):
    """Rename one column without changing its position or values."""
    reject_destructive(destructive)
    require_columns(data, [col])
    if not isinstance(rename_to, str) or not rename_to:
        raise WrangleError("INVALID_OPTION", "rename_to must be a nonempty string.")
    if rename_to != col and rename_to in columns(data):
        raise WrangleError("COLUMN_COLLISION", "Destination column already exists.", {"column": rename_to})
    return finish(frame(data).rename({col: rename_to}), data)
