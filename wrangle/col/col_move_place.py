from .._core import operation
from .._core import reject_destructive, WrangleError, columns, finish, frame, require_columns


@operation(returns=('table',), recipe='yes')
def col_move_place(data, col, position="first", destructive=False):
    """Move selected columns first or last in the supplied order, preserving rows."""
    reject_destructive(destructive)
    if position not in {"first", "last"}:
        raise WrangleError("INVALID_MODE", "position must be 'first' or 'last'.")
    selected = [col] if isinstance(col, str) else list(col)
    require_columns(data, selected)
    if len(set(selected)) != len(selected):
        raise WrangleError("INVALID_COLUMN_NAMES", "Selected columns must be unique.")
    remaining = [name for name in columns(data) if name not in selected]
    order = selected + remaining if position == "first" else remaining + selected
    return finish(frame(data).select(order), data)
