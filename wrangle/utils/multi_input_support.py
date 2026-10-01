"""Unambiguous column selection for legacy preparation helpers."""
from .._core import operation
from .._core import frame, finish, require_columns, WrangleError


@operation(returns=('table',), recipe='never')
def multi_input_support(X, data):
    """Select a name, integer index, or a list of either; two integer indices are two columns, never an implicit range."""
    available = frame(data).collect_schema().names()
    selected = X if isinstance(X, (list, tuple)) else [X]
    if not selected:
        raise WrangleError("INVALID_INPUT", "Select at least one column.")
    names = []
    for value in selected:
        if isinstance(value, int) and not isinstance(value, bool):
            if not -len(available) <= value < len(available):
                raise WrangleError("UNKNOWN_COLUMN", "Column index is out of bounds.")
            names.append(available[value])
        elif isinstance(value, str):
            names.append(value)
        else:
            raise WrangleError("INVALID_INPUT", "Select columns by explicit names or integer positions.")
    if len(names) != len(set(names)):
        raise WrangleError("DUPLICATE_COLUMNS", "Select each column once.")
    require_columns(data, names)
    return finish(frame(data).select(names), data)
