from .._core import operation
import re
from .._core import frame, finish, columns, require_columns, WrangleError
from ._expressions import names


@operation(returns=('dictionary',), recipe='never')
def df_to_dfs(data, groupers, y):
    """Split columns by regex grouper, remove its literal prefix, and retain the target."""
    require_columns(data, [y])
    result = {}
    for label in names(groupers):
        try:
            selected = [name for name in columns(data) if name != y and re.search(label, name)]
        except re.error as exc:
            raise WrangleError("INVALID_OPTION", "Invalid grouper regex.", {"grouper": label}) from exc
        output = [name.replace(label, '') for name in selected]
        if any(not name for name in output) or len(set(output + [y])) != len(output) + 1:
            raise WrangleError("COLUMN_COLLISION", "Removing the grouper creates duplicate or empty column names.", {"grouper": label})
        plan = frame(data).select(selected + [y]).rename(dict(zip(selected, output)))
        result[label] = finish(plan, data)
    return result
