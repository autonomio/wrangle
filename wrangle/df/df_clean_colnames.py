from .._core import operation
from .._core import reject_destructive
import re
from .._core import frame, finish, columns, WrangleError


@operation(returns=('table',), recipe='yes')
def df_clean_colnames(data, destructive=False):
    """Normalize names to lowercase ASCII identifiers; reject collisions."""
    reject_destructive(destructive)
    old = columns(data)
    new = [re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower() for name in old]
    if any(not name for name in new) or len(set(new)) != len(new):
        raise WrangleError("COLUMN_COLLISION", "Cleaning would create empty or duplicate names.", {"names": new})
    return finish(frame(data).rename(dict(zip(old, new))), data)
