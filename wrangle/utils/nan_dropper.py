"""Explicit completeness thresholds and listwise missing-value exclusion."""
from .._core import operation
import polars as pl
from .._core import frame, finish, missing, WrangleError


@operation(returns=('table',), recipe='yes')
def nan_dropper(data, treshold=0.9):
    """Retain columns meeting the declared minimum observed fraction; then drop rows missing any retained field."""
    if not 0 <= treshold <= 1:
        raise WrangleError("INVALID_THRESHOLD", "Completeness threshold must be in [0, 1].")
    plan = frame(data)
    schema = plan.collect_schema()
    stats = plan.select([(1 - missing(pl.col(c), dtype).mean()).alias(c) for c, dtype in schema.items()]).collect()
    names = [c for c in schema.names() if stats[c][0] is not None and stats[c][0] >= treshold]
    if not names:
        raise WrangleError("EMPTY_SCHEMA", "No columns meet the declared completeness threshold.")
    out = plan.select(names).filter(~pl.any_horizontal([missing(pl.col(c), schema[c]) for c in names]))
    return finish(out, data)
