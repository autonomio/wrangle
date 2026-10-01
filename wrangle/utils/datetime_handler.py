"""Schema-based datetime handling without guessed parsing or row reordering."""
from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, WrangleError


@operation(returns=('sequence',), recipe='never')
def datetime_detector(data, datetime_mode="retain"):
    """Return Date/Datetime column names from the declared schema; strings are never guessed as timestamps."""
    return [name for name, dtype in frame(data).collect_schema().items() if dtype == pl.Date or isinstance(dtype, pl.Datetime)]


@operation(returns=('table',), recipe='yes')
def datetime_handler(data, datetime_mode="pass"):
    """pass preserves order; retain moves temporal columns first; drop explicitly removes them; sequence uses dense chronological ranks."""
    if datetime_mode not in {"pass", "retain", "drop", "sequence"}:
        raise WrangleError("INVALID_MODE", "Use pass, retain, drop, or sequence.")
    plan = frame(data)
    names = datetime_detector(data)
    if datetime_mode == "drop":
        plan = plan.drop(names)
    elif datetime_mode == "retain":
        plan = plan.select(names + [c for c in plan.collect_schema().names() if c not in names])
    elif datetime_mode == "sequence":
        for name in names:
            plan = frame(datetime_to_sequence(plan, name))
    return finish(plan, data)


@operation(returns=('table',), recipe='yes')
def datetime_to_sequence(data, col):
    """Replace a temporal column by its zero-based dense chronological rank; equal timestamps share a rank and null stays null."""
    require_columns(data, col)
    if col not in datetime_detector(data):
        raise WrangleError("INVALID_DTYPE", "Sequence conversion requires a declared Date/Datetime column.")
    return finish(frame(data).with_columns((pl.col(col).rank(method="dense") - 1).alias(col)), data)
