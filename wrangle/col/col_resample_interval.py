from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, require_columns
from ._expressions import clean, numeric, seed_value


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_resample_interval(data, x, dt_col, mode="first", freq=60, *, seed=0):
    """Aggregate observed timestamps into fixed-minute, right-labeled intervals.

    Input must contain Date/Datetime timestamps; missing timestamps raise.
    Named periods: quarter=6h, half=12h, full=24h, week=7d, month=30d,
    year=365d. Month/year are fixed durations, not calendar periods.
    Empty intervals are omitted. first/last follow timestamp order, with source
    order breaking ties. mode/freq choose the smallest tied mode.
    """
    require_columns(data, [x, dt_col])
    if x == dt_col:
        raise WrangleError("INVALID_PARAMETER", "Value and timestamp columns must differ.")
    seed_value(seed)
    modes = {"median", "mean", "mode", "first", "last", "std", "max", "min", "sum", "random", "freq"}
    if mode not in modes:
        raise WrangleError("INVALID_MODE", f"mode must be one of {sorted(modes)!r}.")
    aliases = {"quarter": 360, "half": 720, "full": 1440, "week": 10080, "month": 43200, "year": 525600}
    minutes = aliases.get(freq, freq) if isinstance(freq, str) else freq
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0:
        raise WrangleError("INVALID_PARAMETER", "freq must be positive integer minutes or a named fixed period.")
    dtype = frame(data).collect_schema()[dt_col]
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        raise WrangleError("INVALID_TIMESTAMP", "Timestamp column must have Date or Datetime dtype.")
    if frame(data).select(pl.col(dt_col).is_null().any()).collect().item():
        raise WrangleError("MISSING_TIMESTAMP", "Every observation needs a timestamp for resampling.")
    value = numeric(data, x, float64=mode != "sum") if mode in {"median", "mean", "std", "sum"} else clean(data, x)
    if mode in {"mode", "freq"}:
        aggregate = value.drop_nulls().mode().sort().first()
    elif mode == "random":
        aggregate = value.drop_nulls().shuffle(seed=seed).first()
    elif mode == "sum":
        total = value.cast(pl.Int128).sum() if frame(data).collect_schema()[x].is_integer() else value.sum()
        aggregate = pl.when(value.count() > 0).then(total).otherwise(None)
    elif mode == "std":
        aggregate = value.std(ddof=1)
    elif mode == "first":
        aggregate = value.drop_nulls().first()
    elif mode == "last":
        aggregate = value.drop_nulls().last()
    else:
        aggregate = getattr(value, mode)()
    plan = frame(data).with_columns(pl.col(dt_col).cast(pl.Datetime) if dtype == pl.Date else pl.col(dt_col)).sort(
        dt_col, maintain_order=True
    ).group_by_dynamic(dt_col, every=f"{minutes}m", closed="left", label="right").agg(aggregate.alias(x))
    return finish(plan, data)
