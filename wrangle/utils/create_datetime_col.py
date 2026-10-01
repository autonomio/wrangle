"""Calendar datetime ranges checked against the observation count."""
from .._core import operation
from datetime import datetime
import polars as pl
from .._core import frame, WrangleError

_INTERVALS = {"year": "1y", "month": "1mo", "week": "1w", "day": "1d", "hour": "1h", "minute": "1m", "second": "1s"}


@operation(returns=('series',), recipe='never')
def create_datetime_col(data, start, end, freq):
    """Return an inclusive datetime Series; reject ranges whose length differs from data. Use ISO start/end and Polars interval syntax."""
    try:
        start = datetime.fromisoformat(start) if isinstance(start, str) else start
        end = datetime.fromisoformat(end) if isinstance(end, str) else end
        result = pl.datetime_range(start, end, interval=_INTERVALS.get(freq, freq), eager=True).alias("timestamp")
    except (ValueError, TypeError, pl.exceptions.PolarsError) as error:
        raise WrangleError("INVALID_DATETIME", str(error)) from error
    count = frame(data).select(pl.len()).collect().item()
    if len(result) != count:
        raise WrangleError("ROW_ALIGNMENT", "Datetime range must match the observation count.", {"rows": count, "timestamps": len(result)})
    return result
