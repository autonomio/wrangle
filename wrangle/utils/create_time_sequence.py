"""Calendar-aware timestamp sequences via native offset expressions."""
from .._core import operation
from datetime import datetime
import polars as pl
from .._core import WrangleError
from ..array._native import positive_int

_INTERVALS = {"year": "y", "month": "mo", "week": "w", "day": "d", "hour": "h", "minute": "m", "second": "s"}


@operation(returns=('series',), recipe='never')
def create_time_sequence(periods, start_year, start_month, start_date=1, time_format="%m-%Y", period="month"):
    """Return a formatted Polars Series; start_date is honored and calendar months/years are anchored to the start."""
    positive_int(periods, "periods", allow_zero=True)
    if period not in _INTERVALS:
        raise WrangleError("INVALID_FREQUENCY", "Use year, month, week, day, hour, minute, or second.")
    try:
        start = datetime(start_year, start_month, start_date)
    except (ValueError, TypeError) as error:
        raise WrangleError("INVALID_DATETIME", str(error)) from error
    return pl.select(pl.int_range(0, periods).alias("offset")).select(pl.lit(start).dt.offset_by(pl.col("offset").cast(pl.String) + _INTERVALS[period]).dt.strftime(time_format).alias("timestamp")).to_series()
