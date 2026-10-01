"""Bucketwise binary outcome percentages across named Polars tables."""
from .._core import operation
import polars as pl
from .._core import frame, require_columns, numeric_columns, WrangleError
from ..col.col_to_buckets import col_to_buckets
from ..col.col_corr_category import col_corr_category


@operation(returns=('table',), recipe='never', aggregates=True)
def dic_corr_perc(data, y, *, cuts=5, warning_threshold=40):
    """Return bucket label, outcome percentage, samples, low_sample, metric and metric_context; require a binary y and numeric predictors."""
    if not isinstance(data, dict):
        raise WrangleError("INVALID_INPUT", "Use a dictionary of named Polars tables.")
    outputs = []
    reserved = {"index", "samples", "metric", "metric_context", "low_sample"}
    if y in reserved or y.startswith("__wr_"):
        raise WrangleError("RESERVED_COLUMN", "Outcome name conflicts with percentage-report fields.")
    for label, table in data.items():
        require_columns(table, y)
        dtype = frame(table).collect_schema()[y]
        value = pl.col(y).cast(pl.UInt8) if dtype == pl.Boolean else pl.col(y)
        if dtype.is_float():
            value = value.fill_nan(None)
        try:
            invalid = frame(table).select((value.is_not_null() & ~value.is_in([0, 1])).any()).collect().item()
        except pl.exceptions.PolarsError as error:
            raise WrangleError("NON_BINARY_OUTCOME", "Outcome values must be zero, one, or missing.", {"column": y}) from error
        if invalid:
            raise WrangleError("NON_BINARY_OUTCOME", "Outcome values must be zero, one, or missing.", {"column": y})
        predictors = [name for name in numeric_columns(table) if name != y]
        for name in predictors:
            buckets = col_to_buckets(table, name, cuts=cuts)
            subset = frame(table).select(pl.col(y)).with_columns(buckets.rename("__wr_bucket"))
            result = frame(col_corr_category(subset, "__wr_bucket", y, warning_threshold=warning_threshold)).rename({"__wr_bucket": "index", "n": "samples"}).with_columns(pl.lit(name).alias("metric"), pl.lit(str(label)).alias("metric_context"))
            outputs.append(result)
    if not outputs:
        return pl.DataFrame(schema={"index": pl.String, y: pl.Float64, "samples": pl.UInt32, "low_sample": pl.Boolean, "metric": pl.String, "metric_context": pl.String})
    return pl.concat(outputs, how="vertical_relaxed").collect()
