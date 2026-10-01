"""Native CSV scanning; absent source rows are never fabricated."""
from .._core import operation
from pathlib import Path
import polars as pl
from .._core import require_columns, WrangleError
from ..array._native import positive_int

_TYPES = {"float64": pl.Float64, "float32": pl.Float32, "int64": pl.Int64, "int32": pl.Int32, "str": pl.String, "string": pl.String}


@operation(returns=('table',), recipe='never')
def read_large_csv(file_path, n, cols=None, chunks=None, dtype=None):
    """Return a LazyFrame containing up to n real rows. dtype=None preserves CSV text, including identifiers and leading zeros; explicit casts are strict. chunks is a retained validation-only argument."""
    positive_int(n, "n", allow_zero=True)
    if chunks is not None:
        positive_int(chunks, "chunks")
    if "://" in str(file_path):
        raise WrangleError("LOCAL_SOURCE_REQUIRED", "CSV input must be an explicit local file; download external sources separately.")
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise WrangleError("SOURCE_NOT_FOUND", "CSV source does not exist.", {"path": str(path)})
    try:
        plan = pl.scan_csv(path, infer_schema=False).head(n)
        plan.collect_schema()
    except pl.exceptions.PolarsError as error:
        raise WrangleError("SOURCE_PARSE_ERROR", "CSV header/schema cannot be read.", {"path": str(path)}) from error
    if cols is not None:
        require_columns(plan, cols)
        plan = plan.select(cols)
    if dtype is not None:
        dtype = _TYPES.get(dtype, dtype) if isinstance(dtype, str) else dtype
        try:
            plan = plan.select(pl.all().cast(dtype, strict=True))
            plan.collect_schema()
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise WrangleError("INVALID_DTYPE", "Declare a supported Polars datatype.") from error
    return plan
