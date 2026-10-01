"""Task declarations checked against observed data; scientific meaning is never guessed."""
from .._core import operation
import polars as pl
from .._core import frame, WrangleError


@operation(returns=('tuple',), recipe='never')
def array_detect_task(y, *, task=None):
    """Return (type, range, format) for an explicitly declared binary, categorical, continuous or multilabel task."""
    if task not in {"binary", "categorical", "continuous", "multilabel"}:
        raise WrangleError("TASK_DECLARATION_REQUIRED", "Declare task='binary', 'categorical', 'continuous', or 'multilabel'; values cannot determine scientific meaning.")
    table = frame(y).collect()
    if table.is_empty() or table.null_count().select(pl.sum_horizontal(pl.all())).item():
        raise WrangleError("INVALID_LABELS", "Task labels must be nonempty and non-null.")
    floats = [name for name, dtype in table.schema.items() if dtype.is_float()]
    if floats and table.select(pl.any_horizontal([~pl.col(c).is_finite() for c in floats]).any()).item():
        raise WrangleError("INVALID_LABELS", "Task labels cannot contain NaN or infinite values.")
    if task in {"binary", "multilabel"}:
        if any(dtype != pl.Boolean and not dtype.is_numeric() for dtype in table.dtypes):
            raise WrangleError("INVALID_LABELS", "Binary indicators must have numeric or boolean types.")
        invalid = table.select(pl.any_horizontal([pl.lit(False) if table.schema[c] == pl.Boolean else ~pl.col(c).is_in([0, 1]) for c in table.columns]).any()).item()
        if invalid or (task == "binary" and table.width != 1):
            raise WrangleError("INVALID_LABELS", "Binary indicators must be zero or one; binary tasks need one column.")
        return ("binary" if task == "binary" else "category", table.width, "single" if task == "binary" else "multilabel")
    if table.width != 1:
        raise WrangleError("INVALID_LABELS", "This declared task requires one target column.")
    value = pl.col(table.columns[0])
    if task == "categorical":
        return "category", table.select(value.n_unique()).item(), "single"
    if not table.to_series().dtype.is_numeric() or table.select((~value.is_finite()).any()).item():
        raise WrangleError("INVALID_LABELS", "Continuous targets must contain finite numeric values.")
    return "continuous", table.select(value.max() - value.min()).item(), "single"
