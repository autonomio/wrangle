from .._core import operation
import polars as pl
from .._core import frame, finish, WrangleError
from ._expressions import require_native


@operation(returns=('table',), recipe='never')
def df_parallelize_process(data, func, threads=16):
    """Execute native expressions; Polars manages parallelism. Python callbacks are rejected."""
    expressions = [func] if isinstance(func, pl.Expr) else func
    if not isinstance(expressions, (list, tuple)) or not all(isinstance(expr, pl.Expr) for expr in expressions):
        raise WrangleError("UNSUPPORTED_CALLBACK", "Pass a Polars expression or list of expressions; arbitrary Python callbacks cannot run inside Wrangle plans.")
    require_native(expressions)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise WrangleError("INVALID_OPTION", "threads must be a positive integer; Polars manages its thread pool.")
    return finish(frame(data).with_columns(expressions), data)
