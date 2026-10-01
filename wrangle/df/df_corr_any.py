from .._core import operation
import polars as pl
from .._core import frame, numeric_columns, WrangleError
from .df_to_numeric import df_to_numeric
from ._expressions import correlation as coefficient, correlation_expr, require_float_exactness


@operation(returns=('table',), recipe='yes', aggregates=True)
def df_corr_any(data, correlation='pearson'):
    """Return a labelled pairwise-complete matrix; native Kendall tau-b caps each coefficient at 10 million observation pairs."""
    if correlation not in {"pearson", "spearman", "kendall"}:
        raise WrangleError("INVALID_OPTION", "correlation must be pearson, spearman, or kendall.")
    converted = df_to_numeric(frame(data).collect())
    selected = numeric_columns(converted)
    # The explicit label replaces the implicit pandas index.
    label = "column"
    while label in selected:
        label = "_" + label
    result = {label: pl.Series(label, selected, dtype=pl.String)}
    if correlation in {"pearson", "spearman"} and selected:
        require_float_exactness(converted, selected)
        schema = converted.schema
        pairs = [(i, j) for i in range(len(selected)) for j in range(i, len(selected))]
        scalar = converted.lazy().select(correlation_expr(selected[i], selected[j], schema, correlation).alias(f"pair_{i}_{j}") for i, j in pairs).collect().row(0, named=True)
        for j, name in enumerate(selected):
            result[name] = pl.Series(name, [scalar[f"pair_{min(i, j)}_{max(i, j)}"] for i in range(len(selected))], dtype=pl.Float64)
    else:
        for name in selected:
            result[name] = pl.Series(name, [coefficient(converted, row, name, correlation) for row in selected], dtype=pl.Float64)
    return pl.DataFrame(result)
