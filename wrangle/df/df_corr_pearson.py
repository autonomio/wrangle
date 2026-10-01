from .._core import operation
import polars as pl
from .._core import frame
from ._expressions import correlation_expr, numeric_only, require_float_exactness


@operation(returns=('dictionary', 'tuple'), recipe='never', aggregates=True)
def df_corr_pearson(data, y, excluded_list=False):
    """Return {column: [pairwise-complete Pearson coefficient, distinct count]}."""
    schema = numeric_only(data, [y])
    require_float_exactness(data, [name for name, dtype in schema.items() if dtype.is_numeric()])
    expressions, excluded = [], []
    for name, dtype in schema.items():
        coefficient = correlation_expr(name, y, schema, "pearson") if dtype.is_numeric() else pl.lit(None, dtype=pl.Float64)
        if not dtype.is_numeric():
            excluded.append(name)
        expressions.append(pl.struct(coefficient.alias("coefficient"), pl.col(name).n_unique().alias("distinct")).alias(name))
    scalar = frame(data).select(expressions).collect().row(0, named=True)
    result = {name: [stat["coefficient"], stat["distinct"]] for name, stat in scalar.items()}
    return (result, excluded) if excluded_list else result
