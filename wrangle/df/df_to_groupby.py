from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, WrangleError
from ._expressions import names, aggregation, seed_value, domain, clean, require_native, require_float_exactness


@operation(returns=('table',), recipe='yes', aggregates=True)
def df_to_groupby(data, by, func, *, seed=0):
    """Aggregate in first-seen group order; integer sums widen to Int128; randomness is seeded."""
    keys = names(by)
    if not keys:
        raise WrangleError("INVALID_OPTION", "Grouping keys must not be empty.")
    require_columns(data, keys)
    seed_value(seed)
    schema = frame(data).collect_schema()
    if isinstance(func, str) and func in {"mean", "median", "std", "entropy"}:
        require_float_exactness(data, [name for name, dtype in schema.items() if name not in keys and dtype.is_numeric()])
    if isinstance(func, str) and func in {"sum", "entropy"}:
        wide = [name for name, dtype in schema.items() if name not in keys and dtype == pl.Int128]
        if wide:
            checks = []
            for name in wide:
                value = pl.col(name)
                count = value.count().cast(pl.Int128)
                minimum = pl.lit(-(2**127), dtype=pl.Int128)
                maximum = pl.lit(2**127 - 1, dtype=pl.Int128)
                largest = value.filter(value != minimum).abs().max().fill_null(0)
                unsafe = (count > 1) & ((value == minimum).any() | (largest > maximum // count.clip(lower_bound=1)))
                checks.append(unsafe.alias(name))
            flags = frame(data).group_by(keys).agg(checks).select(pl.col(name).any().alias(name) for name in wide).collect().row(0, named=True)
            bad = [name for name, unsafe in flags.items() if unsafe]
            if bad:
                raise WrangleError("ARITHMETIC_OVERFLOW", "The Int128 sum cannot be proven safe; use an exact arithmetic analysis environment for these extreme values.", {"columns": bad})
    if isinstance(func, pl.Expr):
        exprs = [func]
    elif isinstance(func, (list, tuple)) and all(isinstance(expr, pl.Expr) for expr in func):
        exprs = list(func)
    elif isinstance(func, str):
        supported = {"median", "mean", "first", "last", "std", "mode", "max", "min", "sum", "random", "freq", "string", "entropy"}
        if func not in supported:
            raise WrangleError("INVALID_OPTION", "Unknown grouping function.", {"allowed": sorted(supported)})
        if func == 'entropy':
            numeric = [name for name, dtype in schema.items() if name not in keys and dtype.is_numeric()]
            domain(data, numeric, 0)
            if numeric:
                totals = frame(data).group_by(keys).agg((clean(name, schema[name]).cast(pl.Int128) if schema[name].is_integer() else clean(name, schema[name])).sum().alias(name) for name in numeric)
                invalid = totals.select((pl.col(name) <= 0).any().alias(name) for name in numeric).collect().row(0, named=True)
                bad = [name for name, value in invalid.items() if value]
                if bad:
                    raise WrangleError("INVALID_DOMAIN", "Entropy requires positive total mass in every group.", {"columns": bad})
        exprs = [expr for name, dtype in schema.items() if name not in keys if (expr := aggregation(name, dtype, func, seed)) is not None]
    else:
        raise WrangleError("UNSUPPORTED_CALLBACK", "Use a supported aggregation name or native Polars expressions.")
    require_native(exprs)
    return finish(frame(data).group_by(keys, maintain_order=True).agg(exprs), data)
