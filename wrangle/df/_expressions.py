"""Native expression building shared by the dataframe operations."""

import polars as pl
from .._storage import collect
from .._core import WrangleError, columns, frame, require_columns


def names(value, *, default=()):
    if value is None:
        return list(default)
    return [value] if isinstance(value, str) else list(value)


def missing(name, dtype):
    expr = pl.col(name).is_null()
    return expr | pl.col(name).is_nan() if dtype.is_float() else expr


def clean(name, dtype):
    expr = pl.col(name)
    return expr.fill_nan(None) if dtype.is_float() else expr


def temp_name(data, base="__wrangle_row"):
    present = columns(data)
    while base in present:
        base += "_"
    return base


def fraction(value, name="threshold"):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise WrangleError("INVALID_OPTION", f"{name} must be between 0 and 1.")


def seed_value(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
        raise WrangleError("INVALID_SEED", "seed must be an integer from 0 through 2**64 - 1.")


def positive_n(n, *, allow_none=False):
    if n is None and allow_none:
        return
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise WrangleError("INVALID_OPTION", "n must be a nonnegative integer.")


def numeric_only(data, selected):
    require_columns(data, selected)
    schema = frame(data).collect_schema()
    bad = [name for name in selected if not schema[name].is_numeric()]
    if bad:
        raise WrangleError("NON_NUMERIC_COLUMN", "Operation requires numeric columns.", {"columns": bad})
    return schema


def domain(data, selected, bound, *, strict=False):
    if not selected:
        return
    schema = numeric_only(data, selected)
    exprs = []
    for name in selected:
        value = clean(name, schema[name])
        invalid = value <= bound if strict else value < bound
        exprs.append(invalid.fill_null(False).sum().alias(name))
    counts = collect(frame(data).select(exprs)).row(0, named=True)
    bad = {name: count for name, count in counts.items() if count}
    if bad:
        raise WrangleError("INVALID_DOMAIN", "Values are outside the transformation domain.", bad)


def require_float_exactness(data, selected):
    """Reject numeric observations that a Float64 conversion would change."""
    require_columns(data, selected)
    schema = frame(data).collect_schema()
    selected = [name for name in selected if schema[name].is_integer() or isinstance(schema[name], pl.Decimal)]
    if not selected:
        return
    checks = []
    for name in selected:
        value = pl.col(name)
        roundtrip = value.cast(pl.Float64).cast(schema[name], strict=False)
        checks.append((~value.eq_missing(roundtrip)).sum().alias(name))
    counts = collect(frame(data).select(checks)).row(0, named=True)
    bad = {name: count for name, count in counts.items() if count}
    if bad:
        raise WrangleError("LOSSY_CAST", "Observed values cannot be represented exactly as Float64; use an exact arithmetic analysis environment.", {"columns": bad})


def impute_expr(name, dtype, mode, seed=0, row_expr=None):
    """Build an imputation expression; null and NaN share missing semantics."""
    allowed = {"mean", "median", "mode", "common", "mean_by_std"}
    if mode not in allowed:
        raise WrangleError("INVALID_OPTION", "Unknown impute_mode.", {"allowed": sorted(allowed)})
    value = clean(name, dtype)
    if mode in {"mode", "common"}:
        replacement = value.drop_nulls().mode().sort().first()
    else:
        if not dtype.is_numeric():
            raise WrangleError("NON_NUMERIC_COLUMN", f"{mode} imputation requires numeric data.", {"column": name})
        value = value.cast(pl.Float64)
        if mode == "mean":
            replacement = value.mean()
        elif mode == "median":
            replacement = value.median()
        else:
            seed_value(seed)
            row_expr = pl.int_range(pl.len(), dtype=pl.UInt64) if row_expr is None else row_expr
            unit = pl.struct(row_expr.alias("row"), pl.lit(name).alias("column")).hash(seed=seed).cast(pl.Float64) / float(2**64 - 1)
            replacement = value.mean() + (2 * unit - 1) * value.std(ddof=1).fill_null(0)
    return value.fill_null(replacement).alias(name)


def correlation_expr(left, right, schema, method):
    """Pairwise-complete coefficient suitable for one batched aggregation."""
    if method not in {"pearson", "spearman"}:
        raise WrangleError("INVALID_OPTION", "Native correlation expression requires pearson or spearman.")
    return pl.corr(clean(left, schema[left]).cast(pl.Float64), clean(right, schema[right]).cast(pl.Float64), method=method).fill_nan(None)


def correlation(data, left, right, method):
    if method not in {"pearson", "spearman", "kendall"}:
        raise WrangleError("INVALID_OPTION", "method must be pearson, spearman, or kendall.")
    schema = numeric_only(data, [left, right])
    if method != "kendall":
        require_float_exactness(data, [left, right])
    pair = frame(data).select(clean(left, schema[left]).alias("x"), clean(right, schema[right]).alias("y")).drop_nulls()
    if method != "kendall":
        result = collect(pair.select(pl.corr("x", "y", method=method))).item()
    else:
        summary = collect(pair.select(pl.len().alias("rows"), pl.col("x").n_unique().alias("x_unique"), pl.col("y").n_unique().alias("y_unique"))).row(0, named=True)
        if summary["rows"] < 2 or summary["x_unique"] < 2 or summary["y_unique"] < 2:
            return None
        pair_count = summary["rows"] * (summary["rows"] - 1) // 2
        if pair_count > 10_000_000:
            raise WrangleError("NATIVE_CORRELATION_LIMIT", "Native Kendall is limited to 10 million valid observation pairs; export the prepared table and compute Kendall in your analysis environment.", {"observations": summary["rows"], "pairs": pair_count, "max_pairs": 10_000_000})
        # Tau-b: ties in either variable are excluded from that denominator.
        pair = pair.with_row_index("i")
        pairs = pair.join(pair.rename({"x": "x2", "y": "y2", "i": "j"}), how="cross").filter(pl.col("i") < pl.col("j"))
        sx = pl.when(pl.col("x") > pl.col("x2")).then(1).when(pl.col("x") < pl.col("x2")).then(-1).otherwise(0)
        sy = pl.when(pl.col("y") > pl.col("y2")).then(1).when(pl.col("y") < pl.col("y2")).then(-1).otherwise(0)
        result = collect(pairs.select((sx * sy).sum() / (((sx != 0).sum().cast(pl.Float64) * (sy != 0).sum().cast(pl.Float64)).sqrt()))).item()
    return None if result is None or result != result else result


def aggregation(name, dtype, func, seed=0):
    value = clean(name, dtype)
    if func in {"mean", "median", "std", "sum", "entropy"} and not dtype.is_numeric():
        return None
    if func == "sum" and dtype.is_integer():
        return value.cast(pl.Int128).sum().alias(name)
    if func in {"mean", "median", "std", "sum", "first", "last", "max", "min"}:
        return getattr(value, func)().alias(name)
    if func in {"mode", "freq"}:
        return value.drop_nulls().mode().sort().first().alias(name)
    if func == "random":
        seed_value(seed)
        return value.shuffle(seed=seed).first().alias(name)
    if func == "string":
        return value.cast(pl.String).drop_nulls().str.join(" ").alias(name)
    if func == "entropy":
        return value.entropy().alias(name)
    raise WrangleError("INVALID_OPTION", "Unknown grouping function.", {"func": func})


# One shared validator also guards input LazyFrames in _core.frame().
from .._core import require_native as require_native
