"""Deterministic synthetic fixtures, generated exclusively with Polars expressions."""
from .._core import operation
import polars as pl
from .._core import WrangleError
from ..array._native import positive_int, checked_seed


@operation(returns=('tuple',), recipe='never')
def create_synth_data(mode="binary", n=1000, features=20, classes=4, *, seed=0):
    """Return x and y Polars tables. Uniform features; independent class indicators; continuous y is their sum. These are fixtures, not fitted scientific models."""
    positive_int(n, "n", allow_zero=True)
    positive_int(features, "features")
    positive_int(classes, "classes")
    checked_seed(seed)
    if mode == "regression":
        mode = "continuous"
    if mode not in {"binary", "multi_class", "multi_label", "continuous"}:
        raise WrangleError("INVALID_MODE", "Use binary, multi_class, multi_label, or continuous synthetic fixtures.")
    if features > 10000 or classes > 10000:
        raise WrangleError("RESOURCE_LIMIT", "Synthetic feature and class counts are limited to 10000.")
    plan = pl.LazyFrame().select(pl.int_range(0, n).alias("row"))
    def uniform(offset):
        return ((pl.col("row").hash(seed=(seed + offset) % 2**64) // 2048).cast(pl.Float64) / 2**53)
    expressions = [uniform(i).alias(f"x_{i}") for i in range(features)]
    x = plan.select(expressions).collect()
    if mode == "continuous":
        y = x.select(pl.sum_horizontal(pl.all()).alias("target"))
    elif mode == "multi_label":
        y = plan.select([(uniform(features+i) >= 0.5).cast(pl.UInt8).alias(f"class_{i}") for i in range(classes)]).collect()
    else:
        count = 2 if mode == "binary" else classes
        y = plan.select((uniform(features) * count).floor().cast(pl.UInt32).alias("target")).collect()
    return x, y
