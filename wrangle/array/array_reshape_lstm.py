"""Chronological rolling sequence preparation with native nested-list expressions."""
from .._core import operation
import polars as pl
from .._core import as_series, WrangleError
from ._native import positive_int
from .array_random_shuffle import array_random_shuffle


@operation(returns=('series',), recipe='never')
def normalise_windows(window_data):
    """Normalize each list by its first value, rejecting zero baselines."""
    try:
        windows = as_series(window_data)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        if isinstance(error, WrangleError):
            raise
        raise WrangleError("INVALID_INPUT", "Normalization requires numeric list windows.") from error
    if not isinstance(windows.dtype, pl.List):
        raise WrangleError("INVALID_INPUT", "Normalization requires a Series of numeric lists.")
    if not windows.dtype.inner.is_numeric():
        raise WrangleError("INVALID_INPUT", "Normalization requires numeric list elements.")
    plan = windows.to_frame().lazy()
    value = pl.col(windows.name)
    invalid = value.is_null() | (value.list.len() == 0) | value.list.eval(pl.element().is_null() | ~pl.element().is_finite()).list.any()
    if plan.select(invalid.any()).collect().item():
        raise WrangleError("INVALID_INPUT", "Normalization requires nonempty windows with finite, non-null values.")
    if plan.select((value.list.first() == 0).any()).collect().item():
        raise WrangleError("ZERO_BASELINE", "Window normalization cannot divide by a zero baseline.")
    return plan.select(value.list.eval(pl.element().cast(pl.Float64) / pl.element().first() - 1).alias(windows.name)).collect().to_series()



@operation(returns=('sequence',), recipe='never')
def array_reshape_lstm(data, seq_len, normalise_window=False, *, validation_split=0.1, shuffled=True, seed=0):
    """Return x_train, y_train, x_test, y_test as Polars Series; tensor elements are singleton feature lists."""
    values = as_series(data)
    positive_int(seq_len, "seq_len")
    if not values.dtype.is_numeric() or values.null_count() or values.to_frame().select((~pl.col(values.name).is_finite()).any()).item():
        raise WrangleError("INVALID_INPUT", "Sequence data must be finite, non-null numeric observations.")
    if not 0 < validation_split < 1:
        raise WrangleError("INVALID_SPLIT", "Validation fraction must be strictly between zero and one.")
    count = len(values) - seq_len
    split = int(count * (1 - validation_split))
    if split <= 0 or split >= count:
        raise WrangleError("EMPTY_SPLIT", "Sequence length and split must leave both partitions nonempty.")
    value = pl.col(values.name)
    windows = values.to_frame().select(pl.concat_list([value.shift(-i) for i in range(seq_len + 1)]).alias("window")).head(count).to_series()
    if normalise_window:
        windows = normalise_windows(windows)
    dataset = windows.to_frame().select(pl.col("window").list.slice(0, seq_len).list.eval(pl.concat_list(pl.element())).alias("tensor"), pl.col("window").list.last().alias("target"))
    train, test = dataset.head(split), dataset.slice(split)
    if shuffled:
        train = array_random_shuffle(train, seed=seed)
    return [train["tensor"], train["target"], test["tensor"], test["target"]]
