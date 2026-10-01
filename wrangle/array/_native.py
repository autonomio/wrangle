"""Native Polars normalization and row-alignment boundaries."""
import polars as pl
from .._core import frame, WrangleError


def rows(data):
    return frame(data).select(pl.len()).collect().item()


def aligned(*data):
    sizes = [rows(item) for item in data]
    if len(set(sizes)) > 1:
        raise WrangleError("ROW_ALIGNMENT", "All inputs must have the same row count.", {"rows": sizes})
    return sizes[0] if sizes else 0


def restore(plan, original):
    if isinstance(original, pl.LazyFrame):
        return plan
    table = plan.collect()
    if isinstance(original, pl.Series) or (isinstance(original, (list, tuple)) and (not original or not isinstance(original[0], (list, tuple, dict)))):
        return table.to_series()
    return table


def positive_int(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise WrangleError("INVALID_ARGUMENT", f"{name} must be {'nonnegative' if allow_zero else 'positive'} integer.")
    return value


def checked_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise WrangleError("INVALID_SEED", "Declare an integer seed in [0, 2**64).")
    return seed


def shuffled(data, seed):
    return restore(frame(data).sort(pl.int_range(0, pl.len()).hash(seed=checked_seed(seed)), maintain_order=True), data)
