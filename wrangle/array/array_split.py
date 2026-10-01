"""Aligned training/validation slices with explicit reproducible shuffling."""
from .._core import operation
from .._core import frame, WrangleError
from ._native import aligned, restore
from .array_random_shuffle import array_random_shuffle


@operation(returns=('tuple',), recipe='never')
def array_split(x, y, split, shuffled=True, *, seed=0):
    """Return x_train, y_train, x_validation, y_validation; split is the validation fraction."""
    if isinstance(split, bool) or not isinstance(split, (int, float)) or not 0 < split < 1:
        raise WrangleError("INVALID_SPLIT", "Validation fraction must be strictly between zero and one.")
    count = aligned(x, y)
    limit = int(count * (1 - split))
    if limit == 0 or limit == count:
        raise WrangleError("EMPTY_SPLIT", "Both partitions must contain observations.")
    if shuffled:
        x, y = array_random_shuffle(x, y, seed=seed)
    return (restore(frame(x).slice(0, limit), x), restore(frame(y).slice(0, limit), y),
            restore(frame(x).slice(limit), x), restore(frame(y).slice(limit), y))
