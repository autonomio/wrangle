"""Disjoint folds that retain every observation, including remainder rows."""
from .._core import operation
from .._core import frame, WrangleError
from ._native import aligned, positive_int, restore
from .array_random_shuffle import array_random_shuffle


@operation(returns=('tuple',), recipe='never')
def array_to_kfold(x, y, folds=10, shuffled=True, *, seed=0):
    """Return lists of aligned feature and target folds; sizes differ by at most one."""
    count = aligned(x, y)
    positive_int(folds, "folds")
    if folds < 2 or folds > count:
        raise WrangleError("INVALID_FOLDS", "Use between two and the number of observations folds.")
    if shuffled:
        x, y = array_random_shuffle(x, y, seed=seed)
    size, remainder = divmod(count, folds)
    out_x, out_y, offset = [], [], 0
    for i in range(folds):
        length = size + (i < remainder)
        out_x.append(restore(frame(x).slice(offset, length), x))
        out_y.append(restore(frame(y).slice(offset, length), y))
        offset += length
    return out_x, out_y
