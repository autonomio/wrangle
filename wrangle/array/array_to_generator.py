"""Aligned native Polars batches."""
from .._core import operation
from .._core import frame, WrangleError
from ._native import aligned, positive_int, restore


@operation(returns=('generator',), recipe='never')
def array_to_generator(x, y, batch_size, *, repeat=True):
    """Yield every row, including a short final batch. repeat=True retains the legacy infinite generator."""
    count = aligned(x, y)
    positive_int(batch_size, "batch_size")
    if not count:
        raise WrangleError("EMPTY_DATA", "Batch generation requires observations.")
    while True:
        for offset in range(0, count, batch_size):
            yield restore(frame(x).slice(offset, batch_size), x), restore(frame(y).slice(offset, batch_size), y)
        if not repeat:
            return
