"""Reproducible row shuffling, without input mutation."""
from .._core import operation
from .._core import WrangleError
from ._native import aligned, shuffled, checked_seed


@operation(returns=('sequence', 'series', 'table', 'tuple'), recipe='conditional')
def array_random_shuffle(x, y=None, multi_input=False, *, seed=0):
    """Return equally permuted inputs. Multi-input lists require explicit multi_input=True; seed defaults to 0."""
    checked_seed(seed)
    inputs = list(x) if multi_input else [x]
    if not inputs:
        raise WrangleError("INVALID_INPUT", "At least one feature input is required.")
    aligned(*inputs, *([] if y is None else [y]))
    result = [shuffled(item, seed) for item in inputs]
    x_out = result if multi_input else result[0]
    return x_out if y is None else (x_out, shuffled(y, seed))
