"""Explicit legacy X/Y selection; no automatic text vectorization or suppressed warnings."""
from .._core import operation
from .._core import frame, WrangleError
from .multi_input_support import multi_input_support


@operation(returns=('table',), recipe='never')
def X_data(X, data):
    """Select features without lossy conversion or inferred encoding."""
    return multi_input_support(X, data)


@operation(returns=('series', 'table'), recipe='never')
def Y_data(Y, data, flatten=False):
    """Select targets; flatten=True returns a Series only when exactly one target was selected."""
    result = multi_input_support(Y, data)
    if flatten:
        table = frame(result).collect()
        if table.width != 1:
            raise WrangleError("INVALID_INPUT", "Flattening requires exactly one target column.")
        return table.to_series()
    return result


@operation(returns=('series', 'table', 'tuple'), recipe='conditional')
def transform_data(data, flatten=False, X=False, Y=False):
    """Return selected features, targets, or their pair as Polars objects; preserve scientific types."""
    if X is False and Y is False:
        raise WrangleError("INVALID_INPUT", "Declare X and/or Y column selections.")
    if Y is False:
        return X_data(X, data)
    if X is False:
        return Y_data(Y, data, flatten)
    return X_data(X, data), Y_data(Y, data, flatten)
