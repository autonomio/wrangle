"""Compatibility failures for model factories, outside Wrangle's preparation scope."""
from .._core import operation
from .._core import WrangleError


def _base_for_model(mode, n=50, neurons=50):
    raise WrangleError("MODEL_ENGINE_REQUIRED", "Wrangle prepares data and does not build or fit neural networks. Use an explicit external model engine.")


@operation(returns=('none',), recipe='never', retired=True)
def create_synth_multi_class_model(n=50, neurons=50):
    """Retired model factory; always raises MODEL_ENGINE_REQUIRED."""
    return _base_for_model("multi_class", n, neurons)


@operation(returns=('none',), recipe='never', retired=True)
def create_synth_regression_model(n=50, neurons=50):
    """Retired model factory; always raises MODEL_ENGINE_REQUIRED."""
    return _base_for_model("continuous", n, neurons)


@operation(returns=('none',), recipe='never', retired=True)
def create_synth_multi_label_model(n=50, neurons=50):
    """Retired model factory; always raises MODEL_ENGINE_REQUIRED."""
    return _base_for_model("multi_label", n, neurons)


@operation(returns=('none',), recipe='never', retired=True)
def create_synth_binary_model(n=50, neurons=50):
    """Retired model factory; always raises MODEL_ENGINE_REQUIRED."""
    return _base_for_model("binary", n, neurons)
