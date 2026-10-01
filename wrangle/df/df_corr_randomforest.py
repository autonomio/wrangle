from .._core import operation
from .._core import WrangleError


@operation(returns=('none',), recipe='never', retired=True)
def df_corr_randomforest(data, y, destructive=False):
    """Retired: predictive model fitting is outside data preparation."""
    raise WrangleError("MODEL_ENGINE_REQUIRED", "Export the prepared Polars table and fit a random forest in your analysis environment; Wrangle prepares data only.")
