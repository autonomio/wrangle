from .._core import operation
from .._core import frame, finish


@operation(returns=('table',), recipe='yes')
def df_impute_nan(data, cols='all', impute_mode='mean_by_std', destructive=False, *, seed=0, parameters=None):
    """Impute chosen columns; optional resolved parameters reuse replacements across batches.

    Legacy methods/default seed stay available. mean/median/mean_by_std require
    exact Float64 input conversion; mode/common preserve scalar dtypes. Seeded
    uniform draws use independent named-column streams. All-missing fits fail;
    supplied dtype-checked parameters never refit. Inputs remain immutable.
    """
    from .._recipe_statistics import impute, resolve_parameters
    plan = frame(data)
    resolved = resolve_parameters('df_impute_nan', plan, {
        'cols': cols, 'impute_mode': impute_mode, 'destructive': destructive,
        'seed': seed, 'parameters': parameters,
    })
    selected = resolved['cols']
    if not selected:
        return finish(plan, data)
    method = {'common': 'mode', 'mean_by_std': 'uniform'}.get(impute_mode, impute_mode)
    return finish(impute(plan, selected, method=method, parameters=resolved['parameters'], seed=seed if method == 'uniform' else None), data)
