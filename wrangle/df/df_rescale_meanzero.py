from .._core import operation
from .._core import frame, finish


@operation(returns=('table',), recipe='yes')
def df_rescale_meanzero(data, retain=None, *, ddof=1, parameters=None):
    """Standardize numeric fields; optional resolved means/scales apply without refitting.

    Constant/singleton fits map observed values to zero; null/NaN remain null.
    Frozen zero scales reject different new observations. Exact Float64 input
    conversion, observed counts and ddof are checked. retain preserves named
    fields, and the returned table preserves the input's eager/lazy form.
    """
    from .._recipe_statistics import resolve_parameters, standardize
    from ._expressions import names
    plan = frame(data)
    resolved = resolve_parameters('df_rescale_meanzero', plan, {'retain': retain, 'ddof': ddof, 'parameters': parameters})
    retained = names(retain)
    selected = [name for name, dtype in plan.collect_schema().items() if dtype.is_numeric() and name not in retained]
    return finish(standardize(plan, selected, ddof=ddof, parameters=resolved['parameters']) if selected else plan, data)
