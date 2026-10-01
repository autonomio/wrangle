from .._core import operation
import polars as pl
from .._core import WrangleError, finish, frame, numeric_columns, require_columns
from ._expressions import clean, levels, names_available, numeric


@operation(returns=('table',), recipe='yes', aggregates=True)
def col_to_cols(data, label_col, handler_col):
    """Pivot categories into summed numeric columns per observation identifier.

    One numeric metric uses category names as output columns. Several metrics
    use 'metric_category' names. With no metrics, outputs count observations.
    Missing labels/identifiers raise. Absent categories give zero; an observed
    category with all-missing metric gives null. Handler first-appearance order
    is retained. Duplicate records are intentionally summed, never joined.
    """
    require_columns(data, [label_col, handler_col])
    if label_col == handler_col:
        raise WrangleError("INVALID_PARAMETER", "Label and handler columns must differ.")
    if frame(data).select(pl.any_horizontal(clean(data, label_col).is_null(), clean(data, handler_col).is_null()).any()).collect().item():
        raise WrangleError("MISSING_IDENTIFIER", "Pivot labels and observation identifiers cannot be missing.")
    labels = levels(data, label_col)
    metrics = [name for name in numeric_columns(data) if name not in {label_col, handler_col}]
    for metric in metrics:
        numeric(data, metric)
    outputs = [str(label) if len(metrics) <= 1 else f"{metric}_{label}" for metric in (metrics or [None]) for label in labels]
    names_available(frame(data).select(handler_col), outputs)
    expressions = []
    for metric in metrics or [None]:
        for label in labels:
            matches = clean(data, label_col) == pl.lit(label)
            name = str(label) if len(metrics) <= 1 else f"{metric}_{label}"
            if metric is None:
                expression = matches.cast(pl.UInt64).sum()
            else:
                selected = clean(data, metric).filter(matches)
                total = selected.cast(pl.Int128).sum() if frame(data).collect_schema()[metric].is_integer() else selected.sum()
                expression = pl.when(matches.sum() == 0).then(0).when(selected.count() > 0).then(total).otherwise(None)
            expressions.append(expression.alias(name))
    return finish(frame(data).group_by(handler_col, maintain_order=True).agg(expressions), data)
