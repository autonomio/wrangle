from .._core import operation
import polars as pl
from .._core import frame, finish, require_columns, WrangleError
from ._expressions import names, missing, temp_name


@operation(returns=('table',), recipe='yes')
def df_merge(data1, data2, on_index=True, on_column=False, *, how="inner", validate="1:1", allow_unmatched=False, suffix="_right"):
    """Join explicitly checked keys, or equally sized row positions; reject losses by default."""
    if how not in {"inner", "left", "right", "full"}:
        raise WrangleError("INVALID_OPTION", "how must be inner, left, right, or full.")
    if validate not in {"1:1", "1:m", "m:1", "m:m"}:
        raise WrangleError("INVALID_OPTION", "validate must be 1:1, 1:m, m:1, or m:m.")
    left, right = frame(data1), frame(data2)
    positional = on_column is False or on_column is None
    if positional:
        if not on_index:
            raise WrangleError("INVALID_OPTION", "Specify on_column or enable row-position joining.")
        keys = [temp_name(data1, temp_name(data2, "__wrangle_join_row"))]
        counts = [left.select(pl.len()).collect().item(), right.select(pl.len()).collect().item()]
        if counts[0] != counts[1]:
            raise WrangleError("JOIN_LOSS", "Row-position joining requires equal row counts.", {"left": counts[0], "right": counts[1]})
        left, right = left.with_row_index(keys[0]), right.with_row_index(keys[0])
    else:
        keys = names(on_column)
        if not keys:
            raise WrangleError("INVALID_OPTION", "Join keys must not be empty.")
        require_columns(data1, keys)
        require_columns(data2, keys)
    left_schema, right_schema = left.collect_schema(), right.collect_schema()
    mismatched = [key for key in keys if left_schema[key] != right_schema[key]]
    if mismatched:
        raise WrangleError("DTYPE_MISMATCH", "Join key types differ; explicitly cast keys before joining.", {"columns": mismatched})
    output_names = list(left_schema) + [name + suffix if name in left_schema else name for name in right_schema if name not in keys]
    if len(output_names) != len(set(output_names)):
        raise WrangleError("COLUMN_COLLISION", "Join suffix creates duplicate output names; rename columns or choose another suffix.")
    for label, plan in (("left", left), ("right", right)):
        schema = plan.collect_schema()
        invalid = plan.select(pl.any_horizontal(missing(key, schema[key]) for key in keys).sum()).collect().item()
        if invalid:
            raise WrangleError("MISSING_KEY", "Join keys contain missing values.", {"source": label, "rows": invalid})
        unique_required = (label == "left" and validate in {"1:1", "1:m"}) or (label == "right" and validate in {"1:1", "m:1"})
        if unique_required:
            duplicates = plan.group_by(keys).len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
            if duplicates:
                raise WrangleError("JOIN_CARDINALITY", "Join keys violate declared cardinality.", {"source": label, "duplicate_keys": duplicates, "validate": validate})
    if not allow_unmatched:
        left_only = left.join(right.select(keys).unique(), on=keys, how="anti").select(pl.len()).collect().item()
        right_only = right.join(left.select(keys).unique(), on=keys, how="anti").select(pl.len()).collect().item()
        if left_only or right_only:
            raise WrangleError("JOIN_LOSS", "Join has unmatched keys; declare allow_unmatched=True to permit them.", {"left_unmatched": left_only, "right_unmatched": right_only})
    result = left.join(right, on=keys, how=how, validate=validate, suffix=suffix, coalesce=True, maintain_order="left_right")
    if positional:
        result = result.drop(keys)
    return finish(result, data1)
