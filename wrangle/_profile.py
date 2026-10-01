"""Read-only native profiles with explicit summary and sampling boundaries."""
from __future__ import annotations

from collections.abc import Mapping
import json

import polars as pl

from ._core import WrangleError, floating_count, require_columns

__all__ = ["profile", "validate_options"]

_SUMMARY_POLICY = {
    "denominator": "non-null observations",
    "ddof": 1,
    "nulls": "reported separately; excluded from descriptive statistics",
    "nonfinite": "any NaN or infinity leaves statistics unresolved; none are excluded",
    "precision": "integer and Decimal observations must survive a Float64 roundtrip",
    "std": "null when fewer than two non-null observations",
}


def _names(value, name):
    if value is not None and (not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value) or len(set(value)) != len(value)):
        raise WrangleError("INVALID_ARGUMENT", f"{name} must be a nonempty list of distinct column names.")


def validate_options(*, sample_rows=5, columns=None, summary=False, groups=None, max_groups=20):
    """Validate inspection controls without evaluating or reading a source."""
    if type(sample_rows) is not int or not 0 <= sample_rows <= 100:
        raise WrangleError("INVALID_ARGUMENT", "sample_rows must be an integer between 0 and 100.")
    if type(summary) is not bool:
        raise WrangleError("INVALID_ARGUMENT", "summary must be true or false.")
    if type(max_groups) is not int or not 1 <= max_groups <= 100:
        raise WrangleError("INVALID_ARGUMENT", "max_groups must be an integer between 1 and 100.")
    _names(columns, "columns")
    _names(groups, "groups")


def _has_binary(dtype):
    if dtype == pl.Binary:
        return True
    if isinstance(dtype, pl.Struct):
        return any(_has_binary(field.dtype) for field in dtype.fields)
    return isinstance(dtype, (pl.List, pl.Array)) and _has_binary(dtype.inner)


def _display(expression, dtype):
    """Encode bytes only for JSON display, preserving container nulls natively."""
    if dtype == pl.Binary:
        return expression.bin.encode("hex")
    if isinstance(dtype, pl.Struct) and _has_binary(dtype):
        fields = [_display(expression.struct.field(field.name), field.dtype).alias(field.name) for field in dtype.fields]
        return pl.when(expression.is_null()).then(None).otherwise(pl.struct(fields))
    if isinstance(dtype, (pl.List, pl.Array)) and _has_binary(dtype):
        items = expression.arr.to_list() if isinstance(dtype, pl.Array) else expression
        return items.list.eval(_display(pl.element(), dtype.inner))
    return expression


def _expressions(name, dtype, prefix, summary):
    value = pl.col(name)
    # Row-dependent zeros also aggregate correctly on the supported Polars minimum.
    expressions = [
        value.is_null().sum().alias(prefix + "nulls"),
        (floating_count(value, dtype, kind="nan") + value.is_null().cast(pl.UInt64) * 0).sum().alias(prefix + "nans"),
        (floating_count(value, dtype, kind="infinite") + value.is_null().cast(pl.UInt64) * 0).sum().alias(prefix + "infinities"),
        value.n_unique().alias(prefix + "distinct"),
    ]
    if not summary:
        return expressions
    expressions.append(value.count().alias(prefix + "count"))
    if dtype.is_numeric():
        numeric = value.cast(pl.Float64, strict=True)
        lossy = (~value.eq_missing(numeric.cast(dtype, strict=False))).sum() if dtype.is_integer() or isinstance(dtype, pl.Decimal) else pl.lit(0, dtype=pl.UInt64)
        statistics = {"min": value.min(), "max": value.max(), "mean": numeric.mean(), "median": numeric.median(), "std": numeric.std(ddof=1)}
        expressions.extend(expression.alias(prefix + statistic) for statistic, expression in statistics.items())
        expressions.extend([
            lossy.alias(prefix + "lossy"),
            pl.any_horizontal([expression.is_not_null() & ~expression.is_finite() for key, expression in statistics.items() if key in {"mean", "median", "std"}]).alias(prefix + "nonfinite_statistics"),
        ])
    return expressions


def _field(name, dtype, values, prefix, summary):
    result = {"name": name, "dtype": str(dtype), **{key: values[prefix + key] for key in ("nulls", "nans", "infinities", "distinct")}}
    if summary:
        count = values[prefix + "count"]
        reason = "NONFINITE_VALUES" if result["nans"] or result["infinities"] else "NON_NUMERIC" if not dtype.is_numeric() else "NO_OBSERVATIONS" if not count else "LOSSY_FLOAT64" if values[prefix + "lossy"] else "NONFINITE_STATISTIC" if values[prefix + "nonfinite_statistics"] else None
        descriptive = {"status": "unresolved" if reason else "resolved", "count": count, **{key: None if reason else values[prefix + key] for key in ("min", "max", "mean", "median", "std")}}
        if reason:
            descriptive["reason"] = reason
        result["summary"] = descriptive
    return result


def _prefixes(names, reserved):
    prefixes = {}
    occupied = set(reserved)
    for i, name in enumerate(names):
        prefix = f"__wrangle_profile_{i}_"
        while any(prefix + suffix in occupied for suffix in ("nulls", "nans", "infinities", "distinct", "count", "min", "max", "mean", "median", "std", "lossy", "nonfinite_statistics")):
            prefix = "_" + prefix
        prefixes[name] = prefix
        occupied.update(prefix + suffix for suffix in ("nulls", "nans", "infinities", "distinct", "count", "min", "max", "mean", "median", "std", "lossy", "nonfinite_statistics"))
    return prefixes


def _fields(data, names, summary):
    if not names:
        return []
    prefixes = _prefixes(names, data.columns)
    expressions = [expression for name in names for expression in _expressions(name, data.schema[name], prefixes[name], summary)]
    values = json.loads(data.lazy().select(expressions).collect().write_json())[0]
    return [_field(name, data.schema[name], values, prefixes[name], summary) for name in names]


def _groups(data, names, groups, max_groups):
    count = data.lazy().select(pl.struct(groups).n_unique()).collect().item() if data.height else 0
    result = {"by": groups, "count": count, "max_groups": max_groups, "truncated": count > max_groups, "items": []}
    nonfinite = [floating_count(pl.col(name), data.schema[name], kind=kind).sum().alias(f"{i}_{kind}") for i, name in enumerate(groups) for kind in ("nan", "infinite")]
    counts = data.lazy().select(nonfinite).collect().row(0) if nonfinite else ()
    if any(counts):
        return {**result, "status": "unresolved", "reason": "NONFINITE_GROUP_KEY"}
    prefixes = _prefixes(names, data.columns)
    row_name = "__wrangle_profile_rows"
    while row_name in data.columns or any(row_name.startswith(prefix) for prefix in prefixes.values()):
        row_name = "_" + row_name
    expressions = [pl.len().alias(row_name), *[expression for name in names for expression in _expressions(name, data.schema[name], prefixes[name], True)]]
    observed = data.lazy().group_by(groups, maintain_order=True).agg(expressions).head(max_groups).collect()
    labels = observed.with_columns([_display(pl.col(name), data.schema[name]).alias(name) for name in groups])
    for values in json.loads(labels.write_json()):
        result["items"].append({"values": {name: values[name] for name in groups}, "rows": values[row_name], "fields": [_field(name, data.schema[name], values, prefixes[name], True) for name in names]})
    return result


def _schema_changes(columns, baseline):
    if not isinstance(baseline, Mapping) or any(not isinstance(name, str) or not name or not isinstance(dtype, str) or not dtype for name, dtype in baseline.items()):
        raise WrangleError("INVALID_ARGUMENT", "baseline_columns must map column names to observed dtype strings.")
    return {
        "added": [{"name": name, "dtype": dtype} for name, dtype in columns.items() if name not in baseline],
        "removed": [{"name": name, "dtype": dtype} for name, dtype in baseline.items() if name not in columns],
        "type_changed": [{"name": name, "before": baseline[name], "after": dtype} for name, dtype in columns.items() if name in baseline and baseline[name] != dtype],
    }


def profile(data, source_info, *, sample_rows=5, columns=None, summary=False, groups=None, max_groups=20, baseline_columns=None):
    """Observe a captured table; numeric summaries use non-null counts and sample std (ddof=1).

    Nonfinite observations, lossy Float64 conversions, and nonfinite statistics
    leave summaries unresolved. Group requests include summaries, bounded in
    first-appearance order; null grouping keys remain observed groups. Baseline
    comparison reports complete observed schemas without converting either.
    JSON displays Binary values as hexadecimal strings and nonfinite floats as
    null; encoding metadata makes those displays distinct from source values.
    """
    validate_options(sample_rows=sample_rows, columns=columns, summary=summary, groups=groups, max_groups=max_groups)
    names = data.columns if columns is None else columns
    require_columns(data, names)
    if groups is not None:
        require_columns(data, groups)
    try:
        displayed_names = [*names, *(name for name in groups or [] if name not in names)]
        display = data.lazy().select([_display(pl.col(name), data.schema[name]).alias(name) for name in names]).head(sample_rows).collect()
        result = {**source_info, "fields": _fields(data, names, summary), "examples": json.loads(display.write_json()), "example_encoding": {"binary": "lowercase hexadecimal strings", "binary_columns": [name for name in displayed_names if _has_binary(data.schema[name])], "nonfinite_floats": "JSON null; nans and infinities counts preserve their distinction", "scope": "examples and group labels only; source dtypes and snapshot are unchanged"}, "decisions": ["Declare variable meanings, units, row identity, and missing-value codes in the research protocol."]}
        if summary or groups is not None:
            result["summary_policy"] = dict(_SUMMARY_POLICY)
        if groups is not None:
            result["groups"] = _groups(data, names, groups, max_groups)
        if baseline_columns is not None:
            result["schema_changes"] = _schema_changes({name: str(dtype) for name, dtype in data.schema.items()}, baseline_columns)
        return result
    except pl.exceptions.PolarsError as error:
        raise WrangleError("PROFILE_UNSUPPORTED", "The selected native dtype cannot be profiled with these controls.", {"error": str(error)}) from error
