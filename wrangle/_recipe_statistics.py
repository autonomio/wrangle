"""Explicit missing-value fills and reusable native Polars preparation parameters."""
from __future__ import annotations

import json
import math
import re

import polars as pl

from ._storage import collect
from ._core import WrangleError, dtype_spec, frame, operation, reject_destructive, require_columns
from .df._expressions import clean, names, require_float_exactness, seed_value


_METHODS = {"mean", "median", "mode", "uniform"}
_SHIFT = float(2**512)
_SMALL = 1.0 / _SHIFT


def _fields(data, columns):
    if not isinstance(columns, list) or not columns or any(not isinstance(name, str) or not name for name in columns) or len(columns) != len(set(columns)):
        raise WrangleError("INVALID_ARGUMENT", "columns must be a nonempty list of distinct field names.")
    require_columns(data, columns)
    return data.collect_schema()


def _ddof(value):
    if type(value) is not int or value < 0:
        raise WrangleError("INVALID_OPTION", "ddof must be a nonnegative integer.")


def _method(value):
    if not isinstance(value, str) or value not in _METHODS:
        raise WrangleError("INVALID_OPTION", "method must be mean, median, mode or uniform.")


def _json(value):
    def validate(item):
        if item is None or type(item) in {str, int, bool}:
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child)
            return
        raise WrangleError("INVALID_PARAMETERS", "Parameters must contain finite JSON scalars, lists and string-keyed objects.")
    validate(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise WrangleError("INVALID_PARAMETERS", "Parameters must be exact finite JSON values.") from error


def _descriptor(dtype):
    if isinstance(dtype, pl.Datetime):
        return {"Datetime": {"time_unit": dtype.time_unit, "time_zone": dtype.time_zone}}
    if isinstance(dtype, pl.Duration):
        return {"Duration": dtype.time_unit}
    if isinstance(dtype, pl.Decimal):
        return {"Decimal": {"precision": dtype.precision, "scale": dtype.scale}}
    if isinstance(dtype, pl.Enum):
        return {"Enum": dtype.categories.to_list()}
    if isinstance(dtype, pl.Categorical):
        return "Categorical"
    if dtype.is_numeric() or dtype in {pl.String, pl.Boolean, pl.Date, pl.Time, pl.Null}:
        return str(dtype)
    raise WrangleError("NON_SCALAR_COLUMN", "Preparation parameters require scalar numeric, text, category or temporal fields.", {"dtype": str(dtype)})


def _declared_dtype(value):
    if value == "Categorical":
        return pl.Categorical
    if isinstance(value, dict) and set(value) == {"Decimal"}:
        declaration = value["Decimal"]
        if isinstance(declaration, dict) and set(declaration) == {"precision", "scale"} and declaration["precision"] is None and type(declaration["scale"]) is int and 0 <= declaration["scale"] <= 38:
            return pl.Decimal(None, declaration["scale"])
    return dtype_spec(value)


def _decimal_text(value, name):
    if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value):
        raise WrangleError("INVALID_FILL_VALUE", "Decimal replacements require finite fixed-point strings.", {"column": name})
    scale = len(value.split(".", 1)[1]) if "." in value else 0
    if scale > 38:
        raise WrangleError("LOSSY_FILL", "Decimal replacements exceed native precision.", {"column": name})
    return pl.lit(value).cast(pl.Decimal(38, scale), strict=True)


def _literal(value, dtype, name, *, physical=False):
    """Construct declared JSON scalar metadata with a native exact round trip."""
    if value is None:
        return pl.lit(None, dtype=dtype)
    if type(value) not in {str, int, float, bool} or type(value) is float and not math.isfinite(value):
        raise WrangleError("INVALID_FILL_VALUE", "Replacements must be finite JSON scalars.", {"column": name})
    try:
        if dtype.is_temporal():
            if physical:
                if type(value) is not int:
                    raise WrangleError("INVALID_PARAMETERS", "Temporal resolved values require exact integer physical units.", {"column": name})
                literal = pl.lit(value, dtype=pl.Int64)
                typed = literal.cast(dtype, strict=True)
                exact = pl.select(literal.eq_missing(typed.cast(pl.Int64))).item()
            elif dtype == pl.Date and isinstance(value, str):
                typed = pl.lit(value).str.strptime(pl.Date, "%Y-%m-%d", strict=True, exact=True)
                exact = pl.select(typed.dt.strftime("%Y-%m-%d") == value).item()
            else:
                raise WrangleError("INVALID_FILL_VALUE", "Temporal fill values require a typed physical scalar; Date also accepts YYYY-MM-DD.", {"column": name})
        else:
            text = dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))
            if text and type(value) is not str or dtype == pl.Boolean and type(value) is not bool:
                raise WrangleError("INVALID_FILL_VALUE", "Replacement type must match the declared field dtype.", {"column": name})
            if dtype.is_numeric() and type(value) not in {int, float} and not (isinstance(dtype, pl.Decimal) and type(value) is str):
                raise WrangleError("INVALID_FILL_VALUE", "Numeric fills require numbers; Decimal also accepts exact strings.", {"column": name})
            literal = _decimal_text(value, name) if isinstance(dtype, pl.Decimal) and type(value) is str else pl.lit(value)
            if dtype == pl.Null:
                return literal
            source = pl.select(literal).dtypes[0]
            typed = literal.cast(dtype, strict=True)
            exact = pl.select(literal.eq_missing(typed.cast(source, strict=False))).item()
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, WrangleError):
            raise
        raise WrangleError("INVALID_FILL_VALUE", "Replacement cannot be represented in the declared field dtype.", {"column": name, "dtype": str(dtype)}) from error
    if not exact:
        raise WrangleError("LOSSY_FILL", "Replacement would lose precision in the declared field dtype.", {"column": name, "dtype": str(dtype)})
    return typed


def _numeric(data, columns):
    schema = data.collect_schema()
    nonnumeric = [name for name in columns if not schema[name].is_numeric()]
    if nonnumeric:
        raise WrangleError("NON_NUMERIC_COLUMN", "This method requires explicitly selected numeric fields.", {"columns": nonnumeric})
    require_float_exactness(data, columns)
    counts = collect(data.select([(clean(name, schema[name]).cast(pl.Float64).is_infinite().sum()).alias(name) for name in columns])).row(0, named=True)
    if any(counts.values()):
        raise WrangleError("NONFINITE_RESULT", "Observed infinite values cannot define preparation statistics.", {"columns": {name: count for name, count in counts.items() if count}})


def _ratio(value, scale):
    # Exact powers of two avoid reciprocal overflow for native subnormal values.
    return pl.when((scale != 0) & (scale.abs() < _SMALL)).then((value * _SHIFT) / (scale * _SHIFT)).otherwise(value / scale)


def _moments(value, ddof):
    anchor = value.drop_nulls().first()
    delta = value - anchor
    delta_scale = delta.abs().max()
    centered = pl.when(delta_scale == 0).then(0.0).otherwise(_ratio(delta, delta_scale))
    scale = value.abs().max()
    normalized = pl.when(scale == 0).then(0.0).otherwise(_ratio(value, scale))
    safe_delta = delta_scale.is_finite()
    mean = pl.when(safe_delta).then(anchor + centered.mean() * delta_scale).otherwise(normalized.mean() * scale)
    std = pl.when(safe_delta).then(centered.std(ddof=ddof) * delta_scale).otherwise(normalized.std(ddof=ddof) * scale)
    return mean, std


def _median(value):
    lower = value.quantile(0.5, interpolation="lower")
    higher = value.quantile(0.5, interpolation="higher")
    difference = higher - lower
    return pl.when(difference.is_finite()).then(lower + difference / 2).otherwise(lower / 2 + higher / 2)


def _finite(value, name, field):
    try:
        finite = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise WrangleError("ARITHMETIC_OVERFLOW", "Computed preparation statistics must be representable as finite Float64.", {"column": name, "parameter": field})
    try:
        _literal(value, pl.Float64, name)
    except WrangleError as error:
        raise WrangleError("LOSSY_CAST", "Resolved statistics must be representable exactly as Float64.", {"column": name, "parameter": field}) from error


def _state(data, columns, method, ddof=1):
    schema = _fields(data, columns)
    if method != "mode":
        _numeric(data, columns)
    else:
        floating = [name for name in columns if schema[name].is_float()]
        if floating and collect(data.select(pl.any_horizontal([pl.col(name).is_infinite() for name in floating]).any())).item():
            raise WrangleError("NONFINITE_RESULT", "Observed infinite values cannot define a finite imputation replacement.")
    parameters = {}
    for name in columns:
        dtype = schema[name]
        descriptor = _descriptor(dtype)
        value = clean(name, dtype)
        expressions = [value.count().alias("count")]
        if method == "mode":
            modes = value.cast(pl.String) if isinstance(dtype, pl.Categorical) else value
            replacement = modes.drop_nulls().mode().sort().first()
            if dtype.is_temporal():
                replacement = replacement.cast(pl.Int64)
            elif isinstance(dtype, pl.Decimal):
                replacement = replacement.cast(pl.String)
            expressions.append(replacement.alias("replacement"))
        else:
            value = value.cast(pl.Float64)
            if method in {"mean", "median"}:
                replacement = _moments(value, 1)[0] if method == "mean" else _median(value)
                expressions.append(replacement.alias("replacement"))
            else:
                mean, std = _moments(value, ddof)
                expressions.extend([mean.alias("mean"), std.alias("std"), value.drop_nulls().n_unique().alias("unique")])
        observed = collect(data.select(expressions)).row(0, named=True)
        count = observed["count"]
        if not count:
            raise WrangleError("NO_OBSERVATIONS", "Cannot compute preparation parameters without observed values.", {"column": name})
        entry = {"dtype": descriptor, "count": count}
        if method in {"mean", "median", "mode"}:
            entry.update(method=method, replacement=observed["replacement"])
            if method != "mode":
                _finite(entry["replacement"], name, "replacement")
        else:
            # Missing values do not constitute another observed category.
            unique = observed["unique"]
            if count == 1 or unique == 1:
                std = 0.0
            elif count <= ddof:
                raise WrangleError("INSUFFICIENT_OBSERVATIONS", "Nonconstant fields need more observed values than ddof.", {"column": name, "count": count, "ddof": ddof})
            else:
                std = observed["std"]
                _finite(std, name, "std")
                if std == 0:
                    raise WrangleError("STATISTIC_UNDERFLOW", "Distinct observations have no representable positive standard deviation.", {"column": name})
            mean = observed["mean"]
            _finite(mean, name, "mean")
            entry.update(mean=mean, std=std)
            entry.update({"ddof": ddof} if method == "standardize" else {"method": "uniform"})
        parameters[name] = entry
    return _json(parameters)


def _parameters(data, columns, supplied, method, ddof=1):
    if supplied is None:
        return _state(data, columns, method, ddof)
    schema = _fields(data, columns)
    supplied = _json(supplied)
    if not isinstance(supplied, dict) or set(supplied) != set(columns):
        raise WrangleError("INVALID_PARAMETERS", "Frozen parameters must name exactly the selected fields.")
    if method != "mode":
        _numeric(data, columns)
    for name in columns:
        entry = supplied[name]
        expected = {"dtype", "count", "mean", "std", "ddof"} if method == "standardize" else {"dtype", "count", "mean", "std", "method"} if method == "uniform" else {"dtype", "count", "replacement", "method"}
        if not isinstance(entry, dict) or set(entry) - {"unit"} != expected:
            raise WrangleError("INVALID_PARAMETERS", "Frozen parameter fields do not match the declared method.", {"column": name, "expected": sorted(expected)})
        if "unit" in entry and entry["unit"] is not None and (not isinstance(entry["unit"], str) or not entry["unit"] or entry["unit"].startswith("@")):
            raise WrangleError("INVALID_PARAMETERS", "Resolved units must be a fixed nonempty physical-unit string or null.", {"column": name})
        if _declared_dtype(entry["dtype"]) != schema[name]:
            raise WrangleError("DTYPE_MISMATCH", "Frozen preparation parameters require the same declared field dtype.", {"column": name, "expected": entry["dtype"], "actual": _descriptor(schema[name])})
        if type(entry["count"]) is not int or entry["count"] < 1:
            raise WrangleError("INVALID_PARAMETERS", "Resolved observation counts must be positive integers.", {"column": name})
        if method == "standardize" and (entry["ddof"] != ddof or type(entry["ddof"]) is not int) or method != "standardize" and entry["method"] != method:
            raise WrangleError("INVALID_PARAMETERS", "Frozen parameters belong to a different method or ddof.", {"column": name})
        if method in {"standardize", "uniform"}:
            for parameter in ("mean", "std"):
                _finite(entry[parameter], name, parameter)
            if entry["std"] < 0:
                raise WrangleError("INVALID_PARAMETERS", "Resolved standard deviations must be nonnegative.", {"column": name})
            if method == "standardize" and entry["std"] > 0 and entry["count"] <= ddof:
                raise WrangleError("INSUFFICIENT_OBSERVATIONS", "Positive sample scales need count greater than ddof.", {"column": name})
        else:
            if entry["replacement"] is None:
                raise WrangleError("INVALID_PARAMETERS", "A fitted replacement must be observed and nonnull.", {"column": name})
            _literal(entry["replacement"], schema[name] if method == "mode" else pl.Float64, name, physical=method == "mode")
    return supplied


@operation(returns=("table",), recipe="yes")
def fill(data: pl.LazyFrame, columns: dict) -> pl.LazyFrame:
    """Fill selected null/NaN fields with declared exact scalar values; preserve dtypes.

    columns maps names to JSON scalars. Decimal accepts exact fixed-point strings;
    Date accepts YYYY-MM-DD. Temporal physical values use {dtype:<descriptor>,
    value:<integer>} to retain nanoseconds and zones. Null fields infer only the
    explicit replacement's scalar type; a typed fill can declare that output dtype.
    No text/numeric coercion is inferred.
    """
    data = frame(data)
    if not isinstance(columns, dict) or not columns or any(not isinstance(name, str) or not name for name in columns):
        raise WrangleError("INVALID_ARGUMENT", "columns must map selected fields to explicit replacements.")
    _json(columns)
    require_columns(data, list(columns))
    schema = data.collect_schema()
    expressions = []
    for name, replacement in columns.items():
        dtype = schema[name]
        _descriptor(dtype)
        physical = False
        if isinstance(replacement, dict):
            if set(replacement) != {"dtype", "value"}:
                raise WrangleError("INVALID_FILL_VALUE", "Typed fills declare dtype and value only.", {"column": name})
            declared = _declared_dtype(replacement["dtype"])
            if dtype != pl.Null and declared != dtype:
                raise WrangleError("DTYPE_MISMATCH", "Typed fill literals must match the declared field dtype.", {"column": name})
            dtype = declared
            replacement, physical = replacement["value"], True
        expressions.append(clean(name, dtype).fill_null(_literal(replacement, dtype, name, physical=physical)).alias(name))
    return data.with_columns(expressions)


@operation(returns=("table",), recipe="yes")
def impute(data: pl.LazyFrame, columns: list[str], *, method: str, parameters: dict | None = None, seed: int | None = None) -> pl.LazyFrame:
    """Impute explicit fields using mean/median/mode or seeded uniform mean +/- sample std.

    mean/median/uniform output Float64 after exact input-cast checks. mode retains
    the scalar dtype; tied text/category modes use lexical order, Enum uses its
    declared order, and other scalars use native order. Missing means null
    or floating NaN. Fitting all-missing fields fails; supplied parameters reuse
    the recorded replacements/moments without refitting, and require matching
    input dtypes. uniform requires seed and independent field/row streams; other
    methods reject unused seeds. A singleton uniform fit uses its observed value.
    """
    data = frame(data)
    _fields(data, columns)
    _method(method)
    if method == "uniform":
        seed_value(seed)
    elif seed is not None:
        raise WrangleError("INVALID_SEED", "A seed applies only to uniform imputation.")
    parameters = _parameters(data, columns, parameters, method)
    schema = data.collect_schema()
    expressions = []
    for name in columns:
        entry = parameters[name]
        value = clean(name, schema[name])
        if method == "mode":
            replacement = _literal(entry["replacement"], schema[name], name, physical=True)
        else:
            value = value.cast(pl.Float64)
            if method == "uniform":
                row = pl.int_range(pl.len(), dtype=pl.UInt64)
                unit = pl.struct(row.alias("row"), pl.lit(name).alias("column")).hash(seed=seed).cast(pl.Float64) / float(2**64 - 1)
                replacement = pl.lit(entry["mean"]) + (2 * unit - 1) * pl.lit(entry["std"])
            else:
                replacement = _literal(entry["replacement"], pl.Float64, name)
        expressions.append(value.fill_null(replacement).alias(name))
    result = data.with_columns(expressions)
    invalid = collect(result.select([pl.col(name).is_infinite().sum().alias(name) for name in columns if result.collect_schema()[name].is_float()]))
    if invalid.width and any(invalid.row(0)):
        raise WrangleError("ARITHMETIC_OVERFLOW", "Resolved replacements overflow floating arithmetic.")
    return result


@operation(returns=("table",), recipe="yes")
def standardize(data: pl.LazyFrame, columns: list[str], *, ddof: int, parameters: dict | None = None) -> pl.LazyFrame:
    """Standardize explicit numeric fields using reusable means and standard deviations.

    ddof is a nonnegative integer. Null/NaN remain null; output is Float64 after
    exact input-cast checks. Constant/singleton fits record std=0 and map observed
    fitted values to zero. A frozen zero scale rejects different new observations.
    All-missing fits fail; other nonconstant fits require count > ddof. Supplied
    parameters require the original dtypes and ddof and never refit on a new batch.
    """
    data = frame(data)
    _fields(data, columns)
    _ddof(ddof)
    parameters = _parameters(data, columns, parameters, "standardize", ddof)
    schema = data.collect_schema()
    expressions = []
    for name in columns:
        entry = parameters[name]
        value = clean(name, schema[name]).cast(pl.Float64)
        mean, std = pl.lit(entry["mean"]), pl.lit(entry["std"])
        if entry["std"] == 0:
            changed = collect(data.select((value.is_not_null() & (value != mean)).sum())).item()
            if changed:
                raise WrangleError("ZERO_VARIANCE", "A frozen zero-variance scale cannot standardize different observations.", {"column": name, "affected_rows": changed})
            scaled = pl.lit(0.0)
        else:
            numerator = value - mean
            scaled = pl.when(pl.max_horizontal(value.abs(), mean.abs()) > _SHIFT).then(_ratio(value / _SHIFT - mean / _SHIFT, std / _SHIFT)).otherwise(_ratio(numerator, std))
        expressions.append(pl.when(value.is_null()).then(None).otherwise(scaled).alias(name))
    result = data.with_columns(expressions)
    bad = collect(result.select([pl.col(name).is_infinite().sum().alias(name) for name in columns])).row(0, named=True)
    if any(bad.values()):
        raise WrangleError("ARITHMETIC_OVERFLOW", "Resolved standardization overflows floating arithmetic.", {"columns": bad})
    return result


def _legacy_fields(data, parameters, op):
    schema = data.collect_schema()
    if op == "df_impute_nan":
        reject_destructive(parameters.get("destructive", False))
        mode = parameters.get("impute_mode", "mean_by_std")
        if not isinstance(mode, str) or mode not in {"mean", "median", "mode", "common", "mean_by_std"}:
            raise WrangleError("INVALID_OPTION", "Unknown impute_mode.")
        seed_value(parameters.get("seed", 0))
        selected = list(schema) if parameters.get("cols", "all") == "all" else names(parameters.get("cols"))
        require_columns(data, selected)
        if parameters.get("cols", "all") == "all" and mode not in {"mode", "common"}:
            selected = [name for name in selected if schema[name].is_numeric()]
        return selected, {"common": "mode", "mean_by_std": "uniform"}.get(mode, mode)
    retained = names(parameters.get("retain"))
    require_columns(data, retained)
    _ddof(parameters.get("ddof", 1))
    return [name for name, dtype in schema.items() if dtype.is_numeric() and name not in retained], "standardize"


def _units(data, columns, units):
    if not isinstance(units, dict) or any(not isinstance(name, str) or not name or not isinstance(unit, str) or not unit for name, unit in units.items()):
        raise WrangleError("INVALID_PARAMETERS", "Unit context must map fields to fixed strings or row-unit references.")
    schema, resolved = data.collect_schema(), {}
    for name in columns:
        unit = units.get(name)
        if unit is not None and unit.startswith("@"):
            field = unit[1:]
            if field not in schema or schema[field] != pl.String:
                raise WrangleError("UNIT_MISMATCH", "A statistical row-unit reference requires an observed String field.", {"column": name, "unit_field": field})
            value = pl.col(field)
            state = collect(data.select(value.n_unique().alias("count"), value.is_null().any().alias("missing"), (value == "").any().alias("empty"), value.str.starts_with("@").any().alias("reference"), value.first().alias("unit"))).row(0, named=True)
            if state["count"] != 1 or state["missing"] or state["empty"] or state["reference"]:
                raise WrangleError("UNIT_MISMATCH", "Compute or apply preparation statistics only when every selected row has one fixed physical unit.", {"column": name, "unit_field": field, "distinct_units": state["count"]})
            unit = state["unit"]
        resolved[name] = unit
    return resolved


def _unit_state(parameters, columns, resolved):
    if parameters is None:
        return
    if not isinstance(parameters, dict) or set(parameters) != set(columns):
        raise WrangleError("INVALID_PARAMETERS", "Frozen parameters must name exactly the selected fields.")
    for name in columns:
        entry = parameters[name]
        if not isinstance(entry, dict):
            raise WrangleError("INVALID_PARAMETERS", "Frozen field parameters must be an object.", {"column": name})
        actual = entry.get("unit")
        expected = resolved[name]
        if actual != expected or expected is not None and "unit" not in entry:
            raise WrangleError("UNIT_MISMATCH", "Frozen preparation parameters require the same known physical unit as the current field.", {"column": name, "parameter_unit": actual, "current_unit": expected})


def resolve_parameters(op: str, data: pl.LazyFrame, parameters: dict, *, units: dict | None = None) -> dict:
    """Return JSON keyword arguments containing actual fitted or validated state.

    Handles impute/standardize and their legacy dataframe entrypoints. Each field
    records its exact input dtype and observed count; imputation records a scalar
    replacement or uniform moments, standardization records mean/std/ddof.
    Temporal mode replacements are physical integers; Decimal values are exact
    strings. Supplied state is validated and returned without recomputation.
    units context records a fixed physical unit (or null when undeclared); frozen
    state must match it. Row-unit references require one nonnull unit throughout
    the batch. Known current units never adopt unknown-origin frozen parameters.
    """
    data = frame(data)
    if isinstance(parameters, dict) and op in {"df_impute_nan", "df_rescale_meanzero"}:
        parameters = dict(parameters)
        selector = "cols" if op == "df_impute_nan" else "retain"
        if isinstance(parameters.get(selector), tuple):
            parameters[selector] = list(parameters[selector])
    parameters = _json(parameters)
    if not isinstance(parameters, dict):
        raise WrangleError("INVALID_PARAMETERS", "Operation arguments must be a JSON object.")
    if op not in {"impute", "standardize", "df_impute_nan", "df_rescale_meanzero"}:
        return parameters
    if op in {"df_impute_nan", "df_rescale_meanzero"}:
        columns, method = _legacy_fields(data, parameters, op)
        if op == "df_impute_nan":
            parameters["cols"] = columns
    else:
        columns = parameters.get("columns")
        _fields(data, columns)
        method = parameters.get("method") if op == "impute" else "standardize"
        if method != "standardize":
            _method(method)
            if method == "uniform":
                seed_value(parameters.get("seed"))
            elif parameters.get("seed") is not None:
                raise WrangleError("INVALID_SEED", "A seed applies only to uniform imputation.")
    ddof = parameters.get("ddof", 1)
    if method == "standardize":
        _ddof(ddof)
    if columns:
        resolved_units = _units(data, columns, units) if units is not None else None
        if resolved_units is not None:
            _unit_state(parameters.get("parameters"), columns, resolved_units)
        parameters["parameters"] = _parameters(data, columns, parameters.get("parameters"), method, ddof)
        if resolved_units is not None:
            for name in columns:
                parameters["parameters"][name]["unit"] = resolved_units[name]
    elif parameters.get("parameters") not in (None, {}):
        raise WrangleError("INVALID_PARAMETERS", "No numeric fields were selected for supplied parameters.")
    else:
        parameters["parameters"] = {}
    return parameters


OPERATIONS = {function.__name__: function for function in (fill, impute, standardize)}
