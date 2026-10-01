"""Shared Polars boundaries; no scientific policy is inferred here."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import json
import warnings

import polars as pl


class WrangleError(ValueError):
    """A preparation failure with a stable code and actionable context."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def _fields(records):
    """Keep every string field in first-appearance order, including late fields."""
    fields = dict.fromkeys(name for record in records for name in record)
    if any(not isinstance(name, str) for name in fields):
        raise WrangleError("INVALID_INPUT", "Record fields must have string names; names are never coerced.")
    return list(fields)


def frame(data: Any) -> pl.LazyFrame:
    """Normalize native tables or checked Python columns/rows to a lazy plan.

    Record fields use first-appearance order, absent fields become null, and
    mixed numeric values retain fractional observations. Nested lists/objects
    receive recursively checked native dtypes; unsupported values fail.
    """
    from ._storage import DiskTable
    if isinstance(data, DiskTable):
        return data.lazy()
    if isinstance(data, pl.LazyFrame):
        require_native(data)
        return data
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    if isinstance(data, pl.Series):
        return data.to_frame().lazy()
    try:
        if isinstance(data, Mapping):
            return pl.DataFrame({name: _series(name, values) if isinstance(values, (list, tuple)) else values for name, values in data.items()}).lazy()
        if isinstance(data, (list, tuple)):
            if any(isinstance(row, Mapping) for row in data):
                if not all(isinstance(row, Mapping) for row in data):
                    raise WrangleError("INVALID_INPUT", "Record tables require one mapping per row; use null field values for missingness.")
                fields = _fields(data)
                if not fields:
                    raise WrangleError("INVALID_INPUT", "Rows without any fields cannot retain observation identity; supply at least one named field.")
                return pl.DataFrame([_series(name, [row.get(name) for row in data]) for name in fields]).lazy()
            if any(isinstance(row, (list, tuple)) for row in data):
                if not all(isinstance(row, (list, tuple)) for row in data) or any(len(row) != len(data[0]) for row in data):
                    raise WrangleError("INVALID_INPUT", "Each data row must have the same number of fields.")
                if not data[0]:
                    raise WrangleError("INVALID_INPUT", "Rows without any fields cannot retain observation identity; supply at least one field.")
                return pl.DataFrame([_series(f"column_{i}", [row[i] for row in data]) for i in range(len(data[0]))]).lazy()
            return _series("value", data).to_frame().lazy()
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError, RecursionError) as error:
        if isinstance(error, WrangleError):
            raise
        raise WrangleError("INVALID_INPUT", "Data fields need consistent native types; declare casts explicitly.", {"error": str(error)}) from error
    raise WrangleError("INVALID_INPUT", "Use a Polars table, series, Python records, or a local CSV/Parquet path.")


def finish(plan: pl.LazyFrame | pl.DataFrame, original: Any) -> pl.LazyFrame | pl.DataFrame:
    """Retain laziness when the caller supplied a lazy table."""
    if isinstance(original, pl.LazyFrame):
        return plan if isinstance(plan, pl.LazyFrame) else plan.lazy()
    return plan.collect() if isinstance(plan, pl.LazyFrame) else plan


def columns(data: Any) -> list[str]:
    """Return column names without collecting table rows."""
    return frame(data).collect_schema().names()


def numeric_columns(data: Any) -> list[str]:
    """Return genuinely numeric columns; boolean and identifiers stay explicit."""
    return [name for name, dtype in frame(data).collect_schema().items() if dtype.is_numeric()]


def require_columns(data: Any, names: str | Sequence[str]) -> None:
    """Reject unknown or duplicate column names before execution."""
    requested = [names] if isinstance(names, str) else list(names)
    available = columns(data)
    if len(available) != len(set(available)):
        raise WrangleError("DUPLICATE_COLUMNS", "Column names must be unique.")
    missing = [name for name in requested if name not in available]
    if missing:
        raise WrangleError("UNKNOWN_COLUMN", "The requested columns are absent.", {"columns": missing, "available": available})


def as_series(data: Any) -> pl.Series:
    """Normalize one column; reject accidental multi-column input."""
    if isinstance(data, pl.Series):
        return data.clone()
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        table = frame(data).collect() if isinstance(data, pl.LazyFrame) else data
        if table.width != 1:
            raise WrangleError("INVALID_INPUT", "This operation requires exactly one column.", {"width": table.width})
        return table.to_series().clone()
    if isinstance(data, (list, tuple)):
        try:
            return _series("value", data)
        except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError, RecursionError) as error:
            if isinstance(error, WrangleError):
                raise
            raise WrangleError("INVALID_INPUT", "A column needs a consistent native dtype.", {"error": str(error)}) from error
    raise WrangleError("INVALID_INPUT", "Use a Polars Series, one-column table, or list of values.")


def missing(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Represent absent values explicitly; floating NaN and null are distinct."""
    return expr.is_null() | expr.is_nan() if dtype.is_float() else expr.is_null()


def floating_count(expr: pl.Expr, dtype: pl.DataType, *, kind: str) -> pl.Expr:
    """Count nonfinite floating leaves per row, including native nested fields."""
    if dtype.is_float():
        predicate = expr.is_nan() if kind == "nan" else expr.is_infinite()
        return predicate.fill_null(False).cast(pl.UInt64)
    if isinstance(dtype, pl.Struct):
        fields = [floating_count(expr.struct.field(field.name), field.dtype, kind=kind) for field in dtype.fields]
        return pl.sum_horizontal(fields) if fields else pl.lit(0, dtype=pl.UInt64)
    if isinstance(dtype, pl.List):
        return expr.list.eval(floating_count(pl.element(), dtype.inner, kind=kind)).list.sum().fill_null(0)
    if isinstance(dtype, pl.Array):
        return expr.arr.to_list().list.eval(floating_count(pl.element(), dtype.inner, kind=kind)).list.sum().fill_null(0)
    return pl.lit(0, dtype=pl.UInt64)


def dtype_spec(value) -> pl.DataType:
    """Compile a declared JSON dtype to a native Polars dtype; never infer it."""
    simple = {name: getattr(pl, name) for name in (
        "String", "Boolean", "Int8", "Int16", "Int32", "Int64", "Int128",
        "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "Date", "Datetime", "Time", "Duration", "Null"
    )}
    if isinstance(value, str) and value in simple:
        return simple[value]
    if not isinstance(value, dict) or len(value) != 1:
        raise WrangleError("INVALID_DTYPE", "Declare a native dtype name or one documented dtype descriptor.", {"dtype": value})
    name, parameters = next(iter(value.items()))
    try:
        if name == "List":
            return pl.List(dtype_spec(parameters))
        if name == "Struct" and isinstance(parameters, dict) and all(isinstance(field, str) and field for field in parameters):
            return pl.Struct({field: dtype_spec(dtype) for field, dtype in parameters.items()})
        if name == "Enum" and isinstance(parameters, list) and parameters and all(isinstance(category, str) for category in parameters) and len(parameters) == len(set(parameters)):
            return pl.Enum(parameters)
        if name == "Duration" and parameters in {"ns", "us", "ms"}:
            return pl.Duration(parameters)
        if name == "Datetime" and isinstance(parameters, dict) and not set(parameters) - {"time_unit", "time_zone"}:
            unit, zone = parameters.get("time_unit", "us"), parameters.get("time_zone")
            if unit in {"ns", "us", "ms"} and (zone is None or isinstance(zone, str) and zone):
                return pl.Datetime(unit, zone)
        if name == "Decimal" and isinstance(parameters, dict) and not set(parameters) - {"precision", "scale"}:
            precision, scale = parameters.get("precision", 38), parameters.get("scale", 0)
            if type(precision) is int and type(scale) is int and 1 <= precision <= 38 and 0 <= scale <= precision:
                return pl.Decimal(precision, scale)
        if name == "Array" and isinstance(parameters, dict) and set(parameters) == {"inner", "shape"}:
            shape = parameters["shape"]
            shape = (shape,) if type(shape) is int else tuple(shape) if isinstance(shape, list) else ()
            if shape and all(type(size) is int and size > 0 for size in shape):
                return pl.Array(dtype_spec(parameters["inner"]), shape)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        if isinstance(error, WrangleError):
            raise
        raise WrangleError("INVALID_DTYPE", "The native dtype descriptor is invalid.", {"dtype": value}) from error
    raise WrangleError("INVALID_DTYPE", "Use a documented native dtype descriptor.", {"dtype": value})


def reject_destructive(destructive: bool) -> None:
    """Keep legacy arguments readable while preserving source immutability."""
    if destructive:
        raise WrangleError("IMMUTABLE_INPUT", "Wrangle never mutates input data; use the returned table.")


def _series(name, values):
    """Construct a native column without inference truncation or lost struct fields."""
    observed = [value for value in values if value is not None]
    if any(isinstance(value, Mapping) for value in observed):
        if not all(isinstance(value, Mapping) for value in observed):
            raise WrangleError("INVALID_INPUT", "Nested object fields need consistent native types.", {"column": name})
        fields = _fields(observed)
        dtype = pl.Struct({field: _series(f"{name}.{field}", [None if value is None else value.get(field) for value in values]).dtype for field in fields})
        return pl.Series(name, [None if value is None else dict(value) for value in values], dtype=dtype, strict=True)
    if any(isinstance(value, (list, tuple)) for value in observed):
        if not all(isinstance(value, (list, tuple)) for value in observed):
            raise WrangleError("INVALID_INPUT", "Nested list fields need consistent native types.", {"column": name})
        items = [item for value in observed for item in value]
        dtype = pl.List(_series(f"{name}[]", items).dtype)
        return pl.Series(name, values, dtype=dtype, strict=True)
    numeric = observed and all(type(value) in {int, float} for value in observed)
    mixed_numbers = numeric and any(type(value) is float for value in observed) and any(type(value) is int for value in observed)
    if mixed_numbers and any(type(value) is int and abs(value) > 2**53 for value in observed):
        raise WrangleError("LOSSY_CAST", "Mixed integers and floats exceed the safe Float64 integer range; declare a dtype explicitly.", {"column": name})
    result = pl.Series(name, values, dtype=pl.Float64 if mixed_numbers else None, strict=True)
    if result.dtype == pl.Object:
        raise WrangleError("INVALID_INPUT", "Use native Polars values; opaque Python objects cannot represent research data.", {"column": name})
    return result


def require_native(value):
    """Reject Python callbacks hidden inside an expression or lazy plan."""
    def contains_callback(node):
        if isinstance(node, dict):
            return any(any(marker in key.lower() for marker in ("anonymousfunction", "pythonudf", "pythonfunction", "pythonscan", "opaquepython")) or contains_callback(item) for key, item in node.items())
        if isinstance(node, list):
            return any(contains_callback(item) for item in node)
        return False

    values = value if isinstance(value, (list, tuple)) else [value]
    for item in values:
        if not isinstance(item, (pl.Expr, pl.LazyFrame)):
            raise WrangleError("UNSUPPORTED_CALLBACK", "Use native Polars expressions and plans.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*JSON.*serializ.*", category=UserWarning)
                encoded = item.meta.serialize(format="json") if isinstance(item, pl.Expr) else item.serialize(format="json")
            metadata = json.loads(encoded)
        except (pl.exceptions.PolarsError, TypeError, ValueError, ImportError) as error:
            raise WrangleError("UNSUPPORTED_CALLBACK", "The plan must be native and serializable; Python callbacks cannot execute.") from error
        if contains_callback(metadata):
            raise WrangleError("UNSUPPORTED_CALLBACK", "Python UDFs are not native Polars operations.")


def operation(*, returns, recipe="never", retired=False, aggregates=False):
    """Declare an operation's output and recipe contract on its own definition."""
    from types import MappingProxyType
    kinds = tuple(returns)
    allowed = {"table", "series", "dictionary", "tuple", "sequence", "generator", "scalar", "none"}
    if not kinds or len(kinds) != len(set(kinds)) or not set(kinds) <= allowed or recipe not in {"yes", "conditional", "never"}:
        raise ValueError("Operation definitions require explicit output kinds and recipe eligibility.")
    if recipe != "never" and ("table" not in kinds or retired):
        raise ValueError("Recipe operations must return a Polars table and cannot be retired.")
    contract = MappingProxyType({"returns": kinds, "recipe": recipe, "retired": retired, "aggregates": aggregates})
    def declare(function):
        function.__wrangle_contract__ = contract
        return function
    return declare
