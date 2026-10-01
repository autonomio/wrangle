"""Declared field preparation: native expressions, fixed domains, explicit policies."""
from __future__ import annotations

import math
import inspect as signatures
import re
from typing import Any

import polars as pl

from ._storage import collect
from ._core import WrangleError, _series, frame, operation, require_columns

_TYPES = {name: getattr(pl, name) for name in (
    "String", "Boolean", "Int8", "Int16", "Int32", "Int64", "Int128",
    "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "Null",
)}


def _columns(data, names):
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise WrangleError("INVALID_ARGUMENT", "columns must be a nonempty list of distinct field names.")
    require_columns(data, names)


def _policy(value, choices, name):
    if not isinstance(value, (str, type(None))) or value not in choices:
        raise WrangleError("INVALID_POLICY", f"{name} must be one of {sorted(choices, key=str)!r}.")


def _dtype(name):
    if not isinstance(name, str) or name not in _TYPES:
        raise WrangleError("INVALID_DTYPE", "Declare a supported scalar output dtype.", {"dtype": name, "allowed": list(_TYPES)})
    return _TYPES[name]


def _typed(value, dtype, column):
    """Check a JSON scalar against a field without coercing text or losing bits."""
    if value is None:
        return pl.lit(None, dtype=dtype)
    if type(value) not in {str, int, float, bool} or isinstance(value, float) and not math.isfinite(value):
        raise WrangleError("INVALID_MAPPING", "Codes and mapping values must be finite JSON scalars.", {"column": column})
    text = dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum))
    if text and not isinstance(value, str) or dtype == pl.Boolean and not isinstance(value, bool):
        raise WrangleError("INVALID_MAPPING", "A code's type must match the declared field type.", {"column": column, "dtype": str(dtype)})
    if dtype.is_numeric() and (not isinstance(value, (int, float)) or isinstance(value, bool)) or dtype == pl.Null:
        raise WrangleError("INVALID_MAPPING", "A code's type must match the declared field type.", {"column": column, "dtype": str(dtype)})
    try:
        literal = pl.lit(value)
        original = pl.select(literal).dtypes[0]
        cast = literal.cast(dtype, strict=False)
        same = pl.select(literal.eq_missing(cast.cast(original, strict=False))).item()
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as error:
        raise WrangleError("INVALID_MAPPING", "The code cannot be represented in the declared field type.", {"column": column, "dtype": str(dtype)}) from error
    if not same:
        raise WrangleError("LOSSY_CAST", "The code cannot be represented exactly in the declared field type.", {"column": column, "dtype": str(dtype)})
    return literal.cast(dtype, strict=True)


def _values(values, dtype, column):
    expressions = [_typed(value, dtype, column) for value in values]
    if not expressions:
        return pl.Series("value", [], dtype=dtype)
    kwargs = {"empty_as_null": False} if "empty_as_null" in signatures.signature(pl.Expr.explode).parameters else {}
    return pl.select(pl.concat_list(expressions).explode(**kwargs).alias("value")).to_series()


def _input(data, column, declared):
    require_columns(data, column)
    dtype = data.collect_schema()[column]
    if dtype == pl.Null:
        try:
            dtype = _series("domain", declared).dtype
        except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
            if isinstance(error, WrangleError):
                raise
            raise WrangleError("INVALID_MAPPING", "The declared domain needs one consistent scalar type.", {"column": column}) from error
    if dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return pl.col(column).cast(pl.String), pl.String
    if not dtype.is_numeric() and dtype not in {pl.Boolean, pl.Null}:
        raise WrangleError("INVALID_DTYPE", "This operation requires a scalar string, numeric, or boolean field.", {"column": column, "dtype": str(dtype)})
    expression = pl.col(column).cast(dtype) if data.collect_schema()[column] == pl.Null else pl.col(column)
    return expression.fill_nan(None) if dtype.is_float() else expression, dtype


def _known(value, keys):
    return value.is_in(pl.lit(keys).implode(), nulls_equal=True)


def _check_policy(data, value, keys, column, unknown, nulls):
    bad = value.is_not_null() & ~_known(value, keys)
    expressions = [bad.sum().alias("unknown"), value.is_null().sum().alias("nulls")]
    counts = collect(data.select(expressions)).row(0, named=True)
    if counts["unknown"] and unknown == "error":
        raise WrangleError("UNKNOWN_CATEGORY", "Observed values are absent from the declared mapping or domain.", {"column": column, "affected_rows": counts["unknown"]})
    if counts["nulls"] and nulls == "error":
        raise WrangleError("MISSING_CATEGORY", "Missing category values require an explicit permitted null policy.", {"column": column, "affected_rows": counts["nulls"]})
    return bad


def _pairs(mapping):
    if isinstance(mapping, dict) and mapping and all(isinstance(key, str) for key in mapping):
        return list(mapping.items())
    if isinstance(mapping, list) and mapping and all(isinstance(pair, list) and len(pair) == 2 for pair in mapping):
        return mapping
    raise WrangleError("INVALID_MAPPING", "Use a nonempty string-keyed object or list of typed [old, new] pairs.")



def _normalize_nan(value, dtype):
    if dtype.is_float():
        return value.fill_nan(None)
    if isinstance(dtype, pl.Struct):
        fields = [_normalize_nan(value.struct.field(field.name), field.dtype).alias(field.name) for field in dtype.fields]
        return pl.when(value.is_null()).then(pl.lit(None, dtype=dtype)).otherwise(pl.struct(fields)) if fields else value
    if isinstance(dtype, pl.List):
        return value.list.eval(_normalize_nan(pl.element(), dtype.inner))
    if isinstance(dtype, pl.Array):
        return value.arr.to_list().list.eval(_normalize_nan(pl.element(), dtype.inner)).cast(dtype)
    return value

@operation(returns=("table",), recipe="yes")
def normalize_missing(data: pl.LazyFrame, columns: dict[str, list[Any]], *, nan: bool = True) -> pl.LazyFrame:
    """Convert explicitly declared, exactly typed codes to null in selected fields.

    columns maps each field to its literal missing codes, e.g. {"mass":[-999],
    "note":["NA",""]}. No trimming, parsing, or case folding is inferred.
    nan=True also normalizes floating NaN leaves in selected scalar/nested fields.
    Nested fields use codes=[]; explicit scalar codes require scalar fields. Null stays
    null; other values, field dtypes, row identities, and order remain intact.
    """
    data = frame(data)
    if not isinstance(columns, dict) or not columns or any(not isinstance(name, str) or not name for name in columns):
        raise WrangleError("INVALID_ARGUMENT", "columns must map field names to lists of literal missing codes.")
    if not isinstance(nan, bool):
        raise WrangleError("INVALID_ARGUMENT", "nan must be true or false.")
    require_columns(data, list(columns))
    schema = data.collect_schema()
    expressions = []
    for name, codes in columns.items():
        if not isinstance(codes, list):
            raise WrangleError("INVALID_ARGUMENT", "Each field's missing codes must be a list.", {"column": name})
        value = pl.col(name)
        if codes and isinstance(schema[name], (pl.List, pl.Array, pl.Struct)):
            raise WrangleError("INVALID_MAPPING", "Nested fields use codes=[] for NaN normalization; map scalar codes after explicit reshape.", {"column": name})
        typed = _values(codes, schema[name], name)
        absent = _known(value, typed) if codes else pl.lit(False)
        normalized = _normalize_nan(value, schema[name]) if nan else value
        expressions.append(pl.when(absent).then(pl.lit(None, dtype=schema[name])).otherwise(normalized).alias(name))
    return data.with_columns(expressions)


@operation(returns=("table",), recipe="yes")
def clean_text(data: pl.LazyFrame, columns: list[str], *, trim: bool = True, case: str | None = None, unicode: str | None = None) -> pl.LazyFrame:
    """Clean selected String fields only: Unicode normalization, trim, then case.

    case is None/lower/upper; unicode is None/NFC/NFD/NFKC/NFKD. Null stays null.
    Values are never parsed or recoded; unselected fields remain unchanged.
    Categorical/Enum fields require an explicit cast or recode first. Changing
    a declared observation key requires an explicit preparation step.key transition.
    """
    data = frame(data)
    _columns(data, columns)
    if not isinstance(trim, bool):
        raise WrangleError("INVALID_ARGUMENT", "trim must be true or false.")
    _policy(case, {None, "lower", "upper"}, "case")
    _policy(unicode, {None, "NFC", "NFD", "NFKC", "NFKD"}, "unicode")
    schema = data.collect_schema()
    bad = [name for name in columns if schema[name] != pl.String]
    if bad:
        raise WrangleError("INVALID_DTYPE", "Text cleaning requires selected String fields; cast explicitly first.", {"columns": bad})
    expressions = []
    for name in columns:
        value = pl.col(name)
        if unicode is not None:
            value = value.str.normalize(unicode)
        if trim:
            value = value.str.strip_chars()
        if case == "lower":
            value = value.str.to_lowercase()
        elif case == "upper":
            value = value.str.to_uppercase()
        expressions.append(value.alias(name))
    return data.with_columns(expressions)


def _fraction_regex(format):
    """Locate a variable-width fraction from format metadata, never row Python."""
    if "%+" in format:
        return r"\d{2}:\d{2}:\d{2}\.([0-9]+)"
    if "%.f" not in format:
        return None
    if format.count("%.f") != 1:
        raise WrangleError("INVALID_DATETIME_FORMAT", "Declare exactly one variable fractional-second field.")
    pattern, position = "^", 0
    for token in re.finditer(r"%(?:[:#]*z|[-_0]?(?:\d+|\.\d*)?[A-Za-z+%])", format):
        pattern += re.escape(format[position:token.start()])
        if token.group() == "%.f":
            pattern += r"(?:\.([0-9]+))?"
        elif token.group() == "%%":
            pattern += "%"
        else:
            pattern += ".*?"
        position = token.end()
    return pattern + re.escape(format[position:]) + "$"


def _datetime_counts(data, column, expressions):
    try:
        return collect(data.select(expressions)).row(0, named=True)
    except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
        raise WrangleError("INVALID_DATETIME", "Datetime format, time zone, or DST policy cannot parse the observed values.", {"column": column, "error": str(error)}) from error


@operation(returns=("table",), recipe="yes")
def parse_datetime(data: pl.LazyFrame, columns: list[str], format: str, *, dtype: str = "Datetime", time_zone: str | None = None, time_unit: str = "us", ambiguous: str = "raise", nonexistent: str = "raise", invalid: str = "error", precision_loss: str = "error") -> pl.LazyFrame:
    """Parse selected String fields using one exact declared Chrono format.

    dtype is Date or Datetime. Naive text stays naive when time_zone=None;
    otherwise the explicit IANA zone localizes wall time. Offset-bearing formats
    require time_zone and preserve instants while converting to that output zone.
    ambiguous=raise/earliest/latest/null; nonexistent=raise/null apply to naive
    text localized in a zone; nondefault DST policies fail elsewhere. invalid=error/null
    controls malformed observations; existing nulls remain null. time_unit is
    ns/us/ms. Precision loss fails unless precision_loss=truncate. Nanosecond
    overflow is checked against a wider native parser; it never wraps a date.
    Leap seconds fail or become null under invalid policy; they never shift into
    another date. Date rejects zones and nondefault time-unit/DST/precision policies.
    """
    data = frame(data)
    _columns(data, columns)
    if not isinstance(format, str) or not format:
        raise WrangleError("INVALID_DATETIME_FORMAT", "Declare a nonempty exact datetime format.")
    effective = format.replace("%%", "")
    if "%Z" in effective:
        raise WrangleError("INVALID_DATETIME_FORMAT", "Use numeric offsets or an explicit IANA zone; timezone abbreviations are ambiguous.")
    _policy(dtype, {"Date", "Datetime"}, "dtype")
    _policy(time_unit, {"ns", "us", "ms"}, "time_unit")
    _policy(ambiguous, {"raise", "earliest", "latest", "null"}, "ambiguous")
    _policy(nonexistent, {"raise", "null"}, "nonexistent")
    _policy(invalid, {"error", "null"}, "invalid")
    _policy(precision_loss, {"error", "truncate"}, "precision_loss")
    if time_zone is not None and (not isinstance(time_zone, str) or not time_zone):
        raise WrangleError("INVALID_DATETIME_FORMAT", "time_zone must be an explicit nonempty IANA zone or None.")
    offset = bool(re.search(r"%(?:[:#]*z|\+)", effective))
    if dtype == "Date" and (time_zone is not None or time_unit != "us" or ambiguous != "raise" or nonexistent != "raise" or precision_loss != "error" or offset):
        raise WrangleError("INVALID_POLICY", "Date parsing cannot apply timezone, datetime precision, or DST policies.")
    if offset and time_zone is None:
        raise WrangleError("UNDECLARED_TIME_ZONE", "Offset-bearing text requires an explicit output time_zone.")
    if (offset or time_zone is None) and (ambiguous != "raise" or nonexistent != "raise"):
        raise WrangleError("INVALID_POLICY", "Nondefault DST policies require naive text and an explicit localization time_zone.")
    schema = data.collect_schema()
    bad = [name for name in columns if schema[name] != pl.String]
    if bad:
        raise WrangleError("INVALID_DTYPE", "Datetime parsing requires String fields.", {"columns": bad})
    expressions = []
    for name in columns:
        source = pl.col(name)
        target = pl.Date if dtype == "Date" else pl.Datetime(time_unit)
        parsed = source.str.strptime(target, format, strict=False, exact=True)
        invalid_rows = source.is_not_null() & parsed.is_null()
        checks = []
        if dtype == "Datetime":
            clock = source.str.strptime(pl.Time, format, strict=False, exact=True)
            if re.search(r"%[HIMSTRrXc+]", effective):
                # Chrono Datetime normalizes leap seconds; its Time parser rejects them.
                unsupported_clock = parsed.is_not_null() & clock.is_null()
                invalid_rows = invalid_rows | unsupported_clock
                parsed = pl.when(unsupported_clock).then(None).otherwise(parsed)
            if precision_loss == "error":
                quantum = {"ns": 1, "us": 1000, "ms": 1_000_000}[time_unit]
                lost = ((clock.dt.nanosecond() % quantum) != 0).fill_null(False)
                fraction_pattern = _fraction_regex(format)
                if fraction_pattern is not None:
                    fraction = source.str.extract(fraction_pattern, 1)
                    lost = lost | fraction.str.slice(9).str.contains(r"[1-9]").fill_null(False)
                checks.append((lost & parsed.is_not_null()).sum().alias("precision"))
            if time_unit == "ns":
                broad = source.str.strptime(pl.Datetime("us"), format, strict=False, exact=True)
                checks.append((parsed.is_not_null() & broad.is_not_null() & ~parsed.cast(pl.Datetime("us", "UTC" if offset else None)).eq_missing(broad)).sum().alias("range"))
        checks.append(invalid_rows.sum().alias("invalid"))
        counts = _datetime_counts(data, name, checks)
        if counts["invalid"] and invalid == "error":
            raise WrangleError("INVALID_DATETIME", "Observed text does not match the declared datetime format.", {"column": name, "affected_rows": counts["invalid"]})
        if counts.get("precision"):
            raise WrangleError("DATETIME_PRECISION_LOSS", "Observed fractions exceed the declared time precision; choose a finer unit or explicitly truncate.", {"column": name, "affected_rows": counts["precision"]})
        if counts.get("range"):
            raise WrangleError("DATETIME_RANGE", "Observed timestamps exceed exact native nanosecond range.", {"column": name, "affected_rows": counts["range"]})
        if dtype == "Datetime" and time_zone is not None:
            try:
                parsed = parsed.dt.convert_time_zone(time_zone) if offset else parsed.dt.replace_time_zone(time_zone, ambiguous=ambiguous, non_existent=nonexistent)
            except (pl.exceptions.PolarsError, TypeError, ValueError) as error:
                raise WrangleError("INVALID_TIME_ZONE", "Declare a valid IANA output time zone.", {"column": name, "time_zone": time_zone}) from error
            _datetime_counts(data, name, [parsed.is_null().sum().alias("nulls")])
        expressions.append(parsed.alias(name))
    return data.with_columns(expressions)


@operation(returns=("table",), recipe="yes")
def recode(data: pl.LazyFrame, column: str, mapping: dict[str, Any] | list[list[Any]], *, dtype: str, unknown: str = "error", nulls: str = "preserve", domain: list[Any] | None = None) -> pl.LazyFrame:
    """Apply a declared typed mapping; no categories or missing codes are inferred.

    mapping is a string-keyed object or typed [old,new] pairs. dtype declares the
    scalar output type. unknown=error/keep/null; keep requires unchanged dtype.
    nulls=preserve/error/map; map requires a literal null source key, and other
    policies reject such a key. Floating NaN follows the null policy. An optional
    domain declares allowed nonnull outputs. Source and target codes must be
    exactly representable; duplicate source codes fail. Rows/order stay intact.
    """
    data = frame(data)
    _policy(unknown, {"error", "keep", "null"}, "unknown")
    _policy(nulls, {"preserve", "error", "map"}, "nulls")
    pairs = _pairs(mapping)
    old, new = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    value, source_dtype = _input(data, column, old)
    target = _dtype(dtype)
    keys, replacements = _values(old, source_dtype, column), _values(new, target, column)
    if keys.n_unique() != len(keys):
        raise WrangleError("DUPLICATE_MAPPING", "Each source code must appear exactly once.", {"column": column})
    has_null = keys.null_count() > 0
    if has_null != (nulls == "map"):
        raise WrangleError("INVALID_POLICY", "A null mapping key requires nulls=map, and nulls=map requires that key.", {"column": column})
    if unknown == "keep" and target != source_dtype:
        raise WrangleError("INVALID_POLICY", "Keeping unmapped values requires the same source and output dtype.", {"column": column})
    _check_policy(data, value, keys, column, unknown, nulls)
    default = value if unknown == "keep" else pl.lit(None, dtype=target)
    result = value.replace_strict(keys, replacements, default=default, return_dtype=target)
    if nulls == "preserve":
        result = pl.when(value.is_null()).then(pl.lit(None, dtype=target)).otherwise(result)
    if domain is not None:
        if not isinstance(domain, list) or not domain or any(item is None for item in domain):
            raise WrangleError("INVALID_MAPPING", "domain must be a nonempty list of allowed nonnull output values.")
        allowed = _values(domain, target, column)
        if allowed.n_unique() != len(allowed):
            raise WrangleError("DUPLICATE_MAPPING", "Output domain values must be unique.", {"column": column})
        count = collect(data.select((result.is_not_null() & ~_known(result, allowed)).sum())).item()
        if count:
            raise WrangleError("CATEGORY_DOMAIN", "Recoded values are outside the declared output domain.", {"column": column, "affected_rows": count})
    return data.with_columns(result.alias(column))


@operation(returns=("table",), recipe="yes")
def encode(data: pl.LazyFrame, column: str, mapping: dict[str, int] | list[list[Any]] | None = None, *, categories: list[Any] | None = None, names: list[str] | None = None, mode: str = "ordinal", unknown: str = "error", nulls: str = "preserve", drop: bool = True) -> pl.LazyFrame:
    """Encode a fixed declared domain; codes and generated indicators never refit.

    Ordinal mode uses distinct integer mapping codes, or categories in their
    declared order with codes from zero. nulls=preserve/error/category; category
    requires an explicit null level/key. One-hot mode requires categories and
    exactly matching unique output names; categories absent in a batch still
    produce zero columns. Missing values yield null indicators unless an explicit
    null category is declared. unknown=error/null; unknown-null yields null codes
    or indicators. drop applies only to one-hot source retention. Row order stays.
    """
    data = frame(data)
    _policy(mode, {"ordinal", "one_hot"}, "mode")
    _policy(unknown, {"error", "null"}, "unknown")
    _policy(nulls, {"preserve", "error", "category"}, "nulls")
    if not isinstance(drop, bool):
        raise WrangleError("INVALID_ARGUMENT", "drop must be true or false.")
    if mode == "ordinal":
        if names is not None or (mapping is None) == (categories is None):
            raise WrangleError("INVALID_MAPPING", "Ordinal encoding requires exactly one mapping or categories declaration, without one-hot names.")
        if categories is not None:
            if not isinstance(categories, list) or not categories:
                raise WrangleError("INVALID_MAPPING", "categories must be a nonempty list of literal levels.")
            mapping = [[category, index] for index, category in enumerate(categories)]
        pairs = _pairs(mapping)
        codes = [pair[1] for pair in pairs]
        if any(type(code) is not int for code in codes) or len(set(codes)) != len(codes):
            raise WrangleError("INVALID_MAPPING", "Ordinal codes must be distinct integers.")
        return recode(data, column, mapping, dtype="Int64", unknown=unknown, nulls="map" if nulls == "category" else nulls)
    if mapping is not None or not isinstance(categories, list) or not categories:
        raise WrangleError("INVALID_MAPPING", "One-hot encoding requires a fixed nonempty categories list and no ordinal mapping.")
    _columns(data, [column])
    if not isinstance(names, list) or len(names) != len(categories) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise WrangleError("INVALID_MAPPING", "One-hot names must provide one distinct nonempty name per declared category.")
    existing = set(data.collect_schema().names()) - ({column} if drop else set())
    if set(names) & existing:
        raise WrangleError("COLUMN_COLLISION", "One-hot outputs would overwrite existing fields.", {"columns": sorted(set(names) & existing)})
    value, source_dtype = _input(data, column, categories)
    levels = _values(categories, source_dtype, column)
    if levels.n_unique() != len(levels):
        raise WrangleError("DUPLICATE_MAPPING", "Category levels must be unique.", {"column": column})
    if bool(levels.null_count()) != (nulls == "category"):
        raise WrangleError("INVALID_POLICY", "A null category requires nulls=category, and that policy requires an explicit null level.")
    unknown_rows = _check_policy(data, value, levels, column, unknown, nulls)
    absent = unknown_rows | (value.is_null() if nulls == "preserve" else pl.lit(False))
    indicators = [pl.when(absent).then(pl.lit(None, dtype=pl.UInt8)).otherwise(value.eq_missing(_typed(category, source_dtype, column)).cast(pl.UInt8)).alias(name) for category, name in zip(categories, names)]
    retained = [pl.col(name) for name in data.collect_schema().names() if name != column or not drop]
    return data.select(*retained, *indicators)


OPERATIONS = {
    "normalize_missing": normalize_missing,
    "clean_text": clean_text,
    "parse_datetime": parse_datetime,
    "recode": recode,
    "encode": encode,
}
