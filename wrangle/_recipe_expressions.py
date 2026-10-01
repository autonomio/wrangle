"""Bounded JSON expressions, compiled only to native Polars expressions."""
from __future__ import annotations

import math
import re
import polars as pl
from ._storage import collect
from ._core import WrangleError, _series, dtype_spec


EXPRESSION_DEFINITIONS = {
    "when": {"shape": {"if": "expression", "then": "expression", "else": "expression", "nulls": ["error", "else", "null"]}, "meaning": "Choose a branch; every branch must be valid independently."},
    "case": {"shape": {"branches": [{"when": "expression", "then": "expression"}], "else": "expression", "nulls": ["error", "skip", "null"]}, "meaning": "First true branch wins; null policy is explicit."},
    "coalesce": {"shape": ["expression", "expression"], "meaning": "First nonnull expression in declared order; all-null returns null."},
    "is_in": {"shape": {"value": "expression", "values": ["JSON scalar"], "nulls": ["error", "false", "null", "match"]}, "meaning": "Exact typed set membership; null matching is explicit."},
    "literal": {"shape": {"value": "JSON value", "dtype": "dtype descriptor"}, "meaning": "Exactly typed native literal; temporal strings use ISO syntax."},
    "cast": {"shape": {"value": "expression", "dtype": "dtype descriptor", "invalid": ["error", "null"], "precision": ["exact", "allow"]}, "meaning": "Declared dtype conversion; temporal text requires parse_datetime instead."},
    "trim": {"shape": "expression", "meaning": "Trim outer Unicode whitespace; String input, null remains null."},
    "lower": {"shape": "expression", "meaning": "Unicode lowercase; String input, null remains null."},
    "upper": {"shape": "expression", "meaning": "Unicode uppercase; String input, null remains null."},
    "length": {"shape": "expression", "meaning": "Unicode character count, not bytes; null remains null."},
    "slice": {"shape": {"value": "expression", "offset": "integer", "length": "nonnegative integer"}, "meaning": "Bounded character slice; negative offset counts from the end."},
    "contains": {"shape": {"value": "expression", "pattern": "string", "literal": "boolean"}, "meaning": "Explicit literal or Rust-regex search; null remains null."},
    "replace": {"shape": {"value": "expression", "pattern": "string", "replacement": "string", "literal": "boolean", "all": "boolean", "capture_groups": "boolean"}, "meaning": "Replace first/all matches; replacement captures require explicit permission."},
    "extract": {"shape": {"value": "expression", "pattern": "string", "group": "nonnegative integer"}, "meaning": "Extract a Rust-regex capture; no match returns null."},
    "concat": {"shape": {"values": ["String expression"], "separator": "string", "nulls": ["error", "propagate", "ignore"]}, "meaning": "Concatenate strings; ignore with all-null inputs produces empty string."},
    "date_part": {"shape": {"value": "expression", "part": ["year", "iso_year", "quarter", "month", "week", "day", "weekday", "ordinal_day", "hour", "minute", "second"]}, "meaning": "Native temporal component; weekday is ISO Monday=1 through Sunday=7; timestamp zone is retained."},
    "split": {"shape": {"value": "String expression", "separator": "nonempty literal string", "max_splits": "integer 0..128", "nulls": ["error", "keep"]}, "meaning": "At most max_splits literal splits; final element retains the remainder; empty elements remain."},
    "convert_time_zone": {"shape": {"value": "zoned Datetime expression", "time_zone": "IANA zone string"}, "meaning": "Represent the same instant in a declared zone; naive timestamps require explicit localization first."},
    "datetime_shift": {"shape": {"value": "Date or Datetime expression", "offset": "signed integer plus ns/us/ms/s/m/h/d/w/mo/q/y", "mode": ["calendar", "elapsed"], "precision": ["exact", "allow"], "month_end": ["error", "clip"]}, "meaning": "Calendar accepts d/w/mo/q/y; elapsed uses fixed durations. Calendar DST ambiguity fails; month clipping and duration truncation require explicit permission. Null remains null; representational overflow fails."},
    "round": {"shape": {"value": "expression", "decimals": "integer 0..38", "mode": ["half_to_even", "half_away_from_zero"]}, "meaning": "Explicit decimal rounding rule; null remains null."},
    "clip": {"shape": {"value": "expression", "min": "expression or null for unbounded", "max": "expression or null for unbounded", "nulls": ["error", "keep"]}, "meaning": "Clip exact numeric bounds; missing/reversed bounds fail."},
}


def _fail(code, message, **details):
    raise WrangleError(code, message, details)


def _shape(arguments, fields, operator):
    if not isinstance(arguments, dict) or set(arguments) != set(fields):
        _fail("INVALID_EXPRESSION", f"{operator} requires exactly its documented fields.", expected=list(fields))


def _choice(value, values, name):
    if value not in values:
        _fail("INVALID_EXPRESSION", f"{name} requires one of {', '.join(values)}.")


def _dtype(data, expr):
    return data.select(expr.alias("__wrangle_expression")).collect_schema().dtypes()[0]


def _count(data, predicate):
    return collect(data.select(predicate.fill_null(False).sum())).item()


def _nonnull(data, expr):
    count = _count(data, expr.is_null())
    if count:
        _fail("UNRESOLVED_EXPRESSION", "Missing expression values require an explicit permitted null policy.", affected_rows=count)


def _cast(data, expr, target, *, invalid="error", precision="exact"):
    _choice(invalid, ("error", "null"), "cast invalid")
    _choice(precision, ("exact", "allow"), "cast precision")
    original = _dtype(data, expr)
    if original == pl.String and target.is_temporal():
        _fail("INVALID_EXPRESSION", "Parse temporal text with parse_datetime and an exact format before deriving components.")
    converted = expr.cast(target, strict=False)
    failures = _count(data, expr.is_not_null() & converted.is_null())
    if failures and invalid == "error":
        _fail("INVALID_CAST", "Some observed values cannot be represented in the requested dtype.", affected_rows=failures)
    if precision == "exact" and original != target:
        # Full native roundtrip equality also detects nested numeric precision
        # loss, numeric narrowing and temporal unit truncation.
        numeric = (original.is_numeric() or original == pl.Boolean) and (target.is_numeric() or target == pl.Boolean)
        temporal = original.is_temporal() and target.is_temporal()
        nested = isinstance(original, (pl.List, pl.Array, pl.Struct))
        if numeric or temporal or nested:
            restored = converted.cast(original, strict=False)
            changed = expr.is_not_null() & converted.is_not_null() & ~expr.eq_missing(restored)
            losses = _count(data, changed)
            if losses:
                _fail("LOSSY_CAST", "Expression conversion changes observed precision; declare precision='allow' deliberately.", affected_rows=losses)
    return expr.cast(target, strict=invalid == "error")


def _compatible(data, expressions):
    dtypes = [_dtype(data, expr) for expr in expressions]
    observed = [dtype for dtype in dtypes if dtype != pl.Null]
    if not observed:
        return expressions
    if len(set(observed)) == 1:
        target = observed[0]
    elif all(dtype.is_numeric() for dtype in observed):
        target = _dtype(data, pl.coalesce(expressions))
    else:
        _fail("DTYPE_MISMATCH", "Branches need compatible native types; declare casts explicitly.", dtypes=[str(dtype) for dtype in dtypes])
    return [_cast(data, expr, target) if dtype != target else expr for expr, dtype in zip(expressions, dtypes)]


def _predicate(data, expr, policy):
    if _dtype(data, expr) not in (pl.Boolean, pl.Null):
        _fail("INVALID_EXPRESSION", "Conditional predicates must be Boolean.")
    if policy == "error":
        _nonnull(data, expr)
    return expr.cast(pl.Boolean)


def _finite_literal(value):
    if isinstance(value, float) and not math.isfinite(value):
        _fail("INVALID_EXPRESSION", "Typed literals must contain finite JSON values.")
    if isinstance(value, list):
        for item in value:
            _finite_literal(item)
    elif isinstance(value, dict):
        for item in value.values():
            _finite_literal(item)
    elif value is not None and type(value) not in (str, int, float, bool):
        _fail("INVALID_EXPRESSION", "Typed literals contain JSON values only.")


def _literal(data, arguments):
    _shape(arguments, ("value", "dtype"), "literal")
    value, dtype = arguments["value"], dtype_spec(arguments["dtype"])
    _finite_literal(value)
    if isinstance(value, str) and (dtype == pl.Date or isinstance(dtype, pl.Datetime)):
        from ._recipe_fields import parse_datetime
        date_only = dtype == pl.Date
        if date_only:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
                _fail("INVALID_EXPRESSION", "Date literals require ISO YYYY-MM-DD syntax.")
            options = dict(dtype="Date")
            format = "%Y-%m-%d"
        else:
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})?", value):
                _fail("INVALID_EXPRESSION", "Datetime literals require exact ISO date/time syntax.")
            offset = bool(re.search(r"(?:Z|[+-][0-9]{2}:[0-9]{2})$", value))
            options = dict(time_unit=dtype.time_unit, time_zone=dtype.time_zone)
            format = "%+" if offset else "%Y-%m-%dT%H:%M:%S%.f"
        result = collect(parse_datetime(pl.DataFrame({"literal": [value]}).lazy(), ["literal"], format, **options)).to_series()
        return pl.lit(result).first()
    source = _series("literal", [value])
    if dtype.is_numeric() and source.dtype in (pl.String, pl.Boolean) or dtype == pl.Boolean and source.dtype.is_numeric():
        _fail("INVALID_EXPRESSION", "Literal value type must agree with its declared numeric/boolean dtype; use cast for conversion.")
    # Literal validity is independent of whether the observation table is empty.
    _cast(source.to_frame().lazy(), pl.col("literal"), dtype)
    return pl.lit(source).first().cast(dtype, strict=True)


def _string(data, expr):
    if _dtype(data, expr) not in (pl.String, pl.Null):
        _fail("INVALID_EXPRESSION", "Text operations require String expressions; cast explicitly.")
    return expr.cast(pl.String)


def _pattern(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        _fail("INVALID_EXPRESSION", "A pattern requires 1..4096 literal characters.")


def _regex_groups(pattern):
    # Native Rust regex parsing supplies actual capture metadata; Python never
    # compiles user patterns or evaluates them against observation strings.
    dtype = pl.select(pl.lit("").str.extract_groups(pattern).alias("captures")).schema["captures"]
    return {"0", *(str(i) for i in range(1, len(dtype.fields) + 1)), *(field.name for field in dtype.fields)}


def _replacement_groups(value, known):
    position = 0
    while position < len(value):
        if value[position] != "$":
            position += 1
            continue
        if value[position:position + 2] == "$$":
            position += 2
            continue
        if value[position:position + 2] == "${":
            end = value.find("}", position + 2)
            if end < 0:
                _fail("INVALID_EXPRESSION", "Replacement capture reference is unclosed.")
            name, position = value[position + 2:end], end + 1
        else:
            match = re.match(r"[A-Za-z0-9_]+", value[position + 1:])
            if not match:
                position += 1
                continue
            name = match.group()
            position += len(name) + 1
        if name not in known:
            _fail("INVALID_EXPRESSION", "Replacement refers to an absent regex capture.", capture=name)




def _datetime_shift(data, expr, arguments):
    _shape(arguments, ("value", "offset", "mode", "precision", "month_end"), "datetime_shift")
    _choice(arguments["mode"], ("calendar", "elapsed"), "datetime_shift mode")
    _choice(arguments["precision"], ("exact", "allow"), "datetime_shift precision")
    _choice(arguments["month_end"], ("error", "clip"), "datetime_shift month_end")
    offset = arguments["offset"]
    parsed = re.fullmatch(r"([+-]?[0-9]+)(ns|us|ms|s|m|h|d|w|mo|q|y)", offset) if isinstance(offset, str) else None
    if not parsed:
        _fail("INVALID_EXPRESSION", "datetime_shift requires one signed integer duration and a documented unit.")
    amount, unit = int(parsed[1]), parsed[2]
    if abs(amount) > 2**63 - 1:
        _fail("INVALID_EXPRESSION", "datetime_shift duration magnitude exceeds the supported integer domain.")
    dtype = _dtype(data, expr)
    if dtype != pl.Date and not isinstance(dtype, pl.Datetime):
        _fail("INVALID_EXPRESSION", "datetime_shift requires Date or Datetime observations.")
    if arguments["mode"] == "elapsed":
        factors = {"ns": 1, "us": 1000, "ms": 1000000, "s": 1000000000, "m": 60000000000, "h": 3600000000000, "d": 86400000000000, "w": 604800000000000}
        if unit not in factors or dtype == pl.Date and unit not in ("d", "w") or arguments["month_end"] != "error":
            _fail("INVALID_EXPRESSION", "Elapsed shifts require a fixed duration; Date accepts d/w and month_end must be error.")
        tick = 86400000000000 if dtype == pl.Date else factors[dtype.time_unit]
        duration = amount * factors[unit]
        if duration % tick and arguments["precision"] == "exact":
            _fail("LOSSY_CAST", "The elapsed duration cannot be represented in the timestamp unit; declare precision='allow' to truncate toward zero.")
        delta = abs(duration) // tick * (-1 if duration < 0 else 1)
        shifted = expr.cast(pl.Int64).cast(pl.Int128) + pl.lit(delta, dtype=pl.Int128)
        lower, upper = (-2**31, 2**31 - 1) if dtype == pl.Date else (-2**63, 2**63 - 1)
        if _count(data, (shifted < lower) | (shifted > upper)):
            _fail("DATETIME_OVERFLOW", "The shifted observation exceeds its declared temporal representation.")
        return shifted.cast(pl.Int32 if dtype == pl.Date else pl.Int64).cast(dtype)
    if unit not in ("d", "w", "mo", "q", "y"):
        _fail("INVALID_EXPRESSION", "Calendar shifts accept d/w/mo/q/y; use elapsed for fixed subday durations.")
    day = expr if dtype == pl.Date else expr.dt.date()
    if unit in ("d", "w"):
        physical = day.cast(pl.Int32).cast(pl.Int128) + pl.lit(amount * (7 if unit == "w" else 1), dtype=pl.Int128)
        # Native calendar methods use chrono's supported year domain. Bounds
        # are metadata expressions, never Python conversions of observations.
        low = pl.date(-262143, 1, 1).cast(pl.Int32).cast(pl.Int128)
        high = pl.date(262142, 12, 31).cast(pl.Int32).cast(pl.Int128)
        if _count(data, (physical < low) | (physical > high)):
            _fail("DATETIME_OVERFLOW", "Calendar shift exceeds the native calendar's supported year domain.")
        expected = physical.cast(pl.Int32).cast(pl.Date)
    else:
        months = amount * {"mo": 1, "q": 3, "y": 12}[unit]
        index = day.dt.year().cast(pl.Int128) * 12 + day.dt.month().cast(pl.Int128) - 1 + pl.lit(months, dtype=pl.Int128)
        year, month = index // 12, index % 12 + 1
        if _count(data, (year < -262143) | (year > 262142)):
            _fail("DATETIME_OVERFLOW", "Calendar shift exceeds the native calendar's supported year domain.")
        first = pl.date(year.cast(pl.Int32), month.cast(pl.Int32), 1)
        final_day = first.dt.month_end().dt.day()
        if arguments["month_end"] == "error" and _count(data, day.dt.day() > final_day):
            _fail("INVALID_DATETIME_SHIFT", "The calendar target lacks the original day; declare month_end='clip' deliberately.")
        expected = pl.date(year.cast(pl.Int32), month.cast(pl.Int32), pl.min_horizontal(day.dt.day(), final_day))
    if dtype == pl.Date:
        return expected
    wall = expr.dt.replace_time_zone(None) if dtype.time_zone is not None else expr
    if dtype.time_zone is not None and _count(data, ~wall.dt.date().eq_missing(day) | ~wall.dt.time().eq_missing(expr.dt.time())):
        _fail("DATETIME_OVERFLOW", "The local wall timestamp exceeds the declared temporal representation.")
    shifted_wall = wall.dt.offset_by(str(amount) + unit)
    result = shifted_wall.dt.replace_time_zone(dtype.time_zone, ambiguous="raise", non_existent="raise") if dtype.time_zone is not None else shifted_wall
    try:
        # offset_by can silently wrap ns timestamps. Independent native Date
        # arithmetic verifies the requested wall-calendar date before output.
        failures = _count(data, ~shifted_wall.dt.date().eq_missing(expected))
        if not failures and dtype.time_zone is not None:
            failures = _count(data, ~result.dt.date().eq_missing(expected) | ~result.dt.time().eq_missing(shifted_wall.dt.time()))
    except pl.exceptions.PolarsError as error:
        _fail("INVALID_DATETIME_SHIFT", "Calendar shift reaches an ambiguous or nonexistent local timestamp or unsupported calendar value.", error=str(error))
    if failures:
        _fail("DATETIME_OVERFLOW", "Native timestamp arithmetic cannot represent the requested calendar date.", affected_rows=failures)
    return result


def compile_expression(value, data, compile_child):
    """Compile one extension node; return None when this operator is not defined.

    compile_child is the trusted parent compiler, never a recipe callback.
    Only JSON data select bounded native operators; no expression executes code.
    """
    if not isinstance(value, dict) or len(value) != 1:
        return None
    op, args = next(iter(value.items()))
    if op not in EXPRESSION_DEFINITIONS:
        return None
    try:
        child = lambda item: compile_child(item, data)
        if op == "literal":
            return _literal(data, args)
        if op == "cast":
            _shape(args, ("value", "dtype", "invalid", "precision"), op)
            return _cast(data, child(args["value"]), dtype_spec(args["dtype"]), invalid=args["invalid"], precision=args["precision"])
        if op == "when":
            _shape(args, ("if", "then", "else", "nulls"), op)
            _choice(args["nulls"], ("error", "else", "null"), "when nulls")
            predicate = _predicate(data, child(args["if"]), args["nulls"])
            yes, no = _compatible(data, [child(args["then"]), child(args["else"])])
            result = pl.when(predicate.fill_null(False)).then(yes).otherwise(no)
            return pl.when(predicate.is_null()).then(None).otherwise(result) if args["nulls"] == "null" else result
        if op == "case":
            _shape(args, ("branches", "else", "nulls"), op)
            _choice(args["nulls"], ("error", "skip", "null"), "case nulls")
            if not isinstance(args["branches"], list) or not args["branches"] or len(args["branches"]) > 128:
                _fail("INVALID_EXPRESSION", "case requires 1..128 declared branches.")
            predicates, outcomes = [], []
            for branch in args["branches"]:
                _shape(branch, ("when", "then"), "case branch")
                predicates.append(_predicate(data, child(branch["when"]), args["nulls"]))
                outcomes.append(child(branch["then"]))
            outcomes = _compatible(data, outcomes + [child(args["else"])])
            result = outcomes[-1]
            for predicate, outcome in reversed(list(zip(predicates, outcomes[:-1]))):
                result = pl.when(predicate.fill_null(False)).then(outcome).otherwise(result)
                if args["nulls"] == "null":
                    result = pl.when(predicate.is_null()).then(None).otherwise(result)
            return result
        if op == "coalesce":
            if not isinstance(args, list) or len(args) < 2 or len(args) > 128:
                _fail("INVALID_EXPRESSION", "coalesce requires 2..128 ordered expressions.")
            return pl.coalesce(_compatible(data, [child(item) for item in args]))
        if op == "is_in":
            _shape(args, ("value", "values", "nulls"), op)
            _choice(args["nulls"], ("error", "false", "null", "match"), "membership nulls")
            if not isinstance(args["values"], list) or any(type(item) not in (str, int, float, bool, type(None)) for item in args["values"]):
                _fail("INVALID_EXPRESSION", "Membership values require a list of JSON scalars.")
            expr = child(args["value"])
            dtype = _dtype(data, expr)
            if args["nulls"] == "error":
                _nonnull(data, expr)
            if isinstance(dtype, (pl.Categorical, pl.Enum)):
                expr, dtype = expr.cast(pl.String), pl.String
            values = _series("set", args["values"])
            if dtype != pl.Null and values.dtype != pl.Null:
                if dtype == pl.String and values.dtype != pl.String or dtype.is_numeric() and (not values.dtype.is_numeric()) or dtype == pl.Boolean and values.dtype != pl.Boolean:
                    _fail("DTYPE_MISMATCH", "Membership values must match the tested field type.")
                # Membership literals are separate metadata; validate them once,
                # independently of the number of observation rows.
                metadata = values.to_frame().lazy()
                _cast(metadata, pl.col("set"), dtype)
                values = values.cast(dtype, strict=True)
            result = expr.is_in(pl.lit(values).implode(), nulls_equal=args["nulls"] == "match")
            return result.fill_null(False) if args["nulls"] == "false" else result
        if op in ("trim", "lower", "upper", "length"):
            expr = _string(data, child(args))
            method = {"trim": "strip_chars", "lower": "to_lowercase", "upper": "to_uppercase", "length": "len_chars"}[op]
            return getattr(expr.str, method)()
        if op in ("slice", "contains", "replace", "extract"):
            shapes = {"slice": ("value", "offset", "length"), "contains": ("value", "pattern", "literal"), "replace": ("value", "pattern", "replacement", "literal", "all", "capture_groups"), "extract": ("value", "pattern", "group")}
            _shape(args, shapes[op], op)
            expr = _string(data, child(args["value"]))
            if op == "slice":
                if type(args["offset"]) is not int or type(args["length"]) is not int or args["length"] < 0:
                    _fail("INVALID_EXPRESSION", "slice requires integer offset and nonnegative length.")
                return expr.str.slice(args["offset"], args["length"])
            _pattern(args["pattern"])
            if op == "extract":
                if type(args["group"]) is not int or args["group"] < 0:
                    _fail("INVALID_EXPRESSION", "extract group must be a nonnegative integer.")
                groups = _regex_groups(args["pattern"])
                if str(args["group"]) not in groups:
                    _fail("INVALID_EXPRESSION", "Requested capture does not exist in the declared pattern.")
                return expr.str.extract(args["pattern"], group_index=args["group"])
            if type(args["literal"]) is not bool:
                _fail("INVALID_EXPRESSION", "literal pattern policy must be boolean.")
            groups = _regex_groups(args["pattern"]) if not args["literal"] else set()
            if op == "contains":
                return expr.str.contains(args["pattern"], literal=args["literal"], strict=True)
            if not isinstance(args["replacement"], str) or type(args["all"]) is not bool or type(args["capture_groups"]) is not bool or args["literal"] and args["capture_groups"]:
                _fail("INVALID_EXPRESSION", "Replacement needs explicit literal/group expansion and first/all policies.")
            if args["capture_groups"]:
                _replacement_groups(args["replacement"], groups)
            # A capture wrapper prevents Polars' literal-pattern optimization
            # from changing escaped-dollar semantics for regex replacements.
            pattern = args["pattern"] if args["literal"] or args["capture_groups"] else "(" + args["pattern"] + ")"
            replacement = args["replacement"] if args["literal"] or args["capture_groups"] else args["replacement"].replace("$", "$$")
            method = expr.str.replace_all if args["all"] else expr.str.replace
            return method(pattern, replacement, literal=args["literal"])
        if op == "concat":
            _shape(args, ("values", "separator", "nulls"), op)
            _choice(args["nulls"], ("error", "propagate", "ignore"), "concat nulls")
            if not isinstance(args["values"], list) or len(args["values"]) < 2 or len(args["values"]) > 128 or not isinstance(args["separator"], str):
                _fail("INVALID_EXPRESSION", "concat requires 2..128 strings and a literal separator.")
            exprs = [_string(data, child(item)) for item in args["values"]]
            if args["nulls"] == "error":
                for expr in exprs:
                    _nonnull(data, expr)
            return pl.concat_str(exprs, separator=args["separator"], ignore_nulls=args["nulls"] == "ignore")
        if op == "split":
            _shape(args, ("value", "separator", "max_splits", "nulls"), op)
            _choice(args["nulls"], ("error", "keep"), "split nulls")
            if not isinstance(args["separator"], str) or not 1 <= len(args["separator"]) <= 4096 or type(args["max_splits"]) is not int or not 0 <= args["max_splits"] <= 128:
                _fail("INVALID_EXPRESSION", "split requires a nonempty literal separator and max_splits integer 0..128.")
            expr = _string(data, child(args["value"]))
            if args["nulls"] == "error":
                _nonnull(data, expr)
            parts = expr.str.splitn(args["separator"], n=args["max_splits"] + 1)
            result = pl.concat_list([parts.struct.field("field_" + str(i)) for i in range(args["max_splits"] + 1)]).list.drop_nulls()
            return pl.when(expr.is_null()).then(None).otherwise(result)
        if op == "convert_time_zone":
            _shape(args, ("value", "time_zone"), op)
            expr = child(args["value"])
            dtype = _dtype(data, expr)
            if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None or not isinstance(args["time_zone"], str) or not args["time_zone"]:
                _fail("INVALID_EXPRESSION", "convert_time_zone requires a zoned Datetime and an explicit IANA zone; localize naive timestamps first.")
            pl.select(pl.lit(0).cast(pl.Datetime("us", "UTC")).dt.convert_time_zone(args["time_zone"]))
            return expr.dt.convert_time_zone(args["time_zone"])
        if op == "datetime_shift":
            _shape(args, ("value", "offset", "mode", "precision", "month_end"), op)
            return _datetime_shift(data, child(args["value"]), args)
        if op == "date_part":
            _shape(args, ("value", "part"), op)
            part = args["part"]
            date_parts = ("year", "iso_year", "quarter", "month", "week", "day", "weekday", "ordinal_day")
            clock_parts = ("hour", "minute", "second")
            _choice(part, date_parts + clock_parts, "date part")
            expr = child(args["value"])
            dtype = _dtype(data, expr)
            if part in date_parts and not (dtype == pl.Date or isinstance(dtype, pl.Datetime)) or part in clock_parts and not (dtype == pl.Time or isinstance(dtype, pl.Datetime)):
                _fail("INVALID_EXPRESSION", "Date/clock components require an appropriate native temporal dtype.")
            return getattr(expr.dt, part)()
        if op == "round":
            _shape(args, ("value", "decimals", "mode"), op)
            _choice(args["mode"], ("half_to_even", "half_away_from_zero"), "round mode")
            if type(args["decimals"]) is not int or not 0 <= args["decimals"] <= 38:
                _fail("INVALID_EXPRESSION", "round decimals requires an integer 0..38.")
            expr = child(args["value"])
            if not _dtype(data, expr).is_numeric():
                _fail("INVALID_EXPRESSION", "round requires a numeric expression.")
            return expr.round(args["decimals"], mode=args["mode"])
        if op == "clip":
            _shape(args, ("value", "min", "max", "nulls"), op)
            _choice(args["nulls"], ("error", "keep"), "clip nulls")
            expr = child(args["value"])
            dtype = _dtype(data, expr)
            if not dtype.is_numeric():
                _fail("INVALID_EXPRESSION", "clip requires numeric observations.")
            if args["nulls"] == "error":
                _nonnull(data, expr)
            bounds = []
            for name in ("min", "max"):
                if args[name] is None:
                    bounds.append(None)
                    continue
                bound = child(args[name])
                if not _dtype(data, bound).is_numeric():
                    _fail("INVALID_EXPRESSION", "Clip bounds require numeric expressions or explicit unbounded null.")
                _nonnull(data, bound)
                bounds.append(_cast(data, bound, dtype))
            if all(bound is not None for bound in bounds) and _count(data, bounds[0] > bounds[1]):
                _fail("INVALID_DOMAIN", "Clip lower bounds exceed upper bounds.")
            return expr.clip(*bounds)
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, WrangleError):
            raise
        _fail("INVALID_EXPRESSION", "The native expression cannot satisfy its declared shape or type policies.", operator=op, error=str(error))
