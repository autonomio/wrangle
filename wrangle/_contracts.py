"""Compile declared research checks into native Polars plans and expressions."""
from __future__ import annotations

import math
import polars as pl
from ._core import WrangleError, dtype_spec, missing, require_columns
from ._storage import collect

RULES = {"required", "unique", "ranges", "row_count", "schema", "extra_columns", "allowed", "patterns", "missing", "temporal_ranges", "assertions", "ordering", "foreign_keys", "group_counts", "protocol"}


def _names(value, *, empty=False):
    if not isinstance(value, list) or (not value and not empty) or any(not isinstance(name, str) or not name for name in value) or len(value) != len(set(value)):
        raise WrangleError("INVALID_RECIPE", "Columns must be a list of distinct nonempty names.")


def _mapping(value):
    if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
        raise WrangleError("INVALID_RECIPE", "Field rules must map column names to declarations.")


def _bounds(value, *, counts=False, temporal=False):
    allowed = {"min", "max", "exact"} if counts else {"min", "max"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise WrangleError("INVALID_RECIPE", "Use documented min/max bounds or an exact row count.")
    for bound in value.values():
        valid = type(bound) is int and bound >= 0 if counts else isinstance(bound, str) and bool(bound) if temporal else type(bound) in {int, float} and math.isfinite(bound)
        if not valid:
            raise WrangleError("INVALID_RECIPE", "Bounds require nonnegative integer counts, finite numeric values, or explicit temporal strings.")
    if not temporal and "min" in value and "max" in value and value["min"] > value["max"]:
        raise WrangleError("INVALID_RECIPE", "Range min must not exceed max.")


def validate_rules(rules):
    if not isinstance(rules, dict) or set(rules) - RULES:
        raise WrangleError("INVALID_RECIPE", "Use documented research check names.", {"allowed": sorted(RULES)})
    _names(rules.get("required", []), empty=True)
    unique = rules.get("unique", [])
    if not isinstance(unique, list):
        raise WrangleError("INVALID_RECIPE", "unique must be names or lists of composite keys.")
    if unique and isinstance(unique[0], str):
        _names(unique)
    else:
        for names in unique:
            _names(names)
    for kind in ("ranges", "temporal_ranges"):
        _mapping(rules.get(kind, {}))
        for bounds in rules.get(kind, {}).values():
            if not bounds:
                raise WrangleError("INVALID_RECIPE", "Declare at least one range bound.")
            _bounds(bounds, temporal=kind == "temporal_ranges")
    _bounds(rules.get("row_count", {}), counts=True)
    for kind in ("schema", "allowed", "patterns", "missing"):
        _mapping(rules.get(kind, {}))
    for dtype in rules.get("schema", {}).values():
        dtype_spec(dtype)
    if rules.get("extra_columns", "error") not in ("error", "allow"):
        raise WrangleError("INVALID_RECIPE", "extra_columns must be error or allow.")
    for values in rules.get("allowed", {}).values():
        if not isinstance(values, list) or not values or any(isinstance(value, (dict, list)) for value in values):
            raise WrangleError("INVALID_RECIPE", "Allowed values must be a nonempty list of typed JSON scalars.")
    if any(not isinstance(pattern, str) for pattern in rules.get("patterns", {}).values()):
        raise WrangleError("INVALID_RECIPE", "Patterns must be native regular-expression strings.")
    for bounds in rules.get("missing", {}).values():
        _bounds(bounds)
        if not bounds or any(not 0 <= bound <= 1 for bound in bounds.values()):
            raise WrangleError("INVALID_RECIPE", "Missing fractions must lie between zero and one.")
    for kind in ("assertions", "foreign_keys", "group_counts"):
        if not isinstance(rules.get(kind, []), list):
            raise WrangleError("INVALID_RECIPE", f"{kind} must be a list of declarations.")
    for assertion in rules.get("assertions", []):
        if not isinstance(assertion, dict) or set(assertion) - {"name", "where", "nulls"} or not isinstance(assertion.get("name"), str) or not assertion["name"] or "where" not in assertion or assertion.get("nulls", "error") not in ("error", "pass", "fail"):
            raise WrangleError("INVALID_RECIPE", "Assertions declare name, where expression and nulls=error/pass/fail.")
    for ordering in rules.get("ordering", []) if isinstance(rules.get("ordering", []), list) else [rules["ordering"]]:
        if not isinstance(ordering, dict) or set(ordering) - {"column", "groups", "descending", "ties", "nulls"} or not isinstance(ordering.get("column"), str) or not ordering["column"]:
            raise WrangleError("INVALID_RECIPE", "Ordering declares one column and optional partition groups.")
        _names(ordering.get("groups", []), empty=True)
        if type(ordering.get("descending", False)) is not bool or ordering.get("ties", "allow") not in ("allow", "error") or ordering.get("nulls", "error") not in ("error", "skip"):
            raise WrangleError("INVALID_RECIPE", "Ordering requires Boolean direction and explicit permitted tie/null policies.")
    for foreign in rules.get("foreign_keys", []):
        if not isinstance(foreign, dict) or set(foreign) - {"source", "columns", "reference", "missing"} or not isinstance(foreign.get("source"), str):
            raise WrangleError("INVALID_RECIPE", "Foreign keys declare source, columns, reference and missing policy.")
        _names(foreign.get("columns"))
        _names(foreign.get("reference"))
        if len(foreign["columns"]) != len(foreign["reference"]) or foreign.get("missing", "error") not in ("error", "allow"):
            raise WrangleError("INVALID_RECIPE", "Foreign-key columns must align and missing must be error or allow.")
    for group in rules.get("group_counts", []):
        if not isinstance(group, dict) or set(group) - {"by", "min", "max", "exact", "nulls"}:
            raise WrangleError("INVALID_RECIPE", "Group counts declare by, row-count bounds and nulls=error/group.")
        _names(group.get("by"))
        limits = {name: value for name, value in group.items() if name in {"min", "max", "exact"}}
        if not limits:
            raise WrangleError("INVALID_RECIPE", "Declare at least one group-count bound.")
        _bounds(limits, counts=True)
        if group.get("nulls", "error") not in ("error", "group"):
            raise WrangleError("INVALID_RECIPE", "Group nulls must be error or group.")
    protocol = rules.get("protocol", {})
    if not isinstance(protocol, dict) or set(protocol) - {"key", "units", "descriptions"} or type(protocol.get("key", False)) is not bool:
        raise WrangleError("INVALID_RECIPE", "Protocol checks declare key, units and descriptions requirements.")
    _names(protocol.get("units", []), empty=True)
    _names(protocol.get("descriptions", []), empty=True)


def _count(data, predicate):
    return collect(data.lazy().select(predicate.fill_null(False).sum())).item() or 0


def temporal_literal(value, dtype):
    if dtype == pl.Date:
        return pl.lit(value).str.to_date(format="%Y-%m-%d", strict=True)
    if isinstance(dtype, pl.Datetime):
        return pl.lit(value).str.to_datetime(time_unit=dtype.time_unit, time_zone=dtype.time_zone, strict=True)
    if dtype == pl.Time:
        return pl.lit(value).str.to_time(strict=True)
    raise WrangleError("INVALID_RECIPE", "Temporal bounds require Date, Datetime or Time; declare duration assertions explicitly.")


def _violation(column, limits, temporal=False, dtype=None):
    expression = pl.col(column)
    predicate = pl.lit(False)
    for bound, compare in (("min", lambda value: expression < value), ("max", lambda value: expression > value)):
        if bound not in limits:
            continue
        value = pl.lit(limits[bound])
        if temporal:
            value = temporal_literal(limits[bound], dtype)
        predicate = predicate | compare(value)
    return predicate


def check_data(data, rules, key, *, sources=None, units=None, descriptions=None, expression=None):
    """Execute each declared rule and return a JSON-compatible check receipt."""
    validate_rules(rules)
    passed, schema = [], data.schema
    required = list(dict.fromkeys([*key, *rules.get("required", [])]))
    require_columns(data, required)
    for name in required:
        count = _count(data, missing(pl.col(name), schema[name]))
        if count:
            raise WrangleError("MISSING_REQUIRED", "Required values are missing.", {"column": name, "affected_rows": count})
        passed.append({"check": "required", "column": name, "passed": True})
    unique = rules.get("unique", [])
    sets = ([unique] if unique and isinstance(unique[0], str) else unique) + ([key] if key else [])
    for names in sets:
        require_columns(data, names)
        duplicates = collect(data.lazy().group_by(names).len().filter(pl.col("len") > 1).select(pl.col("len").sum())).item() or 0
        if duplicates:
            raise WrangleError("DUPLICATE_KEY", "Declared observation keys are not unique.", {"columns": names, "affected_rows": duplicates})
        passed.append({"check": "unique", "columns": names, "passed": True})
    declared_schema = rules.get("schema", {})
    if declared_schema:
        require_columns(data, list(declared_schema))
        mismatch = {name: {"actual": str(schema[name]), "expected": str(dtype_spec(dtype))} for name, dtype in declared_schema.items() if schema[name] != dtype_spec(dtype)}
        extra = [name for name in schema if name not in declared_schema]
        if mismatch or extra and rules.get("extra_columns", "error") == "error":
            raise WrangleError("SCHEMA_MISMATCH", "The table does not satisfy the declared schema.", {"columns": mismatch, "extra_columns": extra})
        passed.append({"check": "schema", "passed": True})
    for kind in ("ranges", "temporal_ranges"):
        for name, limits in rules.get(kind, {}).items():
            require_columns(data, name)
            temporal = kind == "temporal_ranges"
            if temporal and not schema[name].is_temporal() or not temporal and not schema[name].is_numeric():
                raise WrangleError("INVALID_RECIPE", "Ranges require the declared numeric or temporal column type.", {"column": name})
            if temporal and "min" in limits and "max" in limits:
                if pl.select(temporal_literal(limits["min"], schema[name]) > temporal_literal(limits["max"], schema[name])).item():
                    raise WrangleError("INVALID_RECIPE", "Temporal min must not exceed max.", {"column": name})
            count = _count(data, _violation(name, limits, temporal, schema[name]))
            if count:
                raise WrangleError("RANGE_VIOLATION", "Measurements violate the declared range.", {"column": name, "bounds": limits, "affected_rows": count})
            passed.append({"check": "range", "column": name, "bounds": limits, "passed": True})
    for name, values in rules.get("allowed", {}).items():
        require_columns(data, name)
        nonnull = [value for value in values if value is not None]
        try:
            typed = pl.Series("allowed", nonnull, dtype=schema[name], strict=True)
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise WrangleError("INVALID_RECIPE", "Allowed values must have the field's exact declared type.", {"column": name}) from error
        predicate = pl.col(name).is_in(typed.implode()).fill_null(None in values) if nonnull else pl.col(name).is_null()
        count = _count(data, ~predicate)
        if count:
            raise WrangleError("DOMAIN_VIOLATION", "Values lie outside the declared domain.", {"column": name, "affected_rows": count})
        passed.append({"check": "allowed", "column": name, "passed": True})
    for name, pattern in rules.get("patterns", {}).items():
        require_columns(data, name)
        if schema[name] != pl.String:
            raise WrangleError("INVALID_RECIPE", "Pattern checks require String columns.", {"column": name})
        count = _count(data, pl.col(name).is_not_null() & ~pl.col(name).str.contains(pattern))
        if count:
            raise WrangleError("PATTERN_VIOLATION", "Values do not match the declared pattern.", {"column": name, "affected_rows": count})
        passed.append({"check": "pattern", "column": name, "passed": True})
    for name, limits in rules.get("missing", {}).items():
        require_columns(data, name)
        count = _count(data, missing(pl.col(name), schema[name]))
        fraction = collect(data.lazy().select(missing(pl.col(name), schema[name]).mean())).item()
        if fraction is None:
            raise WrangleError("UNDEFINED_MISSINGNESS", "Missing fractions need at least one observation.", {"column": name})
        if "min" in limits and fraction < limits["min"] or "max" in limits and fraction > limits["max"]:
            raise WrangleError("MISSINGNESS_VIOLATION", "Missingness violates the declared fraction.", {"column": name, "missing": count, "rows": data.height, "fraction": fraction})
        passed.append({"check": "missing", "column": name, "fraction": fraction, "passed": True})
    for assertion in rules.get("assertions", []):
        predicate = expression(assertion["where"], data.lazy())
        if data.lazy().select(predicate).collect_schema().dtypes() != [pl.Boolean]:
            raise WrangleError("INVALID_EXPRESSION", "A research assertion must evaluate to Boolean values.")
        nulls = _count(data, predicate.is_null())
        policy = assertion.get("nulls", "error")
        if nulls and policy == "error":
            raise WrangleError("UNRESOLVED_ASSERTION", "The assertion is unknown for missing observations.", {"name": assertion["name"], "affected_rows": nulls})
        count = _count(data, ~predicate.fill_null(policy == "pass"))
        if count:
            raise WrangleError("ASSERTION_FAILED", "The declared research assertion failed.", {"name": assertion["name"], "affected_rows": count})
        passed.append({"check": "assertion", "name": assertion["name"], "passed": True})
    orderings = rules.get("ordering", [])
    orderings = orderings if isinstance(orderings, list) else [orderings]
    for ordering in orderings:
        name, groups = ordering["column"], ordering.get("groups", [])
        require_columns(data, [name, *groups])
        if groups and _count(data, pl.any_horizontal([missing(pl.col(group), schema[group]) for group in groups])):
            raise WrangleError("NULL_KEY", "Ordering partitions require observed group keys.")
        if ordering.get("nulls", "error") == "error" and _count(data, missing(pl.col(name), schema[name])):
            raise WrangleError("MISSING_REQUIRED", "Ordering requires observed values.", {"column": name})
        table = data.lazy().filter(~missing(pl.col(name), schema[name]))
        previous = pl.col(name).shift(1)
        if groups:
            previous = previous.over(groups)
        invalid = pl.col(name) > previous if ordering.get("descending", False) else pl.col(name) < previous
        if ordering.get("ties", "allow") == "error":
            invalid = invalid | (pl.col(name) == previous)
        count = collect(table.select(invalid.fill_null(False).sum())).item()
        if count:
            raise WrangleError("ORDER_VIOLATION", "Observations violate the declared partition ordering.", {"column": name, "groups": groups, "affected_rows": count})
        passed.append({"check": "ordering", "column": name, "groups": groups, "passed": True})
    for foreign in rules.get("foreign_keys", []):
        name = foreign["source"]
        if sources is None or name not in sources:
            raise WrangleError("UNKNOWN_SOURCE", "The foreign-key reference source is absent.", {"source": name})
        parent = sources[name]
        left, right = foreign["columns"], foreign["reference"]
        require_columns(data, left)
        require_columns(parent, right)
        if any(data.schema[a] != parent.schema[b] for a, b in zip(left, right)):
            raise WrangleError("DTYPE_MISMATCH", "Foreign-key fields must have identical declared dtypes.")
        parent_missing = pl.any_horizontal([missing(pl.col(column), parent.schema[column]) for column in right])
        if _count(parent, parent_missing) or collect(parent.lazy().group_by(right).len().filter(pl.col("len") > 1).select(pl.len())).item():
            raise WrangleError("FOREIGN_KEY_CARDINALITY", "Reference keys must be observed and unique.", {"source": name})
        absent = pl.any_horizontal([missing(pl.col(column), schema[column]) for column in left])
        if foreign.get("missing", "error") == "error" and _count(data, absent):
            raise WrangleError("NULL_KEY", "Foreign keys require observed values.", {"columns": left})
        count = collect(data.lazy().filter(~absent).join(parent.lazy().select(right), left_on=left, right_on=right, how="anti", validate="m:m", maintain_order="left").select(pl.len())).item()
        if count:
            raise WrangleError("FOREIGN_KEY_VIOLATION", "Observations reference absent parent records.", {"source": name, "affected_rows": count})
        passed.append({"check": "foreign_key", "source": name, "passed": True})
    for group in rules.get("group_counts", []):
        names = group["by"]
        require_columns(data, names)
        if group.get("nulls", "error") == "error" and _count(data, pl.any_horizontal([missing(pl.col(name), schema[name]) for name in names])):
            raise WrangleError("NULL_KEY", "Group counts require observed group keys.")
        table = data.lazy().group_by(names).len()
        invalid = pl.lit(False)
        for bound, condition in (("min", lambda n: pl.col("len") < n), ("max", lambda n: pl.col("len") > n), ("exact", lambda n: pl.col("len") != n)):
            if bound in group:
                invalid = invalid | condition(group[bound])
        count = collect(table.filter(invalid).select(pl.len())).item()
        if count:
            raise WrangleError("GROUP_COUNT_VIOLATION", "Group observation counts violate the declared bounds.", {"by": names, "affected_groups": count})
        passed.append({"check": "group_count", "by": names, "passed": True})
    counts = rules.get("row_count", {})
    if ("min" in counts and data.height < counts["min"]) or ("max" in counts and data.height > counts["max"]) or ("exact" in counts and data.height != counts["exact"]):
        raise WrangleError("ROW_COUNT_VIOLATION", "The output observation count violates the recipe.", {"rows": data.height, "expected": counts})
    if counts:
        passed.append({"check": "row_count", "rows": data.height, "expected": counts, "passed": True})
    protocol = rules.get("protocol", {})
    if protocol:
        unresolved = {"key": bool(protocol.get("key") and not key), "units": [name for name in protocol.get("units", []) if not (units or {}).get(name)], "descriptions": [name for name in protocol.get("descriptions", []) if not (descriptions or {}).get(name)]}
        require_columns(data, [*protocol.get("units", []), *protocol.get("descriptions", [])])
        if any(unresolved.values()):
            raise WrangleError("UNRESOLVED_PROTOCOL", "Resolve the required scientific declarations before releasing this table.", unresolved)
        passed.append({"check": "protocol", "passed": True})
    return passed
