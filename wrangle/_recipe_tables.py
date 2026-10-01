"""Scientific table recipes: native lazy plans, explicit grain and ordering choices."""
from __future__ import annotations

import math
import inspect
import re
from fractions import Fraction
import polars as pl

from ._storage import collect
from ._core import WrangleError, operation, require_columns
from .df._expressions import clean, missing, require_float_exactness, seed_value, temp_name


def _fail(code, message, **details):
    raise WrangleError(code, message, details)


def _names(value, name, *, empty=False):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail("INVALID_OPTION", f"{name} must be a list of column names.")
    if (not value and not empty) or len(value) != len(set(value)):
        _fail("INVALID_OPTION", f"{name} must contain distinct column names.")
    return value


def _choice(value, choices, name):
    if value not in choices:
        _fail("INVALID_OPTION", f"{name} must be one of {', '.join(choices)}.")


def _bool(value, name):
    if not isinstance(value, bool):
        _fail("INVALID_OPTION", f"{name} must be boolean.")


def _integer(value, name, *, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        _fail("INVALID_OPTION", f"{name} must be an integer >= {minimum}.")


def _output(data, name):
    if not isinstance(name, str) or not name or name in data.collect_schema():
        _fail("OUTPUT_COLLISION", "Output names must be nonempty and absent from the input.", column=name)


def _nonmissing(data, names):
    if not names:
        return
    schema = data.collect_schema()
    require_columns(data, names)
    counts = collect(data.select([missing(name, schema[name]).sum().alias(name) for name in names])).row(0, named=True)
    bad = {name: n for name, n in counts.items() if n}
    if bad:
        _fail("MISSING_KEY", "Ordering, grouping, and identity columns require observed values.", columns=bad)


def _duplicates(data, names):
    return collect(data.group_by(names).len().filter(pl.col("len") > 1).select(pl.len())).item() if names else 0


def _tie_check(data, by, order, ties):
    _choice(ties, ("error", "source"), "ties")
    _nonmissing(data, by + order)
    if ties == "error" and _duplicates(data, list(dict.fromkeys(by + order))):
        _fail("AMBIGUOUS_ORDER", "Ordering keys contain ties; declare tie breakers or ties='source'.", columns=by + order)


def _sum_safe(data, name, dtype):
    # Division bounds the possible absolute sum; no guard arithmetic can itself
    # overflow and no observation-derived arithmetic runs in Python.
    if dtype == pl.Int128:
        value = pl.col(name)
        if collect(data.select((value == pl.lit(-(2**127), dtype=pl.Int128)).any())).item():
            _fail("ARITHMETIC_OVERFLOW", "Int128 minimum has no representable absolute value; rescale explicitly.", column=name)
        safe = value.abs().max() <= pl.lit(2**127 - 1, dtype=pl.Int128) // value.count().clip(lower_bound=1).cast(pl.Int128)
        if not collect(data.select(safe.fill_null(True))).item():
            _fail("ARITHMETIC_OVERFLOW", "An exact Int128 sum cannot be guaranteed; rescale explicitly.", column=name)


def _uniform(expr, seed, *, positive=False):
    # Use exactly representable mantissas. The log stream uses 52 bits so half
    # a unit remains representable at both ends; ordinary draws use 53 bits.
    bits = 52 if positive else 53
    return (expr.hash(seed=seed) // (2 ** (64 - bits))).cast(pl.Float64).add(0.5 if positive else 0.0) / float(2**bits)


def _explode(data, fields):
    # Polars 1.34 predates empty_as_null. Inputs already contain explicit null
    # placeholders, so both supported versions execute the same native plan.
    kwargs = {"empty_as_null": True} if "empty_as_null" in inspect.signature(pl.LazyFrame.explode).parameters else {}
    return data.explode(fields, **kwargs)


def _floor_share(size, share):
    ratio = share if isinstance(share, Fraction) else Fraction(str(share))
    if ratio.numerator > 2**95 - 1 or ratio.denominator > 2**127 - 1:
        _fail("INVALID_OPTION", "Allocation shares need a representable exact decimal ratio.")
    return (size.cast(pl.Int128) * pl.lit(ratio.numerator, dtype=pl.Int128) // pl.lit(ratio.denominator, dtype=pl.Int128)).cast(pl.UInt64)


def _metrics(data, metrics):
    if not isinstance(metrics, dict) or not metrics:
        _fail("INVALID_OPTION", "metrics must map output names to named reduction specifications.")
    schema = data.collect_schema()
    result = []
    for output, spec in metrics.items():
        if not isinstance(output, str) or not output or not isinstance(spec, dict):
            _fail("INVALID_OPTION", "Each metric needs a nonempty output name and an object specification.")
        unknown = set(spec) - {"column", "method", "nulls", "min_count", "ddof", "q", "interpolation"}
        if unknown:
            _fail("INVALID_OPTION", "Unknown metric options.", options=sorted(unknown))
        method = spec.get("method")
        _choice(method, ("len", "count", "n_unique", "sum", "mean", "median", "std", "var", "min", "max", "quantile"), "method")
        if method == "len":
            if set(spec) != {"method"}:
                _fail("INVALID_OPTION", "len counts rows and takes no column or missingness options.")
            result.append(pl.len().alias(output))
            continue
        allowed = {"column", "method", "nulls"}
        if method not in {"count", "n_unique"}:
            allowed.add("min_count")
        if method in {"std", "var"}:
            allowed.add("ddof")
        if method == "quantile":
            allowed.update({"q", "interpolation"})
        if set(spec) - allowed:
            _fail("INVALID_OPTION", "Reduction options do not apply to the declared method.", options=sorted(set(spec) - allowed))
        name = spec.get("column")
        require_columns(data, [name] if isinstance(name, str) else [])
        if not isinstance(name, str) or name not in schema:
            _fail("UNKNOWN_COLUMN", "Metrics require an observed column.", column=name)
        nulls = spec.get("nulls")
        _choice(nulls, ("ignore", "error"), "metric nulls")
        if nulls == "error":
            _nonmissing(data, [name])
        value = clean(name, schema[name])
        if method == "count":
            expr = value.count()
        elif method == "n_unique":
            expr = value.drop_nulls().n_unique()
        else:
            minimum = spec.get("min_count")
            _integer(minimum, "min_count", minimum=1)
            if method in {"sum", "mean", "median", "std", "var", "quantile"}:
                if not schema[name].is_numeric():
                    _fail("NON_NUMERIC_COLUMN", "This reduction requires numeric observations.", column=name)
                if method == "sum" and schema[name].is_integer():
                    _sum_safe(data, name, schema[name])
                    value = value.cast(pl.Int128)
                elif method != "sum":
                    require_float_exactness(data, [name])
                    value = value.cast(pl.Float64)
            if method in {"std", "var"}:
                ddof = spec.get("ddof")
                _integer(ddof, "ddof")
                expr = getattr(value, method)(ddof=ddof)
            elif method == "quantile":
                q = spec.get("q")
                if isinstance(q, bool) or not isinstance(q, (float, int)) or not 0 <= q <= 1:
                    _fail("INVALID_OPTION", "q must be between 0 and 1.")
                interpolation = spec.get("interpolation")
                _choice(interpolation, ("nearest", "higher", "lower", "midpoint", "linear", "equiprobable"), "interpolation")
                expr = value.quantile(q, interpolation=interpolation)
            else:
                expr = getattr(value, method)()
            expr = pl.when(value.count() >= minimum).then(expr).otherwise(None)
        result.append(expr.alias(output))
    return result


@operation(returns=("table",), recipe="yes")
def sort(data, by, *, descending, nulls, ties):
    """Sort declared columns; ties='source' preserves their prior order, 'error' rejects ties."""
    by = _names(by, "by")
    require_columns(data, by)
    _choice(nulls, ("first", "last", "error"), "nulls")
    _choice(ties, ("error", "source"), "ties")
    if isinstance(descending, list):
        if len(descending) != len(by) or any(not isinstance(item, bool) for item in descending):
            _fail("INVALID_OPTION", "descending must have one boolean per sort column.")
    else:
        _bool(descending, "descending")
    if nulls == "error":
        _nonmissing(data, by)
    sort_keys = [clean(name, data.collect_schema()[name]) for name in by]
    if ties == "error" and _duplicates(data.select([expr.alias(name) for expr, name in zip(sort_keys, by)]), by):
        _fail("AMBIGUOUS_ORDER", "Sort keys contain ties; add a tie breaker or declare source ordering.")
    return data.sort(sort_keys, descending=descending, nulls_last=nulls == "last", maintain_order=True)


@operation(returns=("table",), recipe="yes")
def deduplicate(data, by, *, keep, conflicts, order_by=None, descending=False, ties="error"):
    """Remove duplicate identities only under declared survivor and conflicting-value policies."""
    by = _names(by, "by")
    require_columns(data, by)
    _nonmissing(data, by)
    _choice(keep, ("error", "first", "last", "none"), "keep")
    _choice(conflicts, ("error", "allow"), "conflicts")
    _bool(descending, "descending")
    order = _names(order_by, "order_by", empty=True) if order_by is not None else []
    require_columns(data, order)
    duplicate_count = _duplicates(data, by)
    if keep == "error":
        if duplicate_count:
            _fail("DUPLICATE_KEY", "Repeated identities require a declared survivor policy.", duplicate_groups=duplicate_count)
        return data
    if conflicts == "error":
        other = [name for name in data.collect_schema().names() if name not in by]
        if other:
            conflict = collect(data.group_by(by).agg(pl.struct(other).n_unique().alias("__conflicts")).filter(pl.col("__conflicts") > 1).select(pl.len())).item()
            if conflict:
                _fail("DUPLICATE_CONFLICT", "Repeated identities contain differing observations.", groups=conflict)
    _choice(ties, ("error", "source"), "ties")
    if keep in {"first", "last"} and duplicate_count and not order and ties != "source":
        _fail("AMBIGUOUS_ORDER", "Selecting a duplicate survivor requires order_by or ties='source'.")
    if order:
        _tie_check(data, by, order, ties)
    row = temp_name(data)
    plan = data.with_row_index(row)
    if order:
        plan = plan.sort(order, descending=descending, maintain_order=True)
    return plan.unique(subset=by, keep=keep, maintain_order=True).sort(row).drop(row)


@operation(returns=("table",), recipe="yes")
def concat(data, sources, *, schema, provenance, labels):
    """Append sources in declared order; strict/union never coerce shared field types."""
    if not isinstance(sources, list) or any(not isinstance(source, pl.LazyFrame) for source in sources):
        _fail("INVALID_INPUT", "sources must be bound native lazy tables.")
    _choice(schema, ("strict", "union"), "schema")
    frames = [data, *sources]
    if not isinstance(labels, list) or len(labels) != len(frames) or any(not isinstance(label, str) or not label for label in labels) or len(labels) != len(set(labels)):
        _fail("INVALID_OPTION", "labels need one distinct nonempty string per source, including the input.")
    schemas = [source.collect_schema() for source in frames]
    fields = {}
    for source_schema in schemas:
        for name, dtype in source_schema.items():
            if name in fields and fields[name] != dtype:
                _fail("SCHEMA_MISMATCH", "Shared concat columns need identical dtypes; declare casts before concatenating.", column=name)
            fields[name] = dtype
    if schema == "strict" and any(list(source_schema.items()) != list(schemas[0].items()) for source_schema in schemas[1:]):
        _fail("SCHEMA_MISMATCH", "Strict concat requires identical field names, order, and dtypes.")
    if not isinstance(provenance, str) or not provenance or provenance in fields:
        _fail("OUTPUT_COLLISION", "provenance must name an absent output column.")
    plans = [source.select([pl.col(name) if name in source_schema else pl.lit(None, dtype=dtype).alias(name) for name, dtype in fields.items()]).with_columns(pl.lit(label).alias(provenance)) for source, source_schema, label in zip(frames, schemas, labels)]
    return pl.concat(plans, how="vertical")


@operation(returns=("table",), recipe="yes", aggregates=True)
def pivot(data, index, on, values, domain, *, duplicate_cells, missing_cells, separator="__", nulls="ignore", min_count=1, ddof=1):
    """Pivot a declared category domain; absent cells differ from observed missing measurements."""
    index = _names(index, "index")
    values = _names(values, "values")
    if not isinstance(on, str):
        _fail("INVALID_OPTION", "on must be one category column.")
    if len(set(index + values + [on])) != len(index + values + [on]):
        _fail("INVALID_OPTION", "index, on, and values must be disjoint.")
    require_columns(data, index + values + [on])
    _nonmissing(data, index + [on])
    _choice(duplicate_cells, ("error", "sum", "mean", "median", "min", "max", "std"), "duplicate_cells")
    _choice(missing_cells, ("null", "zero", "error"), "missing_cells")
    if not isinstance(domain, list) or not domain or any(isinstance(item, (list, dict)) or item is None for item in domain):
        _fail("INVALID_OPTION", "domain must be a nonempty list of observed scalar categories.")
    if len(set(domain)) != len(domain) or len({str(item) for item in domain}) != len(domain):
        _fail("OUTPUT_COLLISION", "Pivot categories must be unique with distinct output names.")
    if not isinstance(separator, str) or not separator:
        _fail("INVALID_OPTION", "separator must be nonempty.")
    dtype = data.collect_schema()[on]
    if not (dtype.is_integer() or dtype in (pl.String, pl.Boolean) or isinstance(dtype, (pl.Categorical, pl.Enum))):
        _fail("INVALID_OPTION", "Pivot categories require string, categorical, boolean, or integer identity.")
    try:
        domain_series = pl.Series("domain", domain, dtype=dtype, strict=True)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
        _fail("INVALID_OPTION", "domain must have exactly the category column dtype.", error=str(error))
    if collect(data.filter(~pl.col(on).is_in(domain_series.implode())).select(pl.len())).item():
        _fail("PIVOT_DOMAIN", "Observed categories fall outside the declared domain.")
    if duplicate_cells == "error" and _duplicates(data, index + [on]):
        _fail("DUPLICATE_CELL", "Pivot cells contain multiple observations; declare a reduction.")
    output_names = [f"{name}{separator}{category}" for name in values for category in domain]
    if len(output_names) != len(set(output_names)) or set(output_names) & set(index):
        _fail("OUTPUT_COLLISION", "Generated pivot columns collide; change separator or category labels.")
    exprs = []
    for name in values:
        for category in domain:
            selected = pl.col(on) == pl.lit(category, dtype=dtype)
            metric = {"column": name, "method": duplicate_cells if duplicate_cells != "error" else "min", "nulls": nulls, "min_count": min_count}
            if metric["method"] == "std":
                metric["ddof"] = ddof
            # Build against the filtered domain: scalar safety checks remain native.
            _metrics(data.filter(selected), {"value": metric})
            # A filtered expression is essential: unrelated categories cannot
            # contribute to either numerator or observed-count denominator.
            value = clean(name, data.collect_schema()[name]).filter(selected)
            if duplicate_cells == "error":
                cell = pl.when(value.count() >= min_count).then(value.first()).otherwise(None)
            else:
                schema_name = data.collect_schema()[name]
                if duplicate_cells == "sum" and schema_name.is_integer():
                    value = value.cast(pl.Int128)
                elif duplicate_cells in {"mean", "median", "std"}:
                    value = value.cast(pl.Float64)
                reduction = value.std(ddof=ddof) if duplicate_cells == "std" else getattr(value, duplicate_cells)()
                cell = pl.when(value.count() >= min_count).then(reduction).otherwise(None)
            if missing_cells == "zero":
                if not data.collect_schema()[name].is_numeric():
                    _fail("NON_NUMERIC_COLUMN", "Zero-filled pivot cells require numeric measurements.", column=name)
                cell = pl.when(selected.sum() == 0).then(0).otherwise(cell)
            exprs.append(cell.alias(f"{name}{separator}{category}"))
    if missing_cells == "error":
        incomplete = collect(data.group_by(index).agg(pl.col(on).n_unique().alias("__domain_count")).filter(pl.col("__domain_count") != len(domain)).select(pl.len())).item()
        if incomplete:
            _fail("MISSING_CELL", "The declared domain is absent for some pivot identities.", groups=incomplete)
    return data.group_by(index, maintain_order=True).agg(exprs)


@operation(returns=("table",), recipe="yes")
def unpivot(data, index, on, *, variable, value, nulls):
    """Reshape equal-dtype measurements in source-row then declared-variable order."""
    index = _names(index, "index", empty=True)
    on = _names(on, "on")
    if set(index) & set(on):
        _fail("INVALID_OPTION", "index and on must be disjoint.")
    require_columns(data, index + on)
    _output(data, variable)
    _output(data, value)
    if variable == value:
        _fail("OUTPUT_COLLISION", "variable and value need distinct names.")
    _choice(nulls, ("keep", "drop", "error"), "nulls")
    schema = data.collect_schema()
    if len({schema[name] for name in on}) != 1:
        _fail("SCHEMA_MISMATCH", "Unpivot measurements require identical dtypes; declare exact casts before reshaping.")
    if nulls == "error":
        _nonmissing(data, on)
    row = temp_name(data)
    plan = data.with_row_index(row).unpivot(on=on, index=index + [row], variable_name=variable, value_name=value)
    if nulls == "drop":
        plan = plan.filter(~missing(value, schema[on[0]]))
    return plan.sort([row, pl.col(variable).replace_strict({name: i for i, name in enumerate(on)})]).drop(row)


@operation(returns=("table",), recipe="yes")
def explode(data, columns, *, index, empty, nulls):
    """Explode aligned lists with zero-based element identity; retained absent-list placeholders use index=-1."""
    columns = _names(columns, "columns")
    require_columns(data, columns)
    _output(data, index)
    _choice(empty, ("error", "drop", "null"), "empty")
    _choice(nulls, ("error", "drop", "null"), "nulls")
    schema = data.collect_schema()
    if any(not isinstance(schema[name], (pl.List, pl.Array)) for name in columns):
        _fail("INVALID_DTYPE", "explode requires List or Array columns.")
    plan = data.with_columns([pl.col(name).arr.to_list().alias(name) for name in columns if isinstance(schema[name], pl.Array)])
    lens = [pl.col(name).list.len() for name in columns]
    if len(columns) > 1:
        incompatible = pl.any_horizontal([~lens[0].eq_missing(length) for length in lens[1:]])
        if collect(plan.filter(incompatible).select(pl.len())).item():
            _fail("LIST_LENGTH_MISMATCH", "Columns exploded together must have equal lengths and matching null-list status.")
    length = lens[0]
    for policy, condition, kind in [(empty, length == 0, "empty"), (nulls, length.is_null(), "null")]:
        if policy == "error" and collect(plan.filter(condition).select(pl.len())).item():
            _fail("MISSING_LIST", f"{kind} lists require an explicit retention or exclusion policy.")
        if policy == "drop":
            plan = plan.filter(~condition.fill_null(False))
    # Normalize placeholders before explode so the contract does not depend on
    # changing Polars empty-list defaults (and works on minimum Polars 1.34).
    plan = plan.with_columns([
        pl.when((pl.col(name).list.len() == 0) | pl.col(name).is_null()).then(pl.lit([None], dtype=plan.collect_schema()[name])).otherwise(pl.col(name)).alias(name)
        for name in columns
    ] + [pl.when(length.fill_null(0) == 0).then(pl.lit([-1], dtype=pl.List(pl.Int64))).otherwise(pl.int_ranges(0, length, dtype=pl.Int64)).alias(index)])
    return _explode(plan, columns + [index])


@operation(returns=("table",), recipe="yes")
def unnest(data, columns, *, separator):
    """Expand struct fields with parent-name prefixes; reject every output collision."""
    columns = _names(columns, "columns")
    require_columns(data, columns)
    if not isinstance(separator, str) or not separator:
        _fail("INVALID_OPTION", "separator must be nonempty.")
    schema = data.collect_schema()
    outputs = []
    names = [name for name in schema.names() if name not in columns]
    for name in columns:
        if not isinstance(schema[name], pl.Struct):
            _fail("INVALID_DTYPE", "unnest requires Struct columns.", column=name)
        for field in schema[name].fields:
            output = name + separator + field.name
            if output in names:
                _fail("OUTPUT_COLLISION", "Generated struct fields collide with existing columns.", column=output)
            names.append(output)
            outputs.append(pl.col(name).struct.field(field.name).alias(output))
    return data.select([pl.col(name) for name in schema.names() if name not in columns] + outputs)


@operation(returns=("table",), recipe="yes", aggregates=True)
def aggregate(data, by, metrics, *, null_keys, time=None, every=None, period=None, closed=None, label=None):
    """Reduce declared groups, optionally fixed/calendar time windows; missingness, minimum counts and ddof are explicit per metric."""
    by = _names(by, "by", empty=True)
    require_columns(data, by)
    _choice(null_keys, ("error", "group"), "null_keys")
    if null_keys == "error":
        _nonmissing(data, by)
    exprs = _metrics(data, metrics)
    if set(metrics) & set(by + ([time] if time else [])):
        _fail("OUTPUT_COLLISION", "Metric names cannot replace grouping keys.")
    if time is None:
        if any(option is not None for option in (every, period, closed, label)):
            _fail("INVALID_OPTION", "Window options require a time column.")
        return data.group_by(by, maintain_order=True).agg(exprs) if by else data.select(exprs)
    require_columns(data, [time])
    if time in by or not isinstance(data.collect_schema()[time], (pl.Datetime, pl.Date)):
        _fail("INVALID_DTYPE", "time must be a Date/Datetime column outside by.")
    _nonmissing(data, [time])
    _duration(every, "every")
    _duration(period, "period")
    _choice(closed, ("left", "right", "both", "none"), "closed")
    _choice(label, ("left", "right", "datapoint"), "label")
    return data.sort(by + [time], maintain_order=True).group_by_dynamic(time, every=every, period=period, closed=closed, label=label, group_by=by or None).agg(exprs)


def _duration(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"(?:[1-9][0-9]*(?:ns|us|ms|s|m|h|d|w|mo|q|y))+", value):
        _fail("INVALID_OPTION", f"{name} requires a positive Polars duration, e.g. '7d' or '1mo'.")


@operation(returns=("table",), recipe="yes")
def join_asof(data, source, on, by, *, strategy, tolerance, exact, ties, unmatched, suffix="__right", nearest_ties=None):
    """Match within declared subjects and tolerance; restore left order and reject right-time ambiguity."""
    by = _names(by, "by", empty=True)
    if not isinstance(on, str) or on in by:
        _fail("INVALID_OPTION", "on must name an ordered column outside by.")
    require_columns(data, by + [on])
    require_columns(source, by + [on])
    _nonmissing(data, by + [on])
    _nonmissing(source, by + [on])
    _choice(strategy, ("backward", "forward", "nearest"), "strategy")
    _choice(ties, ("error", "first", "last"), "ties")
    _choice(unmatched, ("error", "keep", "drop"), "unmatched")
    _bool(exact, "exact")
    if strategy == "nearest":
        _choice(nearest_ties, ("error", "backward", "forward"), "nearest_ties")
    elif nearest_ties is not None:
        _fail("INVALID_OPTION", "nearest_ties applies only to strategy='nearest'.")
    left_schema, right_schema = data.collect_schema(), source.collect_schema()
    if any(left_schema[name] != right_schema[name] for name in by + [on]):
        _fail("SCHEMA_MISMATCH", "As-of subject and ordered keys require identical dtypes.")
    if not (left_schema[on].is_numeric() or isinstance(left_schema[on], (pl.Date, pl.Datetime, pl.Duration))):
        _fail("INVALID_DTYPE", "As-of ordered keys must be numeric or temporal.")
    if isinstance(left_schema[on], (pl.Date, pl.Datetime, pl.Duration)):
        if not isinstance(tolerance, str) or not re.fullmatch(r"(?:[0-9]+(?:ns|us|ms|s|m|h|d|w))+", tolerance):
            _fail("INVALID_OPTION", "Temporal tolerance requires a nonnegative fixed duration, e.g. '0d' or '2h'.")
    elif isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
        _fail("INVALID_OPTION", "Numeric tolerance must be finite and nonnegative.")
    if not isinstance(suffix, str) or not suffix:
        _fail("INVALID_OPTION", "suffix must be nonempty.")
    payload = [name for name in right_schema.names() if name not in by + [on]]
    destinations = [name + suffix if name in left_schema else name for name in payload]
    if len(destinations) != len(set(destinations)) or set(destinations) & set(left_schema.names()):
        _fail("OUTPUT_COLLISION", "As-of output columns collide; rename source fields or change suffix.")
    duplicates = _duplicates(source, by + [on])
    if duplicates and ties == "error":
        _fail("AMBIGUOUS_MATCH", "Right subject/time keys contain ties; declare first/last source-row selection.")
    if duplicates:
        source = source.unique(subset=by + [on], keep=ties, maintain_order=True)
    row = temp_name(data)
    all_names = set(left_schema.names()) | set(right_schema.names()) | set(destinations) | {row}
    marker = "__wrangle_match"
    while marker in all_names:
        marker += "_"
    left = data.with_row_index(row).sort(by + [on], maintain_order=True)
    if strategy == "nearest":
        # Build both candidates. Equidistant selection is a scientific decision,
        # independent of Polars' version-dependent nearest tie convention.
        candidates = {}
        plan = left
        for direction in ("backward", "forward"):
            aliases = []
            for i, name in enumerate([on] + payload):
                alias = f"__wrangle_{direction}_{i}"
                while alias in all_names:
                    alias += "_"
                all_names.add(alias)
                aliases.append(alias)
            candidates[direction] = aliases
            right = source.select([pl.col(name) for name in by + [on]] + [pl.col(name).alias(alias) for name, alias in zip([on] + payload, aliases)]).sort(by + [on], maintain_order=True)
            plan = plan.join_asof(right, on=on, by=by or None, strategy=direction, tolerance=tolerance, allow_exact_matches=exact, check_sortedness=False)
        backward, forward = candidates["backward"], candidates["forward"]
        dtype = left_schema[on]
        def distance(candidate):
            left_value, right_value = pl.col(on), pl.col(candidate)
            if dtype.is_integer() or isinstance(dtype, (pl.Date, pl.Datetime, pl.Duration)):
                left_value, right_value = left_value.cast(pl.Int128), right_value.cast(pl.Int128)
            return (left_value - right_value).abs()
        bd, fd = distance(backward[0]), distance(forward[0])
        if dtype.is_float() and collect(plan.filter((~bd.is_finite() & bd.is_not_null()) | (~fd.is_finite() & fd.is_not_null())).select(pl.len())).item():
            _fail("ARITHMETIC_OVERFLOW", "As-of distances overflow floating arithmetic; rescale time explicitly.")
        equidistant = (bd == fd) & (pl.col(backward[0]) != pl.col(forward[0]))
        if nearest_ties == "error" and collect(plan.filter(equidistant).select(pl.len())).item():
            _fail("AMBIGUOUS_MATCH", "Nearest candidates are equidistant; declare nearest_ties='backward' or 'forward'.")
        choose_back = pl.col(backward[0]).is_not_null() & (pl.col(forward[0]).is_null() | (bd < fd) | ((bd == fd) & pl.lit(nearest_ties != "forward")))
        plan = plan.with_columns([pl.when(choose_back).then(pl.col(back)).otherwise(pl.col(fwd)).alias(destination) for back, fwd, destination in zip(backward[1:], forward[1:], destinations)] + [(pl.col(backward[0]).is_not_null() | pl.col(forward[0]).is_not_null()).replace(False, None).alias(marker)]).drop(backward + forward)
    else:
        right = source.with_columns(pl.lit(True).alias(marker)).sort(by + [on], maintain_order=True)
        plan = left.join_asof(right, on=on, by=by or None, strategy=strategy, tolerance=tolerance, allow_exact_matches=exact, suffix=suffix, check_sortedness=False)
    if unmatched == "error" and collect(plan.filter(pl.col(marker).is_null()).select(pl.len())).item():
        _fail("UNMATCHED_ROWS", "Some left observations have no match within the declared subject/tolerance.")
    if unmatched == "drop":
        plan = plan.filter(pl.col(marker).is_not_null())
    return plan.sort(row).drop([row, marker])


@operation(returns=("table",), recipe="yes")
def window(data, by, order_by, metrics, *, ties):
    """Compute subject-local lag/lead, bounded fill, difference/cumulative/rolling features; preserve source row order."""
    by = _names(by, "by", empty=True)
    order = _names(order_by, "order_by")
    require_columns(data, by + order)
    _tie_check(data, by, order, ties)
    if not isinstance(metrics, dict) or not metrics:
        _fail("INVALID_OPTION", "metrics must map output names to window specifications.")
    schema = data.collect_schema()
    row = temp_name(data)
    plan = data.with_row_index(row).sort(by + order, maintain_order=True)
    exprs = []
    for output, spec in metrics.items():
        _output(data, output)
        if not isinstance(spec, dict) or set(spec) - {"column", "method", "n", "size", "period", "min_count", "ddof", "closed", "nulls"}:
            _fail("INVALID_OPTION", "Unknown window specification options.")
        name, method = spec.get("column"), spec.get("method")
        require_columns(data, [name] if isinstance(name, str) else [])
        if name not in schema:
            _fail("UNKNOWN_COLUMN", "Window metric requires an observed column.", column=name)
        _choice(method, ("lag", "lead", "fill_forward", "fill_backward", "difference", "cum_sum", "rolling_sum", "rolling_mean", "rolling_min", "rolling_max", "rolling_std"), "method")
        allowed = {"column", "method", "nulls"}
        if method in {"lag", "lead", "fill_forward", "fill_backward", "difference"}:
            allowed.add("n")
        elif method.startswith("rolling_"):
            allowed.update({"size", "period", "min_count", "closed"})
            if method == "rolling_std":
                allowed.add("ddof")
        if set(spec) - allowed:
            _fail("INVALID_OPTION", "Window options do not apply to the declared method.", options=sorted(set(spec) - allowed))
        _choice(spec.get("nulls"), ("keep", "error"), "window nulls")
        if spec["nulls"] == "error":
            _nonmissing(data, [name])
        value = clean(name, schema[name])
        if method.startswith("rolling_") and not schema[name].is_numeric():
            _fail("NON_NUMERIC_COLUMN", "Rolling windows require numeric observations.", column=name)
        if method not in {"lag", "lead", "fill_forward", "fill_backward", "rolling_min", "rolling_max"}:
            if not schema[name].is_numeric():
                _fail("NON_NUMERIC_COLUMN", "Window arithmetic requires numeric observations.", column=name)
            if schema[name].is_integer() and method in {"difference", "cum_sum", "rolling_sum"}:
                _sum_safe(data, name, schema[name])
                value = value.cast(pl.Int128)
            else:
                require_float_exactness(data, [name])
                value = value.cast(pl.Float64)
        if method in {"fill_forward", "fill_backward"}:
            _integer(spec.get("n"), "n", minimum=1)
            expr = value.fill_null(strategy="forward" if method == "fill_forward" else "backward", limit=spec["n"])
        elif method in {"lag", "lead", "difference"}:
            n = spec.get("n")
            _integer(n, "n", minimum=1)
            expr = value.shift(-n if method == "lead" else n) if method != "difference" else value - value.shift(n)
        elif method == "cum_sum":
            expr = value.cum_sum()
        else:
            minimum = spec.get("min_count")
            _integer(minimum, "min_count", minimum=1)
            period = spec.get("period")
            size = spec.get("size")
            kwargs = {"min_samples": minimum}
            if method == "rolling_std":
                _integer(spec.get("ddof"), "ddof")
                kwargs["ddof"] = spec["ddof"]
            if period is not None:
                if size is not None or len(order) != 1:
                    _fail("INVALID_OPTION", "Time rolling needs one order column and no row size.")
                _duration(period, "period")
                if not isinstance(schema[order[0]], (pl.Date, pl.Datetime)):
                    _fail("INVALID_DTYPE", "Time rolling requires a Date/Datetime order column.")
                _choice(spec.get("closed"), ("left", "right", "both", "none"), "closed")
                expr = getattr(value, method + "_by")(order[0], window_size=period, closed=spec["closed"], **kwargs)
            else:
                _integer(size, "size", minimum=1)
                if minimum > size or "closed" in spec:
                    _fail("INVALID_OPTION", "Row rolling needs min_count <= size and no closed option.")
                expr = getattr(value, method)(window_size=size, **kwargs)
        if by:
            expr = expr.over(by)
        exprs.append(expr.alias(output))
    return plan.with_columns(exprs).sort(row).drop(row)


def _units(data, unit, groups, strata, weights=None, time=None, group_time="error"):
    _choice(unit, ("row", "group"), "unit")
    groups = _names(groups, "groups", empty=True)
    strata = _names(strata, "strata", empty=True)
    if set(groups) & set(strata):
        _fail("INVALID_OPTION", "groups and strata must be disjoint; a group cannot also be its own stratum.")
    if unit == "group" and not groups or unit == "row" and groups:
        _fail("INVALID_OPTION", "Group units require groups; row units require groups=[].")
    require_columns(data, groups + strata + ([weights] if weights else []) + ([time] if time else []))
    _nonmissing(data, groups + strata + ([time] if time else []))
    uid = temp_name(data, "__wrangle_unit")
    row = temp_name(data, "__wrangle_position")
    plan = data.with_row_index(row)
    keys = groups if unit == "group" else [row]
    attrs = list(dict.fromkeys(strata + ([weights] if weights else []) + ([time] if time else [])))
    if unit == "group" and attrs:
        checked = [name for name in attrs if name != time or group_time == "error"]
        if checked:
            conflicts = collect(plan.group_by(keys).agg([pl.col(name).n_unique().alias(name) for name in checked]).filter(pl.any_horizontal([pl.col(name) > 1 for name in checked])).select(pl.len())).item()
            if conflicts:
                _fail("GROUP_CONFLICT", "Each sampling group needs one stratum, weight, and declared time assignment.", groups=conflicts)
    reducers = [getattr(pl.col(name), group_time if name == time and group_time != "error" else "first")().alias(name) for name in attrs if name not in keys]
    units = plan.group_by(keys, maintain_order=True).agg(reducers).with_row_index(uid)
    return plan, units, keys, uid, row, strata


def _targets(units, strata, n, fraction, *, replacement):
    if (n is None) == (fraction is None):
        _fail("INVALID_OPTION", "Declare exactly one of n or fraction; targets apply per stratum.")
    if n is not None:
        _integer(n, "n")
    elif isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction < 0 or (not replacement and fraction > 1):
        _fail("INVALID_OPTION", "fraction must be finite/nonnegative and <= 1 without replacement.")
    size = pl.len().over(strata) if strata else pl.len()
    target = pl.lit(n, dtype=pl.UInt64) if n is not None else _floor_share(size, fraction)
    return units.with_columns(size.alias("__wrangle_size"), target.alias("__wrangle_target"))


@operation(returns=("table",), recipe="yes")
def sample(data, *, unit, groups, strata, seed, replacement, n=None, fraction=None, weights=None, draw_id=None):
    """Sample rows or whole groups, preserving all rows of each selected group.

    n/fraction targets apply per stratum; fraction floors the exact declared
    decimal share of ALL units before zero-weight exclusion. Zero weights are
    ineligible, positive weights must be constant within a group. Replacement
    requires draw_id, shared by all observations in one sampled group draw.
    Source order defines unit identity; replay requires the same input order.
    """
    seed_value(seed)
    _bool(replacement, "replacement")
    if replacement:
        _output(data, draw_id)
    elif draw_id is not None:
        _fail("INVALID_OPTION", "draw_id is only used with replacement.")
    plan, units, keys, uid, row, strata = _units(data, unit, groups, strata, weights)
    reserved = {"__wrangle_size", "__wrangle_target", "__wrangle_score", "__wrangle_rank", "__wrangle_draw", "__wrangle_cdf"}
    if reserved & set(data.collect_schema().names()):
        _fail("OUTPUT_COLLISION", "Rename reserved sampling fields before sampling.", columns=sorted(reserved & set(data.collect_schema().names())))
    units = _targets(units, strata, n, fraction, replacement=replacement)
    unit_random = _uniform(pl.struct(pl.col(uid), pl.lit("sample").alias("operation")), seed, positive=weights is not None)
    if weights is not None:
        if not isinstance(weights, str) or not data.collect_schema()[weights].is_numeric():
            _fail("NON_NUMERIC_COLUMN", "weights must name a numeric field.")
        _nonmissing(units, [weights])
        require_float_exactness(units, [weights])
        if collect(units.filter(~pl.col(weights).cast(pl.Float64).is_finite() | (pl.col(weights) < 0)).select(pl.len())).item():
            _fail("INVALID_WEIGHT", "Sampling weights must be finite and nonnegative.")
        zero_mass = units.group_by(strata).agg(pl.col("__wrangle_target").first().alias("target"), (pl.col(weights) > 0).sum().alias("eligible")) if strata else units.select(pl.col("__wrangle_target").first().alias("target"), (pl.col(weights) > 0).sum().alias("eligible"))
        if collect(zero_mass.filter((pl.col("target") > 0) & (pl.col("eligible") == 0)).select(pl.len())).item():
            _fail("INVALID_WEIGHT", "A requested stratum has no positive sampling weight.")
        units = units.filter(pl.col(weights) > 0)
        score = -(-unit_random.log()).log() + pl.col(weights).cast(pl.Float64).log()
    else:
        score = unit_random
    if n and not collect(units.select(pl.len())).item():
        _fail("INSUFFICIENT_SAMPLE", "No eligible sampling units exist for the requested n.")
    if not replacement:
        count = pl.len().over(strata) if strata else pl.len()
        if collect(units.filter(pl.col("__wrangle_target") > count).select(pl.len())).item():
            _fail("INSUFFICIENT_SAMPLE", "A stratum has fewer eligible units than its requested target.")
        selected = units.with_columns(score.alias("__wrangle_score")).sort(strata + ["__wrangle_score", uid], descending=[False] * len(strata) + [True, False]).with_columns((pl.int_range(pl.len()).over(strata) if strata else pl.int_range(pl.len())).alias("__wrangle_rank")).filter(pl.col("__wrangle_rank") < pl.col("__wrangle_target")).select(keys)
        return plan.join(selected, on=keys, how="semi", maintain_order="left").drop(row)
    # Replacement is a native inverse-CDF join, with one stream per stratum.
    if weights is None:
        mass = pl.lit(1.0)
    else:
        mass = (pl.col(weights).cast(pl.Float64).log() - (pl.col(weights).cast(pl.Float64).log().max().over(strata) if strata else pl.col(weights).cast(pl.Float64).log().max())).exp()
    mass_name = temp_name(data, "__wrangle_mass")
    units = units.with_columns(mass.alias(mass_name)).sort(strata + [uid])
    cumsum = pl.col(mass_name).cum_sum().over(strata) if strata else pl.col(mass_name).cum_sum()
    units = units.with_columns(cumsum.alias("__wrangle_cdf"))
    total = pl.col("__wrangle_cdf").last().over(strata) if strata else pl.col("__wrangle_cdf").last()
    units = units.with_columns((pl.col("__wrangle_cdf") / total).alias("__wrangle_cdf"))
    increment = pl.col("__wrangle_cdf").diff().over(strata) if strata else pl.col("__wrangle_cdf").diff()
    if collect(units.filter((increment <= 0).fill_null(False) | (pl.col("__wrangle_cdf") <= 0)).select(pl.len())).item():
        _fail("LOSSY_WEIGHT", "Some positive probabilities collapse in Float64 cumulative weights; rescale or stratify explicitly.")
    if collect(units.filter(pl.col(mass_name) == 0).select(pl.len())).item():
        _fail("LOSSY_WEIGHT", "The weight range underflows Float64 probabilities; rescale or stratify explicitly.")
    group = units.group_by(strata, maintain_order=True).agg(pl.col("__wrangle_target").first()) if strata else units.select(pl.col("__wrangle_target").first())
    draws = _explode(group.filter(pl.col("__wrangle_target") > 0).with_columns(pl.int_ranges(0, pl.col("__wrangle_target"), dtype=pl.UInt64).alias("__wrangle_draw")), ["__wrangle_draw"])
    random_key = pl.struct([pl.col(name) for name in strata] + [pl.col("__wrangle_draw")])
    draws = draws.with_columns(_uniform(random_key, seed).alias("__wrangle_cdf")).sort(strata + ["__wrangle_cdf"])
    selections = draws.join_asof(units, on="__wrangle_cdf", by=strata or None, strategy="forward", check_sortedness=False).with_row_index(draw_id)
    if collect(selections.filter(pl.col(uid).is_null()).select(pl.len())).item():
        _fail("LOSSY_WEIGHT", "A random draw has no representable probability interval.")
    selections = selections.select(keys + [draw_id])
    return selections.join(plan, on=keys, how="inner", maintain_order="left").sort([draw_id, row]).drop(row)


@operation(returns=("table",), recipe="yes")
def partition(data, *, output, labels, groups, strata, method, fractions=None, boundaries=None, time=None, group_time="error", closed="left", seed=None):
    """Annotate rows with group-safe random or chronological partitions; retain every observation."""
    _output(data, output)
    if not isinstance(labels, list) or len(labels) < 2 or any(not isinstance(label, str) or not label for label in labels) or len(labels) != len(set(labels)):
        _fail("INVALID_OPTION", "labels require at least two distinct nonempty strings.")
    _choice(method, ("random", "chronological"), "method")
    _choice(group_time, ("error", "min", "max"), "group_time")
    unit = "group" if groups else "row"
    if method == "random":
        seed_value(seed)
        if boundaries is not None or time is not None or group_time != "error":
            _fail("INVALID_OPTION", "Random partition takes fractions and seed, without chronology options.")
        if not isinstance(fractions, list) or len(fractions) != len(labels) or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
            _fail("INVALID_OPTION", "fractions require one positive share per label, summing to one.")
    else:
        if fractions is not None or seed is not None:
            _fail("INVALID_OPTION", "Chronological partition takes boundaries/time and no randomness.")
        if not isinstance(time, str) or not isinstance(boundaries, list) or len(boundaries) != len(labels) - 1:
            _fail("INVALID_OPTION", "Chronological partitions need a time column and one boundary between adjacent labels.")
        _choice(closed, ("left", "right"), "closed")
    plan, units, keys, uid, row, strata = _units(data, unit, groups, strata, time=time, group_time=group_time)
    if method == "random":
        score = temp_name(data, "__wrangle_partition_score")
        rank = temp_name(data, "__wrangle_partition_rank")
        units = units.with_columns(pl.struct(pl.col(uid)).hash(seed=seed).alias(score)).sort(strata + [score, uid])
        position = pl.int_range(pl.len()).over(strata) if strata else pl.int_range(pl.len())
        size = pl.len().over(strata) if strata else pl.len()
        units = units.with_columns(position.alias(rank))
        expr = pl.lit(labels[-1])
        cumulative = [sum((Fraction(str(f)) for f in fractions[:i + 1]), Fraction(0)) for i in range(len(fractions) - 1)]
        for i in reversed(range(len(cumulative))):
            expr = pl.when(pl.col(rank) < _floor_share(size, cumulative[i])).then(pl.lit(labels[i])).otherwise(expr)
    else:
        dtype = data.collect_schema()[time]
        if not (dtype.is_numeric() or isinstance(dtype, (pl.Date, pl.Datetime))):
            _fail("INVALID_DTYPE", "Chronological boundaries require numeric, Date, or Datetime time values.")
        literals = []
        for boundary in boundaries:
            if isinstance(dtype, (pl.Date, pl.Datetime)):
                if not isinstance(boundary, str):
                    _fail("INVALID_OPTION", "Temporal boundaries must be ISO date/datetime strings.")
                if isinstance(dtype, pl.Datetime):
                    fractional = re.search(r"\.([0-9]+)", boundary)
                    precision = {"ms": 3, "us": 6, "ns": 9}[dtype.time_unit]
                    if fractional and any(char != "0" for char in fractional.group(1)[precision:]):
                        _fail("LOSSY_CAST", "Boundary precision exceeds the time column's declared unit.", boundary=boundary, time_unit=dtype.time_unit)
                literal = pl.lit(boundary).str.to_date(strict=True) if isinstance(dtype, pl.Date) else pl.lit(boundary).str.to_datetime(time_unit=dtype.time_unit, time_zone=dtype.time_zone, strict=True)
            else:
                if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not math.isfinite(boundary):
                    _fail("INVALID_OPTION", "Numeric boundaries must be finite numbers.")
                literal = pl.lit(boundary).cast(dtype, strict=True)
            try:
                # This evaluates caller-declared metadata, never observed rows.
                pl.select(literal).item()
                if dtype.is_numeric():
                    original = pl.Series("boundary", [boundary])
                    roundtrip = literal.cast(original.dtype, strict=False)
                    if not pl.select(roundtrip.eq_missing(pl.lit(boundary, dtype=original.dtype))).item():
                        _fail("LOSSY_CAST", "A boundary cannot be represented exactly in the time column dtype.", boundary=boundary)
            except (pl.exceptions.PolarsError, ValueError) as error:
                if isinstance(error, WrangleError):
                    raise
                _fail("INVALID_OPTION", "A boundary cannot be parsed into the time column dtype.", error=str(error))
            literals.append(literal)
        try:
            increasing = pl.select([literals[i] < literals[i + 1] for i in range(len(literals) - 1)]).row(0) if len(literals) > 1 else [True]
        except (pl.exceptions.PolarsError, ValueError) as error:
            _fail("INVALID_OPTION", "Boundaries cannot be parsed to the declared time dtype.", error=str(error))
        if not all(increasing):
            _fail("INVALID_OPTION", "Chronological boundaries must be strictly increasing.")
        expr = pl.lit(labels[-1])
        for i in reversed(range(len(literals))):
            condition = pl.col(time) < literals[i] if closed == "left" else pl.col(time) <= literals[i]
            expr = pl.when(condition).then(pl.lit(labels[i])).otherwise(expr)
    assigned = units.with_columns(expr.alias(output)).select(keys + [output])
    return plan.join(assigned, on=keys, how="left", validate="m:1", maintain_order="left").sort(row).drop(row)


OPERATIONS = {function.__name__: function for function in (sort, deduplicate, concat, pivot, unpivot, explode, unnest, aggregate, join_asof, window, sample, partition)}
