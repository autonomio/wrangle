"""Native joins with checked keys, cardinality, unmatched sides and source order."""
from __future__ import annotations

import polars as pl
from ._storage import collect
from ._core import WrangleError, frame, missing, operation, require_columns


def _error(code, message, **details):
    raise WrangleError(code, message, details)


def _keys(value, name):
    value = [value] if isinstance(value, str) else value
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v for v in value) or len(value) != len(set(value)):
        _error("INVALID_ARGUMENT", f"{name} must contain distinct nonempty column names.")
    return value


def resolved_parameters(parameters):
    """Resolve JSON defaults and legacy aliases; usable by receipt generation."""
    defaults = dict(cardinality=None, validate=None, unmatched="error", on=None, left_on=None, right_on=None, how="left", suffix="_right", overlap="suffix", maintain_order="left", nulls="error", coalesce=None)
    unknown = set(parameters) - set(defaults)
    if unknown:
        _error("INVALID_ARGUMENT", "Unknown join options.", options=sorted(unknown))
    options = {**defaults, **parameters}
    if options["cardinality"] is not None and options["validate"] is not None and options["cardinality"] != options["validate"]:
        _error("INVALID_ARGUMENT", "cardinality and legacy validate aliases must agree.")
    options["cardinality"] = options["cardinality"] if options["cardinality"] is not None else options["validate"] if options["validate"] is not None else "m:1"
    if options["cardinality"] not in ("1:1", "1:m", "m:1", "m:m"):
        _error("INVALID_ARGUMENT", "cardinality requires 1:1, 1:m, m:1 or m:m.")
    how = options["how"]
    if how not in ("inner", "left", "right", "full", "semi", "anti"):
        _error("INVALID_ARGUMENT", "how requires inner, left, right, full, semi or anti.")
    if options["on"] is not None:
        if options["left_on"] is not None or options["right_on"] is not None:
            _error("INVALID_ARGUMENT", "Use either on or paired left_on/right_on.")
        options["on"] = _keys(options["on"], "on")
        left, right = options["on"], options["on"]
    else:
        left, right = _keys(options["left_on"], "left_on"), _keys(options["right_on"], "right_on")
        if len(left) != len(right):
            _error("INVALID_ARGUMENT", "left_on and right_on require equally many key fields.")
        options["left_on"], options["right_on"] = left, right
    if options["coalesce"] is None:
        options["coalesce"] = options["on"] is not None
    elif type(options["coalesce"]) is not bool:
        _error("INVALID_ARGUMENT", "coalesce must be boolean when declared.")
    if options["overlap"] not in ("error", "suffix") or not isinstance(options["suffix"], str) or not options["suffix"]:
        _error("INVALID_ARGUMENT", "Choose overlap='error'/'suffix' and a nonempty suffix.")
    if options["nulls"] not in ("error", "drop"):
        _error("INVALID_ARGUMENT", "Join nulls require error or explicit drop; null keys never match.")
    order = options["maintain_order"]
    if order not in ("left", "right", "left_right", "right_left"):
        _error("INVALID_ARGUMENT", "Choose a deterministic Polars source-order policy.")
    if how in ("right", "full") and order not in ("left_right", "right_left"):
        _error("INVALID_ARGUMENT", "right/full joins require explicit left_right or right_left order for both sources.")
    if how in ("semi", "anti") and order not in ("left", "left_right"):
        _error("INVALID_ARGUMENT", "semi/anti joins retain left observations and require left source ordering.")
    unmatched = options["unmatched"]
    if isinstance(unmatched, str):
        unmatched = {"left": unmatched, "right": "keep" if how in ("right", "full") else "drop"}
    elif not isinstance(unmatched, dict) or set(unmatched) != {"left", "right"}:
        _error("INVALID_ARGUMENT", "unmatched requires a left policy or an object with exactly left/right policies.")
    if any(policy not in ("error", "drop", "keep") for policy in unmatched.values()):
        _error("INVALID_ARGUMENT", "Unmatched policies require error, drop or keep.")
    allowed_left = {"left": ("error", "drop", "keep"), "inner": ("error", "drop"), "right": ("error", "drop"), "full": ("error", "drop", "keep"), "semi": ("error", "drop"), "anti": ("keep",)}[how]
    allowed_right = ("error", "drop", "keep") if how in ("right", "full") else ("error", "drop")
    if unmatched["left"] not in allowed_left or unmatched["right"] not in allowed_right:
        _error("INVALID_ARGUMENT", "Unmatched retention must agree with the join mode; anti requires left keep.", how=how, unmatched=unmatched)
    options["unmatched"] = unmatched
    return options


def key_mapping(parameters):
    """Return exact left/right key pairs from resolved JSON parameters."""
    options = resolved_parameters(parameters)
    return list(zip(options["on"] or options["left_on"], options["on"] or options["right_on"]))


def output_mapping(left_columns, right_columns, parameters):
    """Map every right field to its actual output, including coalesced keys."""
    options = resolved_parameters(parameters)
    if options["how"] in ("semi", "anti"):
        return {}
    pairs = dict((right, left) for left, right in key_mapping(options)) if options["coalesce"] else {}
    right_coalesced = options["how"] == "right" and options["coalesce"]
    retained_left = set(left_columns) - set(pairs.values()) if right_coalesced else set(left_columns)
    outputs = {}
    for name in right_columns:
        if name in pairs and not right_coalesced:
            outputs[name] = pairs[name]
        elif name in retained_left:
            if options["overlap"] == "error":
                _error("OVERLAPPING_COLUMNS", "Declare a suffix or rename overlapping right payload fields.", column=name)
            outputs[name] = name + options["suffix"]
        else:
            outputs[name] = name
    payload = [output for name, output in outputs.items() if right_coalesced or name not in pairs]
    if len(payload) != len(set(payload)) or set(payload) & retained_left:
        _error("OUTPUT_COLLISION", "The join suffix creates an existing or repeated output field; rename the source or change suffix.")
    return outputs


def _nonmissing(data, keys, side, policy):
    schema = data.collect_schema()
    absent = pl.any_horizontal([missing(pl.col(name), schema[name]) for name in keys])
    if policy == "error":
        count = collect(data.filter(absent).select(pl.len())).item()
        if count:
            _error("NULL_KEY", "Join keys require observed values; declare explicit exclusion before matching.", side=side, affected_rows=count)
        return data
    return data.filter(~absent)


def _unique(data, keys, side):
    count = collect(data.group_by(keys).len().filter(pl.col("len") > 1).select(pl.len())).item()
    if count:
        _error("JOIN_CARDINALITY", "Join keys violate the declared cardinality.", side=side, duplicate_keys=count)


def _unmatched(left, right, left_keys, right_keys):
    target = right.select(right_keys).unique(maintain_order=True)
    return collect(left.join(target, left_on=left_keys, right_on=right_keys, how="anti", maintain_order="left").select(pl.len())).item()


@operation(returns=("table",), recipe="yes")
def join(data, source, *, cardinality=None, unmatched="error", on=None, left_on=None, right_on=None, how="left", suffix="_right", overlap="suffix", maintain_order="left", nulls="error", coalesce=None, validate=None):
    """Join exact typed keys under declared cardinality and unmatched-side policies.

    Scalar unmatched applies to the left; the right permits unused rows for
    left/inner/semi/anti and retains unmatched rows for right/full. An object
    declares both sides. keep requires an outer side; anti requires left keep.
    nulls='drop' excludes missing-key rows on both sides, never matches nulls.
    right/full require explicit combined source order. Shared on keys coalesce
    by default; paired differently named keys remain separate unless declared.
    cardinality aliases legacy validate; their declarations must agree.
    """
    try:
        left, right = frame(data), frame(source)
        options = resolved_parameters(dict(cardinality=cardinality, validate=validate, unmatched=unmatched, on=on, left_on=left_on, right_on=right_on, how=how, suffix=suffix, overlap=overlap, maintain_order=maintain_order, nulls=nulls, coalesce=coalesce))
        pairs = key_mapping(options)
        left_keys, right_keys = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        require_columns(left, left_keys)
        require_columns(right, right_keys)
        ls, rs = left.collect_schema(), right.collect_schema()
        if any(ls[lk] != rs[rk] for lk, rk in pairs):
            _error("DTYPE_MISMATCH", "Join key dtypes differ; cast both sources explicitly before matching.", keys=pairs, columns=[lk for lk, rk in pairs if ls[lk] != rs[rk]])
        output_mapping(ls.names(), rs.names(), options)
        left, right = _nonmissing(left, left_keys, "left", nulls), _nonmissing(right, right_keys, "right", nulls)
        card = options["cardinality"]
        if card in ("1:1", "1:m"):
            _unique(left, left_keys, "left")
        if card in ("1:1", "m:1"):
            _unique(right, right_keys, "right")
        for side, inputs, targets, own, other in (("left", left, right, left_keys, right_keys), ("right", right, left, right_keys, left_keys)):
            if options["unmatched"][side] == "error":
                count = _unmatched(inputs, targets, own, other)
                if count:
                    _error("UNMATCHED_KEYS", "Some observations have no matching key on the other side.", side=side, affected_rows=count, keys=own)
        # Marker validity controls each unmatched side without relying on nullable
        # payload values. Names are collision-free schema metadata only.
        present = set(ls.names()) | set(rs.names()) | set(output_mapping(ls.names(), rs.names(), options).values())
        markers = []
        for base in ("__wrangle_left_match", "__wrangle_right_match"):
            while base in present:
                base += "_"
            present.add(base)
            markers.append(base)
        left = left.with_columns(pl.lit(True).alias(markers[0]))
        right = right.with_columns(pl.lit(True).alias(markers[1]))
        # Polars 1.34/1.44 do not support native validate on right/semi/anti.
        # Equivalent native uniqueness preflights above already enforce it.
        engine_validation = card if how in ("left", "inner", "full") else "m:m"
        plan = left.join(right, left_on=left_keys, right_on=right_keys, how=how, suffix=suffix, validate=engine_validation, coalesce=options["coalesce"], nulls_equal=False, maintain_order=maintain_order)
        if how in ("semi", "anti"):
            return plan.drop(markers[0])
        if options["unmatched"]["left"] == "drop":
            plan = plan.filter(pl.col(markers[1]).is_not_null())
        if options["unmatched"]["right"] == "drop":
            plan = plan.filter(pl.col(markers[0]).is_not_null())
        return plan.drop(markers)
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, WrangleError):
            raise
        _error("JOIN_FAILED", "The native join cannot satisfy its declared keys and policies.", error=str(error))


OPERATIONS = {"join": join}
