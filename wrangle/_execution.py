"""Observation and semantic contracts shared by every preparation entrypoint."""
from __future__ import annotations

import json
import polars as pl
from ._core import WrangleError, missing, require_columns
from ._storage import DiskTable, collect

STEP_METADATA = {"op", "reason", "allow_expand", "units", "descriptions", "key", "unit_column"}
GRAIN_OPERATIONS = {"aggregate", "pivot", "unpivot", "explode", "concat"}
IDENTITY_OPERATIONS = {"cast", "clean_text", "recode", "derive", "select", "rename", "deduplicate", "unnest", "sort", "filter", "partition", "join", "join_asof", "sample"}


def key_columns(value, *, empty=True):
    value = [value] if isinstance(value, str) else value
    if not isinstance(value, list) or (not empty and not value) or any(not isinstance(name, str) or not name for name in value) or len(value) != len(set(value)):
        raise WrangleError("INVALID_RECIPE", "Observation keys must contain distinct nonempty column names.")
    return value


def source_contract(data, declaration, inherited):
    if not isinstance(declaration, dict) or set(declaration) - {"key", "units", "descriptions"}:
        raise WrangleError("INVALID_RECIPE", "Source contracts declare key, units and descriptions only.")
    result = {name: inherited.get(name, [] if name == "key" else {}).copy() for name in ("key", "units", "descriptions")}
    for name in ("units", "descriptions"):
        values = declaration.get(name, {})
        if not isinstance(values, dict) or any(not isinstance(field, str) or not field or not isinstance(value, str) or not value for field, value in values.items()):
            raise WrangleError("INVALID_RECIPE", "Scientific declarations require named fields and nonempty strings.")
        require_columns(data, list(values))
        conflicts = {field: {"parent": result[name][field], "declared": value} for field, value in values.items() if field in result[name] and result[name][field] != value}
        if conflicts:
            raise WrangleError("SOURCE_CONTRACT_MISMATCH", "The source declaration conflicts with verified parent semantics; perform an explicit transformation.", {"declaration": name, "columns": conflicts})
        result[name].update(values)
    if "key" in declaration:
        key = key_columns(declaration["key"])
        if inherited.get("key") and key != inherited["key"]:
            raise WrangleError("SOURCE_CONTRACT_MISMATCH", "Retain the verified parent key or declare an explicit step transition.")
        require_columns(data, key)
        result["key"] = key
    return result


def compatible_sources(op, before, before_units, before_descriptions, source_names, semantics, source_data):
    """Reconcile declarations only for fields physically present in each batch."""
    if op != "concat":
        return before_units, before_descriptions
    batches = [("input", before, before_units, before_descriptions)] + [
        (name, source_data[name], semantics[name]["units"], semantics[name]["descriptions"])
        for name in source_names
    ]
    merged = [{}, {}]
    for kind, offset in (("units", 2), ("descriptions", 3)):
        fields = set().union(*(set(batch[offset]) for batch in batches))
        for field in fields:
            known = {batch[0]: batch[offset][field] for batch in batches if field in batch[1].columns and field in batch[offset]}
            if len(set(known.values())) > 1:
                raise WrangleError("SOURCE_CONTRACT_MISMATCH", "Appended fields have incompatible declared units or meanings.", {"column": field, "declaration": kind, "sources": known})
            if kind == "units":
                missing = [batch[0] for batch in batches if field in batch[1].columns and field not in batch[offset]]
                if missing:
                    raise WrangleError("UNRESOLVED_SOURCE_UNITS", "Declare each batch's measurement units before combining it.", {"column": field, "sources": missing})
            if known and (kind == "units" or all(field not in batch[1].columns or field in batch[offset] for batch in batches)):
                merged[offset - 2][field] = next(iter(known.values()))
    return tuple(merged)


def validate_unit_references(data, units):
    """A row-specific unit is a required String field, never an absent annotation."""
    for field, unit in units.items():
        if field not in data.columns or not unit.startswith("@"):
            continue
        column = unit[1:]
        if not column or column not in data.columns or data.schema[column] != pl.String:
            raise WrangleError("UNIT_MISMATCH", "Retain the String field referenced by a measurement's row-specific units.", {"column": field, "unit_field": column})
        invalid = collect(data.lazy().select((pl.col(column).is_null() | (pl.col(column) == "") | pl.col(column).str.starts_with("@")).sum())).item()
        if invalid:
            raise WrangleError("UNIT_MISMATCH", "Every row-specific unit must be a nonempty physical-unit declaration.", {"column": field, "unit_field": column, "affected_rows": invalid})


def _reduction_unit(before, after, unit, groups, *, window=False):
    if not unit.startswith("@"):
        return unit
    column = unit[1:]
    require_columns(before, column)
    if groups:
        incompatible = collect(before.lazy().group_by(groups).agg(pl.col(column).n_unique().alias("__unit_count")).filter(pl.col("__unit_count") > 1).select(pl.len())).item()
    else:
        incompatible = collect(before.lazy().select((pl.col(column).n_unique() > 1).cast(pl.UInt64))).item()
    if incompatible:
        raise WrangleError("UNIT_MISMATCH", "Reduce measurements only within groups with one physical unit; group by the unit field or convert explicitly.", {"unit_field": column, "groups": groups, "affected_groups": incompatible})
    if column in after.columns:
        return unit
    observed = collect(before.lazy().select(pl.col(column).unique(maintain_order=True)).head(2))
    if observed.height == 1:
        return observed.item()
    raise WrangleError("UNIT_MISMATCH", "Retain the grouping unit field in the reduction output.", {"unit_field": column})



def _matching_units(left, right, left_unit, right_unit, pairs):
    if left_unit is not None and right_unit is not None and left_unit.startswith("@") and right_unit.startswith("@") and (left_unit[1:], right_unit[1:]) in pairs:
        return True
    def physical(table, unit):
        if unit is None or not unit.startswith("@"):
            return unit
        observed = collect(table.lazy().select(pl.col(unit[1:]).unique(maintain_order=True)).head(2))
        if observed.height != 1:
            raise WrangleError("UNIT_MISMATCH", "Join row-specific measurement axes using their unit fields as matching keys, or convert to one physical unit first.", {"unit_field": unit[1:]})
        return observed.item()
    return physical(left, left_unit) == physical(right, right_unit)

def semantic_effects(before, data, op, step, units, descriptions, right=None):
    """Propagate only meanings implied by declared operations and source contracts."""
    original_units = dict(units)
    units, descriptions = dict(units), dict(descriptions)
    variable_units = None
    if op == "rename":
        renames = step["columns"]
        units = {renames.get(name, name): ("@" + renames.get(unit[1:], unit[1:]) if unit.startswith("@") else unit) for name, unit in units.items()}
        descriptions = {renames.get(name, name): meaning for name, meaning in descriptions.items()}
    if op in {"join", "join_asof"} and right is not None:
        if op == "join":
            from ._recipe_join import output_mapping, key_mapping
            parameters = {name: value for name, value in step.items() if name not in STEP_METADATA and name != "source"}
            outputs = output_mapping(before.columns, right["columns"], parameters)
            pairs = key_mapping(parameters)
            coalesced = {r: outputs[r] for l, r in pairs} if step["coalesce"] else {}
            if step["coalesce"]:
                axis_names = {l: outputs[r] for l, r in pairs}
                units = {axis_names.get(name, name): ("@" + axis_names.get(unit[1:], unit[1:]) if unit.startswith("@") else unit) for name, unit in units.items()}
                # Right coalescing selects the right source axis role.
                if step["how"] == "right":
                    for l, r in pairs:
                        descriptions.pop(l, None)
        else:
            suffix = step.get("suffix", "__right")
            names = key_columns(step["on"]) + key_columns(step["by"])
            pairs = list(zip(names, names))
            coalesced = dict(pairs)
            outputs = {name: name if name in coalesced or name not in before.columns else name + suffix for name in right["columns"]}
        for left_key, right_key in pairs:
            left_unit, right_unit = original_units.get(left_key), right["units"].get(right_key)
            if not _matching_units(before, right["data"], left_unit, right_unit, pairs):
                raise WrangleError("SOURCE_CONTRACT_MISMATCH", "Declare compatible units for both matching axes before joining.", {"left": left_key, "right": right_key, "left_unit": left_unit, "right_unit": right_unit})
        for kind, mapping in (("units", units), ("descriptions", descriptions)):
            for name, value in right[kind].items():
                output = outputs.get(name)
                if output is None or output not in data.columns:
                    continue
                if name in coalesced and kind == "descriptions":
                    if op == "join" and step["how"] == "right":
                        mapping[output] = value
                    elif op == "join" and step["how"] == "full" and mapping.get(output) != value:
                        mapping.pop(output, None)
                    continue
                if name in coalesced and kind == "units" and (op == "join_asof" or step.get("how") in {"left", "inner"}):
                    if output in original_units:
                        continue
                if value.startswith("@") and kind == "units":
                    value = "@" + outputs.get(value[1:], value[1:])
                mapping[output] = value
    if op == "unpivot":
        fields = step["on"] if isinstance(step["on"], list) else [step["on"]]
        declared = {name: units[name] for name in fields if name in units}
        column = step.get("unit_column")
        if any(unit.startswith("@") for unit in declared.values()):
            raise WrangleError("UNIT_MISMATCH", "Resolve row-specific measurement units before another unpivot.")
        if column is not None:
            if not isinstance(column, str) or not column or column in data.columns:
                raise WrangleError("COLUMN_EXISTS", "unit_column must name an absent output field.")
            if set(declared) != set(fields):
                raise WrangleError("UNDECLARED_UNITS", "Every unpivoted measurement needs units before creating a unit column.", {"columns": fields})
            plan = data.lazy().with_columns(pl.col(step["variable"]).replace_strict(declared, return_dtype=pl.String).alias(column))
            data = DiskTable(plan) if isinstance(data, DiskTable) else collect(plan)
            units[step["value"]] = "@" + column
            variable_units = {"value": step["value"], "variable": step["variable"], "unit_column": column, "mapping": declared}
        elif len(set(declared.values())) > 1 or declared and len(declared) != len(fields):
            raise WrangleError("UNIT_MISMATCH", "Unpivot compatible measurements or declare unit_column with units for every variable.", {"units": declared})
        elif declared:
            units[step["value"]] = next(iter(declared.values()))
        descriptions.pop(step["value"], None)
    if op == "pivot":
        for name in key_columns(step["values"]):
            if name not in original_units:
                continue
            unit = original_units[name]
            for category in step["domain"]:
                output = f'{name}{step.get("separator", "__")}{category}'
                if unit.startswith("@"):
                    observed = collect(before.lazy().filter(pl.col(step["on"]) == pl.lit(category)).select(pl.col(unit[1:]).unique(maintain_order=True)).head(2))
                    if observed.height > 1:
                        raise WrangleError("UNIT_MISMATCH", "A fixed pivot field must have one physical unit across observations.", {"column": output})
                    if not observed.height:
                        continue
                    units[output] = observed.item()
                else:
                    units[output] = unit
                descriptions.pop(output, None)
    if op in {"aggregate", "window"}:
        groups = key_columns(step["by"])
        # Calendar windows alter the timestamp's role; its raw description expires.
        if op == "aggregate" and step.get("time"):
            descriptions.pop(step["time"], None)
        for output, metric in step["metrics"].items():
            method, field = metric["method"], metric.get("column")
            units.pop(output, None)
            descriptions.pop(output, None)
            if method in {"len", "count", "n_unique"}:
                units[output] = "1"
            elif field in original_units:
                unit = _reduction_unit(before, data, original_units[field], groups, window=op == "window")
                if method not in {"var", "rolling_var"}:
                    units[output] = unit
    if op == "convert_unit":
        field = step["column"]
        actual = original_units.get(field)
        if actual and actual.startswith("@"):
            count = collect(before.lazy().select((pl.col(actual[1:]) != step["from_unit"]).sum())).item()
            if count:
                raise WrangleError("UNIT_MISMATCH", "A unit conversion must match every row's declared input unit.", {"column": field, "affected_rows": count})
        elif actual and actual != step["from_unit"]:
            raise WrangleError("UNIT_MISMATCH", "Conversion input units disagree with the protocol.", {"column": field, "declared": actual, "from_unit": step["from_unit"]})
        units[field] = step["to_unit"]
    if op == "standardize":
        fields = step.get("columns", step.get("cols", []))
        fields = key_columns(fields) if fields else [name for name, dtype in before.schema.items() if dtype.is_numeric()]
        for field in fields:
            units[field] = "1"
            descriptions.pop(field, None)
    return data, units, descriptions, variable_units



def validate_unit_basis(before, after, op, old_units, new_units, input_key):
    """A unit reference cannot change measurement basis through label cleanup."""
    structural = {"rename", "select", "filter", "sort", "deduplicate", "join", "join_asof", "concat", "pivot", "unpivot", "explode", "unnest", "aggregate", "window", "sample", "partition"}
    if op in structural:
        return
    checked = set()
    for field, unit in old_units.items():
        if not unit.startswith("@") or field not in after.columns or not new_units.get(field, "").startswith("@"):
            continue
        old_column, new_column = unit[1:], new_units[field][1:]
        if (old_column, new_column) in checked or old_column not in before.columns or new_column not in after.columns:
            continue
        checked.add((old_column, new_column))
        if input_key and set(input_key) <= set(after.columns) and before.select(input_key).schema.dtypes() == after.select(input_key).schema.dtypes():
            old_name, new_name = "__wrangle_old_unit", "__wrangle_new_unit"
            while old_name in input_key:
                old_name += "_"
            while new_name in input_key:
                new_name += "_"
            matched = before.lazy().select(*input_key, pl.col(old_column).alias(old_name)).join(after.lazy().select(*input_key, pl.col(new_column).alias(new_name)), on=input_key, how="inner", validate="1:m", maintain_order="left")
            changed = collect(matched.select((~pl.col(old_name).eq_missing(pl.col(new_name))).sum())).item()
        elif before.height == after.height:
            changed = not before.select(old_column).rename({old_column: "unit"}).equals(after.select(new_column).rename({new_column: "unit"}))
        else:
            raise WrangleError("UNIT_MISMATCH", "Use a canonical identity-preserving operation or declare keys to retain row-specific unit evidence.")
        if changed:
            raise WrangleError("UNIT_MISMATCH", "Unit-label cleanup cannot change a measurement's physical basis; convert the measurement explicitly first.", {"column": field, "unit_field": old_column, "affected_rows": changed})


def _parent_columns(op, step, input_key, before_columns, right_columns):
    if op == "rename":
        return [step["columns"].get(name, name) for name in input_key]
    if op == "join" and step.get("coalesce"):
        from ._recipe_join import output_mapping, key_mapping
        options = {name: value for name, value in step.items() if name not in STEP_METADATA and name != "source"}
        outputs = output_mapping(before_columns, right_columns, options)
        renames = {l: outputs[r] for l, r in key_mapping(options)}
        return [renames.get(name, name) for name in input_key]
    return input_key

def observation_effects(before, after, op, step, input_key, *, aggregates=False, right=None, evidence=None):
    """Measure coverage and multiplicity independently of net row-count changes."""
    reduction = aggregates and "key" in step
    excluded = None
    counts = {"excluded_observations": 0, "expanded_parents": 0, "introduced_observations": 0}
    if input_key and not reduction and op != "concat":
        parents = _parent_columns(op, step, input_key, before.columns, right.columns if right is not None else [])
        if set(parents) <= set(after.columns) and before.select(input_key).schema.dtypes() == after.select(parents).schema.dtypes():
            old = before.select(input_key).rename(dict(zip(input_key, parents)))
            present = after.select(parents).unique(maintain_order=True)
            lost = old.join(present, on=parents, how="anti", maintain_order="left")
            counts["excluded_observations"] = lost.height
            if lost.height:
                lost_keys = lost.rename(dict(zip(parents, input_key)))
                excluded = evidence.record(lost_keys.lazy(), "excluded_keys") if evidence is not None else json.loads(lost_keys.write_json())
            repeated = collect(after.lazy().group_by(parents).len().filter(pl.col("len") > 1).select(pl.len())).item()
            counts["expanded_parents"] = repeated
            if op == "join" and step.get("how") in {"right", "full"}:
                new = after.join(old, on=parents, how="anti", maintain_order="left")
                counts["introduced_observations"] = new.height
    if op == "unpivot" and step.get("nulls") == "drop":
        fields = key_columns(step["on"])
        counts["excluded_cells"] = collect(before.lazy().select(pl.sum_horizontal([pl.col(name).is_null().cast(pl.UInt64) for name in fields]).sum())).item()
    has_loss = counts["excluded_observations"] or counts.get("excluded_cells", 0) or after.height < before.height and not reduction
    # Row-preserving identity corrections are explicit one-to-one mappings, not exclusions.
    if "key" in step and before.height == after.height and op in {"cast", "clean_text", "recode", "derive", "unnest", "select"}:
        has_loss = False
        counts["excluded_observations"] = 0
        counts["expanded_parents"] = 0
        counts["introduced_observations"] = 0
        excluded = None
    if has_loss and not step.get("reason"):
        raise WrangleError("UNDECLARED_LOSS", "An exclusion requires a reason in the recipe.", counts)
    if (counts["expanded_parents"] or counts["introduced_observations"] or after.height > before.height) and step.get("allow_expand") is not True:
        raise WrangleError("UNDECLARED_EXPANSION", "Set allow_expand=true to deliberately increase observations or parent multiplicity.", counts)
    return excluded, counts


def observation_transition(before, after, op, step, input_key, output_key, *, aggregates=False, sources=None, right=None, evidence=None):
    """Verify declared output grain and retain its source relation in the receipt."""
    declared = "key" in step
    if (op in GRAIN_OPERATIONS or op == "sample" and step.get("replacement")) and not declared:
        raise WrangleError("OBSERVATION_UNIT_CHANGED", "Declare step.key for the output observation unit.", {"operation": op})
    if not declared:
        if aggregates and input_key:
            raise WrangleError("OBSERVATION_UNIT_CHANGED", "Aggregations need an explicitly declared output observation key.")
        if output_key:
            original = before.select(input_key).rename(dict(zip(input_key, output_key)))
            if original.schema != after.select(output_key).schema:
                raise WrangleError("KEY_CHANGED", "Observation key types changed; declare a deliberate output-key transition.")
            new = after.select(output_key).join(original, on=output_key, how="anti", validate="m:m", maintain_order="left")
            if new.height:
                raise WrangleError("KEY_CHANGED", "The operation changed observation identities; declare a deliberate output-key transition.", {"affected_rows": new.height})
        return None
    if not output_key:
        raise WrangleError("OBSERVATION_UNIT_CHANGED", "An output observation-key transition must declare a nonempty key.")
    if op not in GRAIN_OPERATIONS | IDENTITY_OPERATIONS and not aggregates:
        raise WrangleError("UNSUPPORTED_KEY_TRANSITION", "Use a documented identity-preserving or grain-changing recipe operation.", {"operation": op})
    relation = {"input_key": input_key, "output_key": output_key, "operation": op}
    required = []
    if op == "aggregate":
        required = key_columns(step["by"])
        if step.get("time"):
            required += [step["time"]]
        if set(output_key) != set(required):
            raise WrangleError("OBSERVATION_UNIT_CHANGED", "Aggregate output keys must equal the declared grouping keys and time window.", {"expected": required})
        relation["group_by"] = required
    elif op == "pivot":
        required = key_columns(step["index"])
        if set(output_key) != set(required):
            raise WrangleError("OBSERVATION_UNIT_CHANGED", "Pivot output keys must equal the declared index.", {"expected": required})
        relation["group_by"] = required
    elif op == "unpivot":
        if not set(input_key) <= set(key_columns(step["index"])):
            raise WrangleError("OBSERVATION_UNIT_CHANGED", "Unpivot must retain the original key in its index.")
        required = [*input_key, step["variable"]]
        relation["parent_key"] = input_key
        relation["variable"] = step["variable"]
    elif op == "explode":
        required = [*input_key, step["index"]]
        relation["parent_key"] = input_key
        relation["element_index"] = step["index"]
    elif op == "concat":
        relation["sources"] = sources
        relation["provenance"] = step["provenance"]
        relation["labels"] = step["labels"]
    elif op == "sample" and step.get("replacement"):
        required = [*input_key, step["draw_id"]]
        relation["parent_key"] = input_key
        relation["draw_id"] = step["draw_id"]
    elif aggregates:
        required = key_columns(step.get("by", step.get("handler_col", step.get("dt_col", []))))
        if not required or set(output_key) != set(required):
            raise WrangleError("UNSUPPORTED_KEY_TRANSITION", "Use aggregate/pivot with a declared group key for this observation change.")
        relation["group_by"] = required
    elif input_key:
        if op == "rename":
            retained = [step["columns"].get(name, name) for name in input_key]
            if set(retained) == set(output_key):
                relation["renames"] = step["columns"]
                return relation
        if before.height == after.height and op in {"cast", "clean_text", "recode", "derive", "unnest", "select"}:
            # These operations preserve row position. Mapping values stay data.
            left = before.lazy().select(pl.struct(input_key).alias("input")).with_row_index("row")
            right = after.lazy().select(pl.struct(output_key).alias("output")).with_row_index("row")
            mapping = left.join(right, on="row", validate="1:1", maintain_order="left").drop("row")
            relation["identity_map"] = evidence.record(mapping, "identity_map") if evidence is not None else json.loads(collect(mapping).write_json())
        elif set(_parent_columns(op, step, input_key, before.columns, right.columns if right is not None else [])) <= set(after.columns):
            retained = _parent_columns(op, step, input_key, before.columns, right.columns if right is not None else [])
            old = before.lazy().select(input_key).rename(dict(zip(input_key, retained)))
            unknown = after.lazy().select(retained).join(old, on=retained, how="anti", maintain_order="left")
            parents = collect(unknown.select(pl.len())).item()
            if parents and op == "join" and step.get("how") in {"right", "full"} and right is not None:
                from ._recipe_join import output_mapping, key_mapping
                options = {name: value for name, value in step.items() if name not in STEP_METADATA and name != "source"}
                outputs = output_mapping(before.columns, right.columns, options)
                pairs = dict((l, r) for l, r in key_mapping(options))
                absent = pl.any_horizontal([missing(pl.col(name), after.schema[name]) for name in retained])
                with_parent = unknown.filter(~absent)
                if all(name in pairs and outputs[pairs[name]] == output for name, output in zip(input_key, retained)):
                    known_right = right.lazy().select([pl.col(pairs[name]).alias(output) for name, output in zip(input_key, retained)])
                    invalid = collect(with_parent.join(known_right, on=retained, how="anti", maintain_order="left").select(pl.len())).item()
                else:
                    invalid = collect(with_parent.select(pl.len())).item()
                if invalid:
                    raise WrangleError("KEY_CHANGED", "Introduced observations need identity retained from the joined source.")
                relation["introduced_source"] = sources[-1]
                relation["introduced_rows"] = parents
                relation["join_keys"] = [list(pair) for pair in key_mapping(options)]
            elif parents:
                raise WrangleError("KEY_CHANGED", "Output observations do not retain a known parent identity.")
            relation["parent_key"] = input_key
            if retained != input_key:
                relation["output_parent_key"] = retained
        else:
            raise WrangleError("UNSUPPORTED_KEY_TRANSITION", "The operation must retain parent identity or use a supported grain transition.")
    if not set(required) <= set(output_key):
        raise WrangleError("OBSERVATION_UNIT_CHANGED", "Output keys must retain the declared parent/element identity.", {"required": required})
    return relation
