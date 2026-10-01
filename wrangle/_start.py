"""Observed files and explicit research answers become an editable YAML protocol.

This module drafts intent only. It never prepares data or approves a scientific
choice. The ordinary preparation engine performs every transformation and check.
"""
from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shlex
import shutil
import tempfile
from typing import Any

import polars as pl

from ._core import WrangleError, dtype_spec
from ._protocol import dump_recipe, load_recipe, normalize_recipe
from ._publication import publish_directory

_FIELDS = {"input", "observation", "key", "measurements", "missing", "exclusions", "matching", "descriptions", "source_options"}
_NUMERIC_CHOICES = {"Float64", "Int64", "keep"}
_UNRESOLVED_UNITS = {"none", "unknown", "?"}

QUESTION_DEFINITIONS = tuple(
    {"id": field, "field": field, "question": question, "prompt": question, "choices": choices}
    for field, question, choices in [
        ("input", "Which file contains the observations to prepare?", []),
        ("observation", "What does one row represent in your study?", []),
        ("key", "Which columns, together, identify one observation without repeats?", []),
        ("measurements", "Which columns are measurements, how should their numbers be represented, and what are their supplied units? Use 1 for dimensionless; unknown units remain unresolved. Use {} if there are no measurements.", ["Float64", "Int64", "keep"]),
        ("missing", "Which exact values mean missing in the observation file? Should missing values in the final table be kept or cause an error? Use codes: {} if there are no additional codes. No filling is performed.", ["keep", "error"]),
        ("matching", "Should metadata be attached? Declare matching columns, at most one metadata match per observation, and behavior for absent matches, unused metadata, and overlapping fields. Observation order is retained.", ["none", "attach"]),
        ("exclusions", "Should all observations be retained, or only approved quality-control values? Supply the protocol reason and behavior when QC is missing.", ["none", "keep_values"]),
    ]
)



def _fail(message, field, **details):
    raise WrangleError("START_ANSWER", message, {"field": field, **details})


def _mapping(value, fields, field, *, required=()):
    if not isinstance(value, dict) or set(value) - set(fields) or set(required) - set(value):
        _fail("Use the documented answer fields; unsupported choices are never ignored.", field, allowed=sorted(fields), required=sorted(required))


def _names(value, field):
    if not isinstance(value, list) or not value or any(not isinstance(name, str) or not name for name in value) or len(value) != len(set(value)):
        _fail("Name one or more distinct columns using a list.", field)


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        _fail("Write a nonempty answer in your own words.", field)


def _choice(value, choices, field):
    if not isinstance(value, str) or value not in choices:
        _fail("Choose one of the documented answers.", field, choices=sorted(choices))


def _scalars(value, field, *, nonempty=False):
    if not isinstance(value, list) or nonempty and not value or any(type(item) not in {str, int, float, bool} for item in value):
        _fail("Use a list of exact text, numeric, or Boolean values; null behavior is declared separately.", field)
    # JSON distinguishes true, 1 and 1.0. The native engine checks representability.
    import json
    rendered = [json.dumps(item, allow_nan=False) for item in value]
    if len(rendered) != len(set(rendered)):
        _fail("List each exact value once.", field)


def _validate_answers(answers):
    _mapping(answers, _FIELDS, "answers")
    if answers.get("input") is not None:
        _text(answers["input"], "input")
    if answers.get("observation") is not None:
        _text(answers["observation"], "observation")
    if answers.get("key") is not None:
        _names(answers["key"], "key")
    if answers.get("measurements") is not None:
        value = answers["measurements"]
        if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
            _fail("Map measurement columns to their numeric representation and supplied unit; use {} if there are none.", "measurements")
        for name, measurement in value.items():
            field = "measurements." + name
            _mapping(measurement, {"dtype", "unit"}, field)
            if measurement.get("dtype") is not None:
                _choice(measurement["dtype"], _NUMERIC_CHOICES, field + ".dtype")
            unit = measurement.get("unit")
            if unit is not None:
                _text(unit, field + ".unit")
                if unit.startswith("@"):
                    _fail("This guided path uses one supplied physical unit per measurement. Use an ordinary recipe for row-specific units.", field + ".unit")
    if answers.get("missing") is not None:
        value = answers["missing"]
        _mapping(value, {"codes", "action"}, "missing", required={"codes", "action"})
        _choice(value["action"], {"keep", "error"}, "missing.action")
        if not isinstance(value["codes"], dict) or any(not isinstance(name, str) or not name for name in value["codes"]):
            _fail("Map columns to exact missing codes; use {} when there are no additional codes.", "missing.codes")
        for name, codes in value["codes"].items():
            _scalars(codes, "missing.codes." + name)
    if answers.get("exclusions") is not None:
        value = answers["exclusions"]
        if not isinstance(value, dict):
            _fail("Declare no exclusions or the approved quality-control values to keep.", "exclusions")
        if value.get("action") == "none":
            _mapping(value, {"action"}, "exclusions", required={"action"})
        else:
            _mapping(value, {"action", "column", "values", "reason", "nulls"}, "exclusions", required={"action", "column", "values", "reason", "nulls"})
            _choice(value["action"], {"keep_values"}, "exclusions.action")
            _text(value["column"], "exclusions.column")
            _scalars(value["values"], "exclusions.values", nonempty=True)
            _text(value["reason"], "exclusions.reason")
            _choice(value["nulls"], {"error", "keep", "drop"}, "exclusions.nulls")
    if answers.get("matching") is not None:
        value = answers["matching"]
        if not isinstance(value, dict):
            _fail("Declare no attachment, or the exact metadata matching rules.", "matching")
        if value.get("action") == "none":
            _mapping(value, {"action"}, "matching", required={"action"})
        else:
            required = {"action", "source", "left_on", "right_on", "cardinality", "unmatched", "unused", "overlap", "nulls"}
            _mapping(value, required | {"suffix"}, "matching", required=required)
            _choice(value["action"], {"attach"}, "matching.action")
            _text(value["source"], "matching.source")
            for side in ("left_on", "right_on"):
                _names(value[side], "matching." + side)
            if len(value["left_on"]) != len(value["right_on"]):
                _fail("Use equally many observation and metadata matching columns.", "matching")
            _choice(value["cardinality"], {"1:1", "m:1"}, "matching.cardinality")
            _choice(value["unmatched"], {"error", "keep"}, "matching.unmatched")
            _choice(value["unused"], {"error", "drop"}, "matching.unused")
            _choice(value["overlap"], {"error", "suffix"}, "matching.overlap")
            _choice(value["nulls"], {"error"}, "matching.nulls")
            if "suffix" in value:
                _text(value["suffix"], "matching.suffix")
                if value["overlap"] != "suffix":
                    _fail("A suffix applies only when overlapping metadata columns are deliberately retained.", "matching.suffix")
    if answers.get("source_options") is not None:
        configurations = answers["source_options"]
        if not isinstance(configurations, dict) or any(not isinstance(name, str) or not name for name in configurations):
            _fail("Map named sources to their explicitly supplied file parsing declarations.", "source_options")
        for name, configuration in configurations.items():
            _mapping(configuration, {"format", "options", "schema"}, "source_options." + name)
    if answers.get("descriptions") is not None:
        value = answers["descriptions"]
        if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
            _fail("Map columns to supplied variable descriptions.", "descriptions")
        for name, meaning in value.items():
            _text(meaning, "descriptions." + name)


def validate_decisions(recipe: Mapping[str, Any]) -> None:
    """Validate recorded guided answers, including explicit decisions to do nothing."""
    decisions = recipe.get("research_decisions")
    if not isinstance(decisions, dict):
        raise WrangleError("INVALID_RECIPE", "research_decisions must contain the recorded research answers.")
    try:
        _validate_answers(decisions)
    except WrangleError as error:
        raise WrangleError("INVALID_RECIPE", str(error), error.details) from error
    required = {"input", "observation", "key", "measurements", "missing", "exclusions", "matching"}
    missing = sorted(name for name in required if decisions.get(name) is None)
    measurements = decisions.get("measurements") or {}
    incomplete_measurements = [name for name, value in measurements.items() if not _measurement_resolved(value)]
    if missing or incomplete_measurements:
        raise WrangleError("INVALID_RECIPE", "The recorded research decisions are incomplete; resolve them through wrangle start before preparation.", {"fields": missing, "measurements": incomplete_measurements})
    _validate_intent(recipe, decisions)


def _validate_intent(recipe, decisions):
    """Block a recipe that contradicts the research choices it claims to record."""
    def agree(condition, field):
        if not condition:
            raise WrangleError("INVALID_RECIPE", "The protocol disagrees with its recorded research decision. Update the answers and regenerate the protocol, or deliberately revise both declarations.", {"field": field})

    agree(recipe.get("input") == decisions["input"], "input")
    key = recipe.get("key")
    agree(([key] if isinstance(key, str) else key) == decisions["key"], "key")
    units = {name: value["unit"] for name, value in decisions["measurements"].items()}
    agree(recipe.get("units", {}) == units, "measurements.units")
    agree(recipe.get("descriptions", {}) == (decisions.get("descriptions") or {}), "descriptions")
    agree(recipe.get("source_options", {}) == (decisions.get("source_options") or {}), "source_options")
    steps = recipe.get("steps", [])
    agree(isinstance(steps, list) and all(isinstance(step, dict) for step in steps), "steps")
    def selected(operation):
        return [step for step in steps if step.get("op") == operation]
    casts = {name: value["dtype"] for name, value in decisions["measurements"].items() if value["dtype"] != "keep"}
    agree(selected("cast") == ([{"op": "cast", "columns": casts}] if casts else []), "measurements.dtype")
    codes = decisions["missing"]["codes"]
    normalization = [{"op": "normalize_missing", "columns": codes, "nan": False}] if codes else []
    agree(selected("normalize_missing") == normalization, "missing.codes")
    exclusions = decisions["exclusions"]
    filters = []
    if exclusions["action"] == "keep_values":
        predicates = [{"eq": [{"col": exclusions["column"]}, value]} for value in exclusions["values"]]
        filters = [{"op": "filter", "where": _or(predicates), "nulls": exclusions["nulls"], "reason": exclusions["reason"]}]
    agree(selected("filter") == filters, "exclusions")
    matching = decisions["matching"]
    joins, contract = [], None
    if matching["action"] == "attach":
        joins = [{"op": "join", "source": matching["source"], "left_on": matching["left_on"], "right_on": matching["right_on"], "how": "left", "cardinality": matching["cardinality"], "maintain_order": "left", "unmatched": {"left": matching["unmatched"], "right": matching["unused"]}, "nulls": "error", "overlap": matching["overlap"], "suffix": matching.get("suffix", "_metadata"), "coalesce": True}]
        contract = recipe.get("source_contracts", {}).get(matching["source"], {}).get("key")
        agree(contract == matching["right_on"], "matching.key")
    agree(selected("join") == joins, "matching")
    checks = recipe.get("checks", {})
    agree(isinstance(checks, dict) and isinstance(checks.get("protocol"), dict), "checks")
    agree(checks["protocol"].get("key") is True and checks["protocol"].get("units", []) == list(units), "checks.protocol")
    if decisions["missing"]["action"] == "error":
        agree(isinstance(checks.get("schema"), dict) and checks.get("missing") == {name: {"max": 0} for name in checks["schema"]}, "missing.action")
    else:
        agree(not checks.get("missing"), "missing.action")
    critical = [step.get("op") for step in steps if step.get("op") in {"normalize_missing", "cast", "join", "filter"}]
    expected = (["normalize_missing"] if codes else []) + (["cast"] if casts else []) + (["join"] if joins else []) + (["filter"] if filters else [])
    agree(critical == expected, "steps.order")


def _measurement_resolved(value):
    unit = value.get("unit")
    return value.get("dtype") in _NUMERIC_CHOICES and isinstance(unit, str) and bool(unit.strip()) and unit.strip().lower() not in _UNRESOLVED_UNITS


def _dtype_document(dtype):
    """Record observed native types without trying to infer their scientific role."""
    if isinstance(dtype, pl.Datetime):
        value = {"Datetime": {"time_unit": dtype.time_unit, "time_zone": dtype.time_zone}}
    elif isinstance(dtype, pl.Duration):
        value = {"Duration": dtype.time_unit}
    elif isinstance(dtype, pl.Decimal):
        value = {"Decimal": {"precision": dtype.precision or 38, "scale": dtype.scale}}
    elif isinstance(dtype, pl.List):
        value = {"List": _dtype_document(dtype.inner)}
    elif isinstance(dtype, pl.Array):
        value = {"Array": {"inner": _dtype_document(dtype.inner), "shape": list(dtype.shape)}}
    elif isinstance(dtype, pl.Struct):
        value = {"Struct": {field.name: _dtype_document(field.dtype) for field in dtype.fields}}
    elif isinstance(dtype, pl.Enum):
        value = {"Enum": dtype.categories.to_list()}
    else:
        value = str(dtype)
    try:
        resolved = dtype_spec(value)
    except WrangleError as error:
        raise WrangleError("START_SCOPE", "This observed native type requires an ordinary recipe; the guided path cannot describe it safely.", {"dtype": str(dtype)}) from error
    if resolved != dtype:
        raise WrangleError("START_SCOPE", "This observed native type requires an ordinary recipe.", {"dtype": str(dtype)})
    return value


def _source_bindings(sources):
    if not isinstance(sources, Mapping) or not sources or any(not isinstance(name, str) or not name or "=" in name for name in sources):
        raise WrangleError("INVALID_INPUT", "Supply named local files, for example measurements=instrument.csv.")
    if len(sources) > 2:
        raise WrangleError("START_SCOPE", "The guided path prepares one observation file and optionally attaches one metadata file; use an ordinary recipe for larger workflows.", {"sources": list(sources)})
    paths, declarations = {}, {}
    for name, value in sources.items():
        if isinstance(value, Mapping):
            value = dict(value)
            if set(value) - {"path", "format", "options", "schema"} or not isinstance(value.get("path"), (str, Path)):
                raise WrangleError("INVALID_SOURCE_OPTIONS", "A guided source declares a local path and documented parsing options only.", {"source": name})
            path = value["path"]
            configuration = normalize_recipe({key: item for key, item in value.items() if key != "path"})
        elif isinstance(value, (str, Path)):
            path, configuration = value, {}
        else:
            raise WrangleError("START_SCOPE", "The guided path starts from local data files or saved preparations; use inspect and prepare directly for in-memory tables.", {"source": name})
        paths[name] = str(Path(path).expanduser().resolve())
        declarations[name] = {"path": paths[name], **configuration}
    return paths, declarations


def _observed_names(names, available, field):
    absent = [name for name in names if name not in available]
    if absent:
        _fail("Choose columns present in the observed file; column names are never guessed.", field, columns=absent, available=list(available))


def _literal_values(values, dtype, field):
    from ._recipe_fields import _values
    try:
        _values(values, dtype, field)
    except WrangleError as error:
        _fail("Use exact values of the observed field type. CSV text codes must stay quoted text; no parsing or case folding is inferred.", field, dtype=str(dtype), cause=error.code)


def _or(predicates):
    if len(predicates) == 1:
        return predicates[0]
    split = len(predicates) // 2
    return {"or": [_or(predicates[:split]), _or(predicates[split:])]}


def _questions(answers, sources, input_name, schemas, output_schema=None):
    schema = schemas.get(input_name, {})
    result = []
    for definition in QUESTION_DEFINITIONS:
        field, prompt = definition["id"], definition["question"]
        choices = list(sources) if field == "input" else list(definition["choices"])
        if field == "matching" and len(sources) == 1:
            continue
        unresolved = answers.get(field) is None
        if field == "measurements" and answers.get(field) is not None:
            unresolved = any(not _measurement_resolved(value) for value in answers[field].values())
        if unresolved:
            question_schema = output_schema if field == "exclusions" and output_schema is not None else schema
            item = {"id": field, "field": field, "question": prompt, "prompt": prompt, "choices": choices, "columns": list(question_schema), "schema": {name: str(dtype) for name, dtype in question_schema.items()}}
            if field in {"input", "matching"}:
                item["sources"] = {name: {"columns": list(values), "schema": {column: str(dtype) for column, dtype in values.items()}} for name, values in schemas.items() if field == "input" or name != input_name}
            result.append(item)
    return result


def draft(sources: Mapping[str, Any], answers: Mapping[str, Any] | str | Path | None = None, *, input: str | None = None) -> dict[str, Any]:
    """Observe one file plus optional metadata and draft only supplied intent.

    ``ready`` means every required research decision has an answer. Data checks
    have not run; use the normal preparation engine to validate and execute.
    YAML answer files use the same strict scalar grammar as recipe files.
    """
    from ._api import _source
    from ._profile import profile
    paths, declarations = _source_bindings(sources)
    values = load_recipe(answers) if isinstance(answers, (str, Path)) else normalize_recipe({} if answers is None else answers)
    _validate_answers(values)
    for name, configuration in (values.get("source_options") or {}).items():
        if name not in declarations:
            _fail("Parsing declarations must name supplied source files.", "source_options", source=name)
        supplied = {key: item for key, item in declarations[name].items() if key != "path"}
        conflicts = [key for key, item in configuration.items() if key in supplied and supplied[key] != item]
        if conflicts:
            _fail("Previously recorded parsing declarations and supplied source options disagree; choose the intended declaration explicitly.", "source_options." + name, options=conflicts)
        declarations[name].update(configuration)
    configurations = {name: {key: item for key, item in declaration.items() if key != "path"} for name, declaration in declarations.items() if len(declaration) > 1}
    if configurations:
        values["source_options"] = configurations
    if input is not None:
        if values.get("input") is not None and values["input"] != input:
            _fail("The selected observation file and recorded input answer must agree.", "input")
        values["input"] = input
    elif values.get("input") is None and len(paths) == 1:
        # There is no selection to make when only one file was supplied.
        values["input"] = next(iter(paths))
    input_name = values.get("input")
    if input_name is not None and input_name not in paths:
        _fail("Choose one of the supplied source names as the observation file.", "input", sources=list(paths))
    schemas, observations = {}, {}
    for name, declaration in declarations.items():
        data, identity = _source(declaration)
        schemas[name] = data.schema
        observations[name] = profile(data, identity, sample_rows=5)
    main_schema = schemas.get(input_name, {})
    if len(paths) == 1 and values.get("matching") is None:
        values["matching"] = {"action": "none"}
    if input_name is None:
        dependent = [name for name in ("key", "measurements", "missing", "exclusions", "matching", "descriptions") if values.get(name) is not None]
        if dependent:
            _fail("Select the observation file before answering its column questions.", "input", dependent_answers=dependent)
    for field in ("key", "measurements", "descriptions"):
        if values.get(field) is not None:
            _observed_names(values[field], main_schema, field)
    measurements = values.get("measurements") or {}
    casts, units = {}, {}
    output_schema = dict(main_schema)
    for name, measurement in measurements.items():
        representation = measurement.get("dtype")
        if representation == "keep" and not main_schema[name].is_numeric():
            _fail("A measurement must have a numeric native representation. Choose decimal numbers (Float64) or whole numbers (Int64) for CSV text.", "measurements." + name + ".dtype", observed=str(main_schema[name]))
        if representation in {"Float64", "Int64"}:
            casts[name] = representation
            output_schema[name] = dtype_spec(representation)
        if _measurement_resolved(measurement):
            units[name] = measurement["unit"]
    missing = values.get("missing")
    if missing is not None:
        _observed_names(missing["codes"], main_schema, "missing.codes")
        for name, codes in missing["codes"].items():
            _literal_values(codes, main_schema[name], "missing.codes." + name)
    matching = values.get("matching")
    join = None
    contracts = {}
    if matching is not None and matching["action"] == "attach":
        source = matching["source"]
        if source not in paths or source == input_name:
            _fail("Attach the supplied metadata file, not the observation file itself.", "matching.source", sources=list(paths))
        _observed_names(matching["left_on"], main_schema, "matching.left_on")
        _observed_names(matching["right_on"], schemas[source], "matching.right_on")
        pairs = list(zip(matching["left_on"], matching["right_on"]))
        incompatible = [[left, right] for left, right in pairs if output_schema[left] != schemas[source][right]]
        if incompatible:
            _fail("Matching columns must have the same native type. Use an ordinary recipe for explicit metadata typing before matching.", "matching", columns=incompatible)
        join = {"op": "join", "source": source, "left_on": matching["left_on"], "right_on": matching["right_on"], "how": "left", "cardinality": matching["cardinality"], "maintain_order": "left", "unmatched": {"left": matching["unmatched"], "right": matching["unused"]}, "nulls": "error", "overlap": matching["overlap"], "suffix": matching.get("suffix", "_metadata"), "coalesce": True}
        from ._recipe_join import output_mapping
        try:
            mapping = output_mapping(list(output_schema), list(schemas[source]), {name: value for name, value in join.items() if name not in {"op", "source"}})
        except WrangleError as error:
            _fail("Metadata names overlap or the suffix collides. Declare deliberate suffixing or use an ordinary recipe to rename fields.", "matching.overlap", cause=error.code, **error.details)
        for name, destination in mapping.items():
            if destination not in output_schema:
                output_schema[destination] = schemas[source][name]
        contracts[source] = {"key": matching["right_on"]}
    exclusions = values.get("exclusions")
    if exclusions is not None and exclusions["action"] == "keep_values":
        _observed_names([exclusions["column"]], output_schema, "exclusions.column")
        _literal_values(exclusions["values"], output_schema[exclusions["column"]], "exclusions.values")
    questions = _questions(values, paths, input_name, schemas, output_schema)
    recipe = {"version": 1, "name": "Study preparation protocol", "pending_decisions": [{"id": item["id"], "question": item["question"]} for item in questions], "research_decisions": values, "steps": []}
    if input_name is not None:
        recipe["input"] = input_name
    if values.get("key") is not None:
        recipe["key"] = values["key"]
    if units:
        recipe["units"] = units
    if values.get("descriptions"):
        recipe["descriptions"] = values["descriptions"]
    configurations = {name: {key: value for key, value in declaration.items() if key != "path"} for name, declaration in declarations.items() if len(declaration) > 1}
    if configurations:
        recipe["source_options"] = configurations
    if contracts:
        recipe["source_contracts"] = contracts
    steps = recipe["steps"]
    if missing is not None and missing["codes"]:
        steps.append({"op": "normalize_missing", "columns": missing["codes"], "nan": False})
    if casts:
        steps.append({"op": "cast", "columns": casts})
    if join is not None:
        steps.append(join)
    if exclusions is not None and exclusions["action"] == "keep_values":
        predicates = [{"eq": [{"col": exclusions["column"]}, value]} for value in exclusions["values"]]
        steps.append({"op": "filter", "where": _or(predicates), "nulls": exclusions["nulls"], "reason": exclusions["reason"]})
    recipe["checks"] = {"schema": {name: _dtype_document(dtype) for name, dtype in output_schema.items()}, "extra_columns": "error", "protocol": {"key": True, "units": list(units)}}
    if missing is not None and missing["action"] == "error":
        recipe["checks"]["missing"] = {name: {"max": 0} for name in output_schema}
    if not questions:
        validate_decisions(recipe)
    return {"recipe": recipe, "questions": questions, "unresolved": [item["id"] for item in questions], "observations": observations, "output_schema": {name: str(dtype) for name, dtype in output_schema.items()}, "sources": paths, "answers": values, "ready": not questions, "status": "decisions_complete" if not questions else "decisions_required", "data_validated": False}


def _commented_recipe(recipe):
    comments = {
        "pending_decisions:": "# Preparation is blocked while any required research decision is unresolved.",
        "research_decisions:": "# Supplied scientific answers, including decisions to keep missing values or all rows.",
        "key:": "# These columns identify one observation; missing or repeated keys cause an error.",
        "units:": "# Physical units supplied by the researcher; 1 means dimensionless.",
        "steps:": "# Mechanical operations compiled from the supplied answers. No scientific choices are inferred.",
        "checks:": "# Preparation checks the declared key, units, and observed output schema on every batch.",
    }
    lines = ["# Generated study protocol: intent only; data have not been validated.", "# Resolve pending decisions in answers.yaml and rerun wrangle start, then prepare."]
    for line in dump_recipe(recipe).splitlines():
        name = line.split(" ", 1)[0]
        if not line.startswith(" ") and name in comments:
            lines.append(comments[name])
        lines.append(line)
    return "\n".join(lines) + "\n"


def _validate_proposal(proposal):
    """Keep publication status and pending questions tied to recorded answers."""
    fields = {"recipe", "answers", "sources", "observations"}
    if any(not isinstance(proposal.get(name), dict) for name in fields):
        raise WrangleError("INVALID_RECIPE", "Publish the complete proposal returned by wrangle start.")
    recipe, answers, sources = proposal["recipe"], proposal["answers"], proposal["sources"]
    try:
        _validate_answers(answers)
    except WrangleError as error:
        raise WrangleError("INVALID_RECIPE", str(error), error.details) from error
    if not sources or any(not isinstance(name, str) or not name or not isinstance(path, str) or not path for name, path in sources.items()):
        raise WrangleError("INVALID_RECIPE", "A proposal must retain its named source paths.")
    schemas = {}
    for name in sources:
        observation = proposal["observations"].get(name)
        if not isinstance(observation, dict) or not isinstance(observation.get("columns"), dict):
            raise WrangleError("INVALID_RECIPE", "A proposal must retain the observed source schemas.", {"source": name})
        schemas[name] = observation["columns"]
    checks = recipe.get("checks")
    if not isinstance(checks, dict) or not isinstance(checks.get("schema"), dict):
        raise WrangleError("INVALID_RECIPE", "A proposal must retain its observed output schema.")
    try:
        output_schema = {name: str(dtype_spec(dtype)) for name, dtype in checks["schema"].items()}
    except WrangleError as error:
        raise WrangleError("INVALID_RECIPE", "A proposal's output schema must use documented native type declarations.", error.details) from error
    if proposal.get("output_schema") != output_schema:
        raise WrangleError("INVALID_RECIPE", "The proposal's output schema disagrees with its protocol. Regenerate the draft.")
    questions = _questions(answers, sources, answers.get("input"), schemas, output_schema)
    pending = [{"id": item["id"], "question": item["question"]} for item in questions]
    unresolved = [item["id"] for item in questions]
    ready = not questions
    expected_status = "decisions_complete" if ready else "decisions_required"
    if (proposal.get("questions") != questions or proposal.get("unresolved") != unresolved
            or proposal.get("ready") is not ready or proposal.get("status") != expected_status
            or proposal.get("data_validated") is not False
            or recipe.get("pending_decisions") != pending
            or recipe.get("research_decisions") != answers):
        raise WrangleError("INVALID_RECIPE", "The proposal's status or questions disagree with its recorded answers. Regenerate it through wrangle start; drafting never validates data.")
    if ready:
        validate_decisions(recipe)


def publish(proposal: Mapping[str, Any], directory: str | Path) -> dict[str, Any]:
    """Atomically save an editable draft to a new directory, without copying data."""
    value = normalize_recipe(proposal)
    _validate_proposal(value)
    destination = Path(directory).expanduser().resolve()
    if destination.exists():
        raise WrangleError("OUTPUT_EXISTS", "Choose a new protocol directory; existing files are never overwritten.", {"path": str(destination)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / (destination.name + ".wrangle-lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise WrangleError("OUTPUT_BUSY", "Another process is publishing this destination.", {"path": str(destination)}) from error
    os.close(descriptor)
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=".wrangle-start-", dir=destination.parent))
        (temporary / "recipe.yaml").write_text(_commented_recipe(value["recipe"]), encoding="utf-8")
        answers = dict(value["answers"])
        for item in value.get("questions", []):
            answers.setdefault(item["field"], None)
        text = "# Answer in plain study terms; null marks an unanswered decision.\n# Use the README and question list; no operation catalog is needed.\n" + dump_recipe(answers)
        (temporary / "answers.yaml").write_text(text, encoding="utf-8")
        bindings = " ".join("--source " + shlex.quote(name + "=" + path) for name, path in value["sources"].items())
        recipe_path = destination / "recipe.yaml"
        answers_path = destination / "answers.yaml"
        prepare_command = "wrangle prepare " + shlex.quote(str(recipe_path)) + " " + bindings + " --output " + shlex.quote(str(destination / "prepared"))
        start_command = "wrangle start " + bindings + " --answers " + shlex.quote(str(answers_path)) + " --output " + shlex.quote(str(destination.parent / (destination.name + "-resolved")))
        pending = "\n".join("- " + item["question"] for item in value.get("questions", []))
        readme = "# Your study protocol\n\nThis draft records intent. Preparation has not run or validated these data.\n\n"
        if pending:
            readme += "Required research decisions:\n\n" + pending + "\n\nEdit answers.yaml, then save a new resolved draft:\n\n```sh\n" + start_command + "\n```\n\n"
        else:
            readme += "All required research decisions have answers. Prepare and check the files:\n\n"
        readme += "```sh\n" + prepare_command + "\n```\n\nPreparation publishes data.parquet, recipe.yaml, report.txt, and receipt.json only when the declared checks pass. The input files remain unchanged. For the next batch, use this recipe with the new file paths.\n"
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        publish_directory(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)
    value.update(path=str(destination), directory=str(destination), recipe_path=str(recipe_path), answers_path=str(answers_path), readme_path=str(destination / "README.md"), files=["README.md", "answers.yaml", "recipe.yaml"], next_commands=[start_command if value.get("unresolved") else prepare_command])
    return value
