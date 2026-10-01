"""Readable views of engine observations and receipts; no scientific decisions."""
from __future__ import annotations

import json
from pathlib import Path


def _value(value):
    if value is None:
        return "missing"
    return _terminal_lines(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _terminal_control(character):
    return (ord(character) < 32 or 127 <= ord(character) <= 159
            or character in "\u2028\u2029\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _name(value):
    text = str(value)
    return json.dumps(text, ensure_ascii=True) if any(_terminal_control(character) for character in text) else text


def _terminal_lines(text, *, preserve_newlines=True):
    return "".join(json.dumps(character, ensure_ascii=True)[1:-1]
                   if _terminal_control(character) and (character != "\n" or not preserve_newlines) else character
                   for character in text)


def _table(headers, rows):
    rows = [[_name(value) for value in row] for row in rows]
    widths = [max([len(header), *(len(row[index]) for row in rows)]) for index, header in enumerate(headers)]
    return ["  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip() for row in [headers, *rows]]


def render_inspect(profile):
    """Show observed values and concrete researcher decisions without inferring them."""
    fields = profile.get("fields", [])
    lines = ["Dataset inspection", f"Source: {_name(profile.get('path', 'Python data'))}", f"Rows: {profile['rows']:,} | Columns: {len(profile.get('columns', {})):,}", "", "Observed columns (no scientific meaning inferred):"]
    if fields:
        lines.extend(_table(["Column", "Type", "Null", "NaN", "Infinity", "Distinct"], [[field["name"], field["dtype"], field["nulls"], field["nans"], field["infinities"], field["distinct"]] for field in fields]))
    else:
        lines.append("No columns selected.")
    examples = profile.get("examples", [])
    if examples:
        names = list(examples[0])
        lines.extend(["", f"Example rows ({len(examples)}; strings are quoted):"])
        lines.extend(_table([_name(name) for name in names], [[_value(row.get(name)) for name in names] for row in examples]))
        encoding = profile.get("example_encoding", {})
        if encoding.get("binary_columns"):
            lines.append("Binary examples use hexadecimal: " + ", ".join(_name(name) for name in encoding["binary_columns"]))
        if any(field.get("nans") or field.get("infinities") for field in fields):
            lines.append("Nonfinite values display as missing in examples; NaN/infinity counts above remain separate.")
    summaries = [field for field in fields if "summary" in field]
    if summaries:
        lines.extend(["", "Numeric summaries (non-null observations; sample standard deviation, ddof=1):"])
        for field in summaries:
            summary = field["summary"]
            if summary["status"] == "unresolved":
                lines.append(f"  {_name(field['name'])}: unresolved ({summary['reason']}; observed non-null values: {summary['count']}).")
            else:
                values = ", ".join(f"{label}={_value(summary[label])}" for label in ("min", "max", "mean", "median", "std"))
                lines.append(f"  {_name(field['name'])}: n={summary['count']}, {values}")
    groups = profile.get("groups")
    if groups is not None:
        lines.extend(["", f"Groups by {', '.join(_name(name) for name in groups['by'])}: {groups['count']:,}"])
        if groups.get("status") == "unresolved":
            lines.append(f"Group summaries unresolved: {groups['reason']}.")
        else:
            for item in groups["items"]:
                lines.append(f"  {_value(item['values'])}: {item['rows']:,} rows")
                for field in item.get("fields", []):
                    summary = field.get("summary", {})
                    if summary.get("status") == "resolved":
                        lines.append(f"    {_name(field['name'])}: n={summary['count']}, mean={_value(summary['mean'])}, std={_value(summary['std'])}")
                    elif field["dtype"] not in {"String", "Boolean"}:
                        lines.append(f"    {_name(field['name'])}: unresolved ({summary.get('reason', 'UNKNOWN')})")
            if groups.get("truncated"):
                lines.append(f"Showing the first {len(groups['items'])} observed groups; use --max-groups to show more (maximum 100).")
    drift = profile.get("schema_changes")
    if drift is not None:
        lines.extend(["", "Schema compared with baseline:"])
        if not any(drift.values()):
            lines.append("  No columns added, removed, or changed in type.")
        for field in drift.get("added", []):
            lines.append(f"  Added: {_name(field['name'])} ({field['dtype']})")
        for field in drift.get("removed", []):
            lines.append(f"  Removed: {_name(field['name'])} ({field['dtype']})")
        for field in drift.get("type_changed", []):
            lines.append(f"  Type changed: {_name(field['name'])}: {field['before']} -> {field['after']}")
        lines.append("  Confirm changes against the study protocol before preparing the next batch.")
    lines.extend(["", "Researcher decisions for the YAML recipe:", "  Identify what one row represents and the column(s) that uniquely identify it."])
    strings = [field["name"] for field in fields if field["dtype"] == "String"]
    if strings:
        lines.append("  Text columns: " + ", ".join(_name(name) for name in strings) + ". Keep identifiers as text; explicitly cast measurements.")
    missing = [field["name"] for field in fields if field.get("nulls") or field.get("nans") or field.get("infinities")]
    if missing:
        lines.append("  Missing/nonfinite values: " + ", ".join(_name(name) for name in missing) + ". Declare what they mean and how to handle them.")
    lines.extend(["  Declare measurement units and variable meanings; confirm missing-value codes and any exclusion rule.", "  For linked tables, declare matching columns and permitted join cardinality.", "Next: record these decisions in a YAML recipe; use --source NAME=PATH with the source names declared by that recipe.", "Worked example: wrangle example NEW_DIRECTORY"])
    return "\n".join(lines)


def render_prepare(receipt, output=None):
    """Summarize verified execution while keeping undeclared meaning visible."""
    recipe = receipt.get("recipe", {})
    source = receipt.get("sources", {}).get(receipt.get("input"), {})
    result = receipt["output"]
    steps = receipt.get("steps", [])
    initial_rows = source.get("rows", steps[0]["rows_before"] if steps else result["rows"])
    initial_columns = source.get("columns", result["columns"])
    title = "Preparation complete" + (f": {_name(recipe['name'])}" if recipe.get("name") else "")
    lines = [title, f"Rows: {initial_rows:,} -> {result['rows']:,} | Columns: {len(initial_columns):,} -> {len(result['columns']):,}", f"Execution: {receipt.get('execution', {}).get('storage', 'memory')}"]
    observation = recipe.get("research_decisions", {}).get("observation")
    if observation:
        lines.append("One row represents: " + _name(observation))
    key = receipt.get("key", [])
    lines.append("Observation key: " + (", ".join(_name(name) for name in key) if key else "not declared"))
    if steps:
        lines.extend(["", "Changes:"])
        for step in steps:
            lines.append(f"  {step['step'] + 1}. {step['op']}: {step['rows_before']:,} -> {step['rows_after']:,} rows")
            parameters = step.get("parameters", {})
            if step["op"] == "convert_unit":
                lines.append(f"     Unit conversion: {_name(parameters['column'])}: {_name(parameters['from_unit'])} -> {_name(parameters['to_unit'])}; factor={_value(parameters['factor'])}, offset={_value(parameters['offset'])}")
                lines.append("     Formula: output value = input value * factor + offset")
            counts = step.get("observation_counts", {})
            for name, label in (("excluded_observations", "Excluded observations"), ("excluded_cells", "Excluded cells"), ("expanded_parents", "Observations expanded"), ("introduced_observations", "Observations introduced")):
                if counts.get(name):
                    lines.append(f"     {label}: {counts[name]:,}")
            if step.get("reason"):
                lines.append(f"     Recorded reason: {_name(step['reason'])}")
            excluded = step.get("excluded_keys")
            if isinstance(excluded, list) and excluded:
                preview = excluded[:5]
                lines.append(f"     Excluded identifiers: {_value(preview)}" + (f" (first 5 of {len(excluded):,}; all are in receipt.json)" if len(excluded) > 5 else ""))
            elif isinstance(excluded, dict):
                lines.append(f"     Excluded identifiers: {_name(excluded.get('path', 'evidence file'))}")
            if step.get("replacements"):
                lines.append("     Missing values normalized: " + _value(step["replacements"]))
            for name, label in (("columns_added", "Added columns"), ("columns_removed", "Removed columns")):
                if step.get(name):
                    lines.append(f"     {label}: " + ", ".join(_name(field) for field in step[name]))
    lines.extend(["", "Final columns and declared units:"])
    units = receipt.get("units", {})
    unit_labels = {}
    for name in result["columns"]:
        unit = units.get(name)
        unit_labels[name] = "row-specific (column " + _name(unit[1:]) + ")" if isinstance(unit, str) and unit.startswith("@") else _name(unit) if unit is not None else "not declared"
    lines.extend(_table(["Column", "Type", "Declared unit"], [[name, dtype, unit_labels[name]] for name, dtype in result["columns"].items()]))
    checks = receipt.get("checks", [])
    passed = sum(check.get("passed") is True for check in checks)
    lines.extend(["", f"Final checks recorded: {passed:,} passed of {len(checks):,}."])
    if not recipe.get("checks"):
        lines.append("No additional research checks were declared; built-in safety and declared identity checks still apply.")
    unresolved_units = receipt.get("units_unresolved", [])
    unresolved_descriptions = receipt.get("descriptions_unresolved", [])
    lines.append("Measurement units not declared: " + (", ".join(_name(name) for name in unresolved_units) if unresolved_units else "none among numeric non-key columns"))
    lines.append("Variable meanings not declared: " + (", ".join(_name(name) for name in unresolved_descriptions) if unresolved_descriptions else "none"))
    if unresolved_units or unresolved_descriptions or not key:
        lines.append("Confirm these declarations against the study protocol before analysis; preparation does not infer them.")
    if output is not None:
        destination = Path(output).expanduser().resolve()
        lines.extend(["", f"Saved: {_name(destination)}", "  data.parquet: prepared table", "  recipe.yaml: reusable protocol", "  report.txt: this readable summary", "  receipt.json: complete checks, changes, parameters, and hashes"])
        if receipt.get("evidence_files"):
            lines.append("  evidence/: complete observation transitions and exclusions")
        lines.append("Review report.txt and the recorded exclusion reasons; retain the whole directory with your research.")
    else:
        lines.extend(["", "The receipt records input/output hashes, resolved parameters, declared checks, and observation changes."])
    return "\n".join(lines)


def render_catalog(document):
    """Show task navigation, with exact engine arguments when one operation is selected."""
    entries = document["operations"]
    if len(entries) != 1:
        from ._api import _SIMPLE
        names = set(_SIMPLE) | {"join"}
        lines = ["Recipe operations", "Choose the operation that matches the task; its contract lists required decisions."]
        for entry in entries:
            if entry["name"] in names:
                description = entry["description"].split("\n", 1)[0]
                lines.append(f"  {entry['name']}: {description}")
        lines.extend(["", "Read a contract: wrangle catalog NAME", "Full machine catalog, including Python helpers: wrangle catalog --json", "Start with a working dataset and YAML recipe: wrangle example NEW_DIRECTORY"])
        return "\n".join(lines)
    entry = entries[0]
    lines = [f"Operation: {entry['name']}", entry["description"], f"Recipe eligibility: {entry['recipe']['eligibility']}", "", "Arguments:"]
    for argument in entry["arguments"][1:]:
        condition = "required" if argument["required"] else "default=" + _value(argument.get("default"))
        lines.append(f"  {argument['name']}: {condition}")
    for choice in entry.get("argument_constraints", []):
        lines.append(f"  {choice['argument']}: permitted values {_value(choice['allowed'])}")
    if entry.get("validation"):
        lines.extend(["", "Preconditions and failure codes:"])
        lines.extend(f"  {item['code']}: {item['precondition']}" for item in entry["validation"])
    lines.extend(["", f"Agent manual: {entry['manual']}", f"Python symbol: {entry['symbol']}", f"Implementation: {entry['source']}:{entry['line']}", "Exact structured contract: wrangle catalog " + entry["name"] + " --json"])
    return "\n".join(lines)


def render_start(proposal):
    """Explain an authored draft without claiming execution or data validation."""
    ready = proposal.get("ready", False)
    lines = ["Protocol ready to prepare" if ready else "Protocol draft: researcher decisions still needed", "No dataset has been prepared or validated by this setup."]
    for name, profile in proposal.get("observations", {}).items():
        lines.append(f"Source {_name(name)}: {profile['rows']:,} rows | {len(profile['columns']):,} columns")
        lines.append("  Columns: " + ", ".join(_name(column) for column in profile["columns"]))
    unresolved = proposal.get("unresolved", [])
    if unresolved:
        lines.extend(["", "Decisions still needed:"])
        questions = {question["id"]: question for question in proposal.get("questions", [])}
        for item in unresolved:
            identifier = item["id"] if isinstance(item, dict) else item
            prompt = questions.get(identifier, {}).get("prompt", item.get("question", identifier) if isinstance(item, dict) else identifier)
            lines.append("  " + _name(identifier) + ": " + _name(prompt))
        lines.append("Preparation is blocked until these decisions are answered. Ask the researcher; keep uncertain answers unresolved.")
    if proposal.get("directory"):
        lines.extend(["", "Saved: " + _name(proposal["directory"]), "  recipe.yaml: protocol, with unresolved decisions explicit", "  answers.yaml: editable record of researcher decisions", "  README.md: instructions using your source files"])
        if ready:
            lines.append("Review the protocol, then run the preparation command in README.md. Preparation will validate the files and record evidence.")
        else:
            lines.append("Complete answers.yaml, then rerun the start command in README.md into a new directory.")
        commands = proposal.get("next_commands", [])
        if commands:
            controls = any(_terminal_control(character) for command in commands for character in command)
            label = "Next command (control characters escaped for display):" if controls else "Next command:"
            lines.extend(["", label])
            lines.extend("  " + _terminal_lines(command, preserve_newlines=False) for command in commands)
    else:
        from ._protocol import dump_recipe
        protocol = dump_recipe(proposal["recipe"]).rstrip()
        protocol = _terminal_lines(protocol)
        lines.extend(["", "Draft protocol:", protocol, "", "Add --output NEW_DIRECTORY to save this protocol, answers, and exact next commands."])
        if unresolved:
            lines.append("In a terminal, start asks plain-language questions. For scripts or agents, use --answers ANSWERS.yaml and --json.")
    return "\n".join(lines)


_RECOVERY = {
    "INVALID_INVOCATION": "Run wrangle --help or wrangle COMMAND --help for accepted arguments.",
    "OUTPUT_EXISTS": "Choose a new output directory; retain the existing result.",
    "OUTPUT_BUSY": "Wait for the other writer to finish, then choose an unused output directory.",
    "SOURCE_NOT_FOUND": "Check the file path and source binding; use an existing local file or verified result directory.",
    "DUPLICATE_KEY": "Review duplicate observations and the declared identity; decide whether the data or protocol needs correction.",
    "JOIN_CARDINALITY": "Review matching keys and expected relationships; correct data or declare the researcher-approved cardinality.",
    "UNDECLARED_LOSS": "Review affected observations; record the researcher-approved exclusion rule and reason in the recipe.",
    "UNDECLARED_EXPANSION": "Review why observations multiply; explicitly authorize expansion only if the study protocol requires it.",
    "UNDECLARED_UNITS": "Declare the scientifically correct output units or use an explicit unit conversion.",
    "UNIT_MISMATCH": "Verify physical units against the study protocol; convert measurements explicitly before combining them.",
    "MISSING_REQUIRED": "Review missing observations and the research requirement; decide how they must be handled.",
    "NONFINITE_RESULT": "Review NaN/infinity values and explicitly declare scientifically appropriate handling.",
    "CHECK_FAILED": "Compare the reported observations with the study protocol; correct data or the declared requirement.",
    "UNKNOWN_OPERATION": "Run wrangle catalog, then choose the exact documented operation name.",
    "START_CANCELLED": "Rerun wrangle start when you are ready; provide --answers to retain decisions in a YAML file.",
    "UNRESOLVED_PROTOCOL": "Ask the researcher for the listed decisions, complete answers.yaml, and rerun wrangle start; never remove the pending decisions to bypass review.",
    "UNRESOLVED_DECISIONS": "Ask the researcher for the listed decisions, complete answers.yaml, and rerun wrangle start; never remove the pending decisions to bypass review.",
    "START_ANSWER": "Review the stated answer field and permitted choices; record only researcher-confirmed decisions, then rerun wrangle start.",
    "START_SCOPE": "Use an ordinary YAML recipe for the stated unsupported workflow; read wrangle catalog for operation requirements and retain unresolved study decisions.",
    "START_ANSWERS": "Review the stated answer field and permitted choices; record only researcher-confirmed decisions, then rerun wrangle start.",
    "IO_ERROR": "Check the stated path, directory permissions, and available disk space; rerun after resolving the cause.",
}


def render_error(error):
    """Keep stable codes/context visible and offer recovery without changing a method."""
    details = error.details
    lines = [f"{error.code}: {_name(str(error))}"]
    if "step" in details:
        step = details["step"]
        label = step + 1 if type(step) is int else step
        lines.append(f"Step: {label}" + (f" ({_name(details['operation'])})" if details.get("operation") else ""))
    elif details.get("operation"):
        lines.append(f"Operation: {_name(details['operation'])}")
    if details.get("path"):
        location = _name(details["path"])
        if "line" in details:
            location += f":{details['line']}"
            if "column" in details:
                location += f":{details['column']}"
        lines.append("Location: " + location)
    context = {"step", "operation", "path", "line"}
    if "line" in details:
        context.add("column")
    remaining = {name: value for name, value in details.items() if name not in context}
    if remaining:
        lines.append("Details: " + _value(remaining))
    yaml_error = "YAML" in error.code or error.code == "INVALID_RECIPE" and ("line" in details or str(details.get("path", "")).lower().endswith((".yaml", ".yml")))
    action = "Check YAML indentation and the indicated line; quote string identifiers, dates, and text values when needed. Resolve duplicate fields or unsupported tags before rerunning." if yaml_error else _RECOVERY.get(error.code)
    if action is None:
        action = "Review the stated requirement against the research protocol; correct the declared precondition and rerun."
    lines.append("Next: " + action)
    if details.get("operation"):
        lines.append("Contract: wrangle catalog " + _name(details["operation"]))
    return "\n".join(lines)
