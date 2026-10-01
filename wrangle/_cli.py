"""Readable command-line adapter; explicit JSON output uses the same engine."""
from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

from . import __version__, inspect, prepare
from ._catalog import catalog, operation_document
from ._core import WrangleError
from ._presentation import render_catalog, render_error, render_inspect, render_prepare, render_start
from ._publication import publish_directory

__all__ = ["main"]


class _InvocationError(WrangleError):
    def __init__(self, message, details=None):
        super().__init__("INVALID_INVOCATION", message, details)


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "allow_abbrev": False})

    def error(self, message):
        raise _InvocationError(message)


def _json_option(parser, *, root=False):
    parser.add_argument("--json", action="store_true", default=False if root else argparse.SUPPRESS, help="Write structured JSON results and errors for scripts or agents.")


def _parser():
    parser = _Parser(prog="wrangle", description="Inspect research files, edit a YAML protocol, and prepare a checked dataset.")
    parser.add_argument("--version", action="version", version=f"wrangle {__version__}")
    _json_option(parser, root=True)
    commands = parser.add_subparsers(dest="command", required=True)
    starter = commands.add_parser("example", help="Copy a complete research example and editable YAML recipe to a new directory.")
    starter.add_argument("directory", help="New directory for the example; existing directories are never overwritten.")
    _json_option(starter)
    begin = commands.add_parser("start", help="Create a protocol for your own files, with explicit researcher decisions.")
    begin.add_argument("source_file", nargs="?", metavar="INPUT_FILE", help="Main local dataset; named measurements in the generated protocol.")
    begin.add_argument("--metadata", metavar="PATH", help="Optional file describing samples or observations.")
    begin.add_argument("--sheet", metavar="NAME", help="Explicit worksheet name for an Excel INPUT_FILE.")
    begin.add_argument("--metadata-sheet", metavar="NAME", help="Explicit worksheet name for an Excel --metadata file.")
    begin.add_argument("--source", action="append", metavar="NAME=PATH", help="Alternatively provide named sources; repeat for main data and metadata.")
    begin.add_argument("--input", metavar="NAME", help="Choose the main table from named sources.")
    begin.add_argument("--answers", metavar="ANSWERS.yaml", help="Previously recorded researcher decisions in a YAML file.")
    begin.add_argument("--output", metavar="NEW_DIRECTORY", help="Save recipe.yaml, answers.yaml, and instructions to a new directory.")
    begin.add_argument("--interactive", action="store_true", help="Ask questions even when input is redirected; cannot be combined with --json.")
    _json_option(begin)
    observe = commands.add_parser("inspect", help="Read column types, missing values, examples, and researcher decisions.")
    observe.add_argument("source", help="Local data file or verified preparation bundle.")
    observe.add_argument("--sample-rows", type=int, default=5, help="Example rows, from 0 to 100 (default: 5).")
    observe.add_argument("--column", action="append", dest="columns", metavar="COLUMN", help="Observe selected fields and examples; repeat for multiple columns.")
    observe.add_argument("--summary", action="store_true", help="Observe numeric summaries with explicit null and precision boundaries.")
    observe.add_argument("--group", action="append", dest="groups", metavar="COLUMN", help="Summarize observed groups; repeat for a composite grouping.")
    observe.add_argument("--max-groups", type=int, default=20, help="Maximum reported groups, from 1 to 100 (default: 20).")
    observe.add_argument("--baseline", metavar="PATH", help="Compare observed schema with a local source or verified bundle.")
    _json_option(observe)
    discover = commands.add_parser("catalog", help="Find operations or read one operation's arguments and requirements.")
    discover.add_argument("operation", nargs="?", help="Optional exact operation name.")
    _json_option(discover)
    execute = commands.add_parser("prepare", help="Run the YAML recipe and scientific checks; explain changes and evidence.")
    from ._api import _SIMPLE
    execute.epilog = "Recipe operations: " + ", ".join(sorted(_SIMPLE)) + ". Read wrangle catalog NAME for the exact operation contract."
    execute.add_argument("recipe", help="YAML recipe file (.yaml or .yml); scientific decisions stay in this protocol.")
    execute.add_argument("--source", action="append", required=True, metavar="NAME=PATH", help="Named local source; repeat for multiple sources.")
    execute.add_argument("--execution", choices=("memory", "disk"), default="memory", help="Disk checkpoints and streaming execution require --output; global operations can still need working memory.")
    execute.add_argument("--output", help="Publish the table, YAML recipe, readable report, and receipt to a new directory.")
    _json_option(execute)
    return parser


def _sources(bindings):
    sources = {}
    for binding in bindings:
        name, separator, path = binding.partition("=")
        if not separator or not name or not path:
            raise _InvocationError("Use --source NAME=PATH with a nonempty name and path.", {"source": binding})
        if name in sources:
            raise _InvocationError("Source names must be distinct.", {"source": name})
        sources[name] = path
    return sources


def _example(directory):
    source = Path(__file__).parent / "docs" / "starter"
    if not source.is_dir():
        raise WrangleError("EXAMPLE_UNAVAILABLE", "The installed package is missing its starter example; reinstall the complete package.")
    destination = Path(directory).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / (destination.name + ".wrangle-lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise WrangleError("OUTPUT_BUSY", "Another process is publishing this destination.", {"path": str(destination)}) from error
    os.close(descriptor)
    temporary = None
    try:
        if destination.exists():
            raise WrangleError("OUTPUT_EXISTS", "Choose a new example directory; existing files are never overwritten.", {"path": str(destination)})
        temporary = Path(tempfile.mkdtemp(prefix=".wrangle-example-", dir=destination.parent))
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        files = sorted(str(path.relative_to(temporary)) for path in temporary.rglob("*") if path.is_file())
        publish_directory(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
        lock.unlink(missing_ok=True)
    return {"path": str(destination), "files": files, "next_commands": [
        "cd " + shlex.quote(str(destination)),
        "wrangle inspect samples.csv",
        "wrangle prepare recipe.yaml --source measurements=samples.csv --source metadata=metadata.csv --output prepared",
        "wrangle inspect prepared",
    ]}



def _start_sources(arguments):
    if arguments.sheet is not None and arguments.source_file is None:
        raise _InvocationError("--sheet requires INPUT_FILE; choose an Excel worksheet explicitly.")
    if arguments.metadata_sheet is not None and arguments.metadata is None:
        raise _InvocationError("--metadata-sheet requires --metadata PATH.")
    if any(value is not None and not value.strip() for value in (arguments.sheet, arguments.metadata_sheet)):
        raise _InvocationError("Worksheet names must be nonempty.")
    if arguments.source_file is not None:
        if arguments.source:
            raise _InvocationError("Use INPUT_FILE with optional --metadata, or named --source bindings, not both.")
        if arguments.input not in {None, "measurements"}:
            raise _InvocationError("INPUT_FILE is named measurements; use --source NAME=PATH to choose another name.")
        sources = {"measurements": arguments.source_file}
        if arguments.sheet is not None:
            sources["measurements"] = {"path": arguments.source_file, "format": "excel", "options": {"sheet_name": arguments.sheet}}
        if arguments.metadata is not None:
            sources["metadata"] = arguments.metadata
            if arguments.metadata_sheet is not None:
                sources["metadata"] = {"path": arguments.metadata, "format": "excel", "options": {"sheet_name": arguments.metadata_sheet}}
        return sources, "measurements"
    if arguments.metadata is not None:
        raise _InvocationError("--metadata requires INPUT_FILE; otherwise bind both files with --source NAME=PATH.")
    if not arguments.source:
        raise _InvocationError("Provide INPUT_FILE or --source NAME=PATH to start from your own data.")
    return _sources(arguments.source), arguments.input


def _ask(prompt, *, literal=False):
    """Read study answers; exact-value prompts preserve uncertainty words as data."""
    from ._presentation import _name
    sys.stderr.write(_name(prompt) + "\n> ")
    sys.stderr.flush()
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    answer = line.rstrip("\r\n") if literal else line.strip()
    if answer == "" or not literal and answer.casefold() in {"?", "not sure", "don't know", "unknown", "skip"}:
        return None
    return answer


def _choose(prompt, choices):
    labels = list(choices)
    suffix = " / ".join(labels)
    while True:
        answer = _ask(prompt + " [" + suffix + "; Enter = not sure]")
        if answer is None:
            return None
        for index, label in enumerate(labels, 1):
            if answer.casefold() == label.casefold() or answer == str(index):
                return choices[label]
        sys.stderr.write("Choose one of the listed answers, or press Enter to leave this undecided.\n")


def _columns(prompt, columns):
    from ._presentation import _name
    if columns:
        sys.stderr.write("Columns: " + ", ".join(f"{index}. {_name(name)}" for index, name in enumerate(columns, 1)) + "\n")
    while True:
        answer = _ask(prompt + " Enter names or column numbers separated by commas; Enter = not sure.")
        if answer is None:
            return None
        if answer.casefold() == "none":
            return []
        try:
            values = next(csv.reader([answer], skipinitialspace=True, strict=True))
        except csv.Error:
            values = []
        chosen = []
        for value in values:
            value = value.strip()
            if value in columns:
                chosen.append(value)
            elif value.isdecimal() and 1 <= int(value) <= len(columns):
                chosen.append(columns[int(value) - 1])
            else:
                break
        else:
            if chosen and len(set(chosen)) == len(chosen):
                return chosen
        sys.stderr.write("Use distinct names or numbers from the displayed columns.\n")


def _values(prompt, dtype):
    """Parse declared values according to their observed/declared column type."""
    while True:
        answer = _ask(prompt + " Separate values with commas; quote values containing commas or surrounding spaces. Every nonempty entry is a data value; Enter = not sure.", literal=True)
        if answer is None:
            return None
        try:
            values = next(csv.reader([answer], skipinitialspace=True, strict=True))
            if dtype.startswith(("Int", "UInt")):
                values = [int(value) for value in values]
            elif dtype.startswith(("Float", "Decimal")):
                values = [float(value) for value in values]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError
            elif dtype == "Boolean":
                labels = {"true": True, "false": False}
                values = [labels[value.casefold()] for value in values]
            if values:
                return values
        except (ValueError, KeyError, csv.Error):
            pass
        sys.stderr.write("Enter values that match this column: text, whole numbers, decimal numbers, or true/false as stated.\n")


def _answer(question, answers):
    """Translate researcher-facing questions into the shared answer schema."""
    from ._presentation import _name
    identifier = question["id"]
    columns, schema = question.get("columns", []), question.get("schema", {})
    if identifier == "input":
        choices = question.get("choices", list(question.get("sources", {})))
        return _choose("Which file contains the observations you want to prepare?", {name: name for name in choices})
    if identifier == "observation":
        return _ask("What does one row represent in this study? For example: one blood sample from one visit.")
    if identifier == "key":
        result = _columns("Which column or columns uniquely identify one observation?", columns)
        if result == []:
            sys.stderr.write("An observation identifier is required. This decision remains unresolved.\n")
            return None
        return result
    if identifier == "measurements":
        from ._start import _UNRESOLVED_UNITS
        selected = _columns("Which columns contain measurements? Type none if there are no measurement columns.", columns)
        if selected is None:
            return None
        previous = answers.get("measurements") or {}
        result = {column: {"dtype": previous.get(column, {}).get("dtype"),
                           "unit": previous.get(column, {}).get("unit")} for column in selected}
        answers["measurements"] = result  # Retain selected fields and completed choices even if input ends.
        for column in selected:
            choices = {"decimal numbers (approximate)": "Float64", "whole numbers (exact)": "Int64"}
            if schema.get(column, "").startswith(("Int", "UInt", "Float", "Decimal")):
                choices["keep observed numeric type"] = "keep"
            if result[column]["dtype"] is None:
                result[column]["dtype"] = _choose("For " + _name(column) + ", how should values be read? Decimal numbers use 64-bit floating point; check that this precision fits the study.", choices)
            else:
                label = next(label for label, value in choices.items() if value == result[column]["dtype"])
                sys.stderr.write("Recorded number representation for " + _name(column) + ": " + label + "\n")
            unit = result[column]["unit"]
            if unit is None or unit.strip().lower() in _UNRESOLVED_UNITS:
                result[column]["unit"] = _ask("What unit does " + _name(column) + " use? Enter the unit recorded by your study, or 1 for a dimensionless measurement. Unknown units remain unresolved.")
            else:
                sys.stderr.write("Recorded unit for " + _name(column) + ": " + _name(unit) + "\n")
        return result
    if identifier == "missing":
        coded = _choose("Are special values such as NA or -999 used to mean missing?", {"yes": True, "no": False})
        if coded is None:
            return None
        codes = {}
        if coded:
            selected = _columns("Which columns contain those missing-value codes?", columns)
            if not selected:
                return None
            for column in selected:
                values = _values("Which values mean missing in " + _name(column) + "? Observed type: " + _name(schema.get(column, "String")) + ".", schema.get(column, "String"))
                if values is None:
                    return None
                codes[column] = values
        action = _choose("After those codes are marked missing, may missing values remain?", {"keep missing values": "keep", "stop if any are missing": "error"})
        return None if action is None else {"codes": codes, "action": action}
    if identifier == "exclusions":
        action = _choose("Does the study have a rule for excluding observations?", {"keep all observations": "none", "keep specified values in one column": "keep_values"})
        if action is None:
            return None
        if action == "none":
            return {"action": "none"}
        chosen = _columns("Which one column determines whether an observation is kept?", columns)
        if chosen is None:
            return None
        if len(chosen) != 1:
            sys.stderr.write("This guide supports a rule on one column. Leave this undecided and describe a more complex rule in the protocol.\n")
            return None
        column = chosen[0]
        declared = (answers.get("measurements") or {}).get(column, {}).get("dtype") or "keep"
        dtype = schema.get(column, "String") if declared == "keep" else declared
        values = _values("Which values in " + _name(column) + " should be kept? Type: " + _name(dtype) + ".", dtype)
        if values is None:
            return None
        reason = _ask("What study rule justifies removing the other observations? Record the reason.")
        if reason is None:
            return None
        nulls = _choose("What if the exclusion column is missing?", {"stop and review": "error", "keep that observation": "keep", "exclude that observation": "drop"})
        return None if nulls is None else {"action": "keep_values", "column": column, "values": values, "reason": reason, "nulls": nulls}
    if identifier == "matching":
        action = _choose("Should information from the other file be attached to these observations?", {"do not attach": "none", "attach matching information": "attach"})
        if action is None:
            return None
        if action == "none":
            return {"action": "none"}
        other = question.get("sources", {})
        source = next(iter(other)) if len(other) == 1 else _choose("Which file contains the information to attach?", {name: name for name in other})
        if source is None:
            return None
        left = _columns("Which columns identify matches in the main dataset?", columns)
        if not left:
            return None
        right = _columns("Which columns identify matches in " + _name(source) + "? Select them in the same order.", other[source]["columns"])
        if not right:
            return None
        cardinality = _choose("Can several observations share the same identifier in the metadata? The metadata must have at most one row per matching identifier.", {"one observation per identifier": "1:1", "several observations per identifier": "m:1"})
        unmatched = _choose("If an observation has no matching information, what should happen?", {"stop and review": "error", "keep the observation": "keep"})
        unused = _choose("If the other file contains an identifier absent from the observations, what should happen?", {"stop and review": "error", "allow unused metadata": "drop"})
        overlap = _choose("If both files contain the same non-matching column name, what should happen?", {"stop and review": "error", "keep both with a metadata suffix": "suffix"})
        if None in (cardinality, unmatched, unused, overlap):
            return None
        result = {"action": "attach", "source": source, "left_on": left, "right_on": right, "cardinality": cardinality, "unmatched": unmatched, "unused": unused, "overlap": overlap, "nulls": "error"}
        if overlap == "suffix":
            suffix = _ask("What suffix should distinguish columns from the metadata file? For example: _metadata.")
            if suffix is None:
                return None
            result["suffix"] = suffix
        sys.stderr.write("Matching identifiers must be present, and each identifier in the metadata must identify at most one row.\n")
        return result
    return None


def _start(arguments, as_json):
    from ._protocol import load_recipe
    if as_json and arguments.interactive:
        raise _InvocationError("--json never asks questions; use --answers to provide decisions, or omit --json for interactive setup.")
    sources, selected = _start_sources(arguments)
    from ._start import draft, publish
    answers = load_recipe(arguments.answers) if arguments.answers else {}
    result = draft(sources, answers=answers, input=selected)
    interactive = arguments.interactive or not as_json and sys.stdin.isatty() and sys.stderr.isatty()
    if interactive and result["questions"]:
        sys.stderr.write("Protocol setup. Your answers record study decisions; no dataset is prepared yet.\nPress Enter or type not sure to leave a decision unresolved.\n")
        asked = set()
        try:
            while questions := [item for item in result["questions"] if item["id"] not in asked]:
                if result["answers"].get("input") is None:
                    questions = [item for item in questions if item["id"] == "input"]
                for question in questions:
                    asked.add(question["id"])
                    value = _answer(question, answers)
                    if value is not None:
                        answers[question["id"]] = value
                        if question["id"] == "matching":
                            break  # Refresh exclusion columns after the approved metadata mapping.
                result = draft(sources, answers=answers, input=selected)
                if result["answers"].get("input") is None:
                    break
        except EOFError:
            sys.stderr.write("Input ended. Unanswered decisions remain explicit in the draft.\n")
            result = draft(sources, answers=answers, input=selected)
    if arguments.output is not None:
        result = publish(result, arguments.output)
    return result

def _emit(value, stream):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _write(value, stream):
    stream.write(value + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Readable output by default; --json preserves engine structures and error codes."""
    tokens = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in tokens
    try:
        arguments = _parser().parse_args(tokens)
        as_json = arguments.json
        if arguments.command == "inspect":
            result = inspect(arguments.source, sample_rows=arguments.sample_rows, columns=arguments.columns, summary=arguments.summary, groups=arguments.groups, max_groups=arguments.max_groups, baseline=arguments.baseline)
            text = None if as_json else render_inspect(result)
        elif arguments.command == "catalog":
            result = catalog() if arguments.operation is None else operation_document(arguments.operation)
            text = None if as_json else render_catalog(result)
        elif arguments.command == "start":
            result = _start(arguments, as_json)
            text = None if as_json else render_start(result)
        elif arguments.command == "example":
            result = _example(arguments.directory)
            text = "Example ready: " + result["path"] + "\nRead README.md, then run:\n  " + "\n  ".join(result["next_commands"]) + "\nEdit recipe.yaml to record your study decisions before adapting it to your files."
        else:
            result = prepare(_sources(arguments.source), arguments.recipe, output=arguments.output, execution=arguments.execution).receipt
            text = None if as_json else render_prepare(result, arguments.output)
            if not as_json and arguments.output is None:
                text += "\nNot saved. Add --output NEW_DIRECTORY to retain the table and evidence."
        if as_json:
            _emit(result, sys.stdout)
        else:
            _write(text, sys.stdout)
        return 0
    except _InvocationError as error:
        _emit(error.to_dict(), sys.stderr) if as_json else _write(render_error(error), sys.stderr)
        return 2
    except WrangleError as error:
        _emit(error.to_dict(), sys.stderr) if as_json else _write(render_error(error), sys.stderr)
        return 1
    except KeyboardInterrupt:
        if arguments.command != "start":
            raise
        wrapped = WrangleError("START_CANCELLED", "Protocol setup was cancelled.")
        _emit(wrapped.to_dict(), sys.stderr) if as_json else _write(render_error(wrapped), sys.stderr)
        return 130
    except OSError as error:
        details = {"path": str(error.filename)} if error.filename is not None else {}
        wrapped = WrangleError("IO_ERROR", str(error), details)
        _emit(wrapped.to_dict(), sys.stderr) if as_json else _write(render_error(wrapped), sys.stderr)
        return 1
