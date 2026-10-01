"""Discover operation contracts directly from installed implementation definitions."""
from __future__ import annotations

import ast
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
import textwrap

from ._core import WrangleError


def operations():
    """Yield operation callables from the package's own implementation modules."""
    from ._api import _SIMPLE, _join
    yield from {**_SIMPLE, "join": _join}.items()
    root = Path(__file__).parent
    for group in ("df", "col", "array", "dic", "utils"):
        for source in sorted((root / group).glob("*.py")):
            if source.name.startswith("_"):
                continue
            module = importlib.import_module(f"wrangle.{group}.{source.stem}")
            for name, function in inspect.getmembers(module, inspect.isfunction):
                if not name.startswith("_") and function.__module__ == module.__name__:
                    yield name, function


def resolve(name):
    """Resolve a documented operation; arbitrary imports and callbacks are forbidden."""
    for operation, function in operations():
        if operation == name:
            return function
    raise WrangleError("UNKNOWN_OPERATION", "Choose an operation from the shipped catalog.", {"operation": name})


@lru_cache(maxsize=None)
def _definition(function):
    source = textwrap.dedent(inspect.getsource(function))
    return ast.parse(source).body[0]


def _nodes(tree):
    """Walk one definition without confusing nested helper returns with its output."""
    yield tree
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield from _nodes(child)


@lru_cache(maxsize=None)
def _local_functions(function, tree):
    """Resolve only imports already present in this trusted package's source."""
    names = dict(function.__globals__)
    package = function.__module__.rsplit(".", 1)[0]
    for node in _nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = importlib.util.resolve_name("." * node.level + node.module, package) if node.level else node.module
            if module_name.startswith("wrangle."):
                module = importlib.import_module(module_name)
                for alias in node.names:
                    if alias.name != "*":
                        names[alias.asname or alias.name] = getattr(module, alias.name)
    return names


def _validation(function, *, include_boundaries=False):
    """Expose declared failures from the call graph; shared boundaries are stored once."""
    pending, visited, result = [function], set(), []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if not include_boundaries and current.__module__ == "wrangle._core":
            continue
        tree = _definition(current)
        scope = _local_functions(current, tree)
        for node in _nodes(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            target = scope.get(node.func.id)
            code_wrapper = (node.func.id in {"_fail", "_error"} and inspect.isfunction(target)
                            and next(iter(inspect.signature(target).parameters), None) == "code")
            if (node.func.id == "WrangleError" or code_wrapper) and node.args and isinstance(node.args[0], ast.Constant):
                code = node.args[0].value
                if not isinstance(code, str):
                    continue
                message = node.args[1] if len(node.args) > 1 else None
                text = message.value if isinstance(message, ast.Constant) and isinstance(message.value, str) else ast.unparse(message) if message else "Inspect WrangleError.details."
                result.append({"code": code, "precondition": text})
            else:
                if inspect.isfunction(target) and target.__module__.startswith("wrangle."):
                    # Fixed-code wrappers can take a message first, rather than a code.
                    # Resolve that message from the call site, never label it an error code.
                    fixed = []
                    if node.func.id in {"_fail", "_error"} and not code_wrapper:
                        parameters = list(inspect.signature(target).parameters)
                        supplied = dict(zip(parameters, node.args))
                        supplied.update({item.arg: item.value for item in node.keywords if item.arg})
                        for failure in _nodes(_definition(target)):
                            if (isinstance(failure, ast.Call) and isinstance(failure.func, ast.Name)
                                    and failure.func.id == "WrangleError" and len(failure.args) >= 2
                                    and isinstance(failure.args[0], ast.Constant) and isinstance(failure.args[0].value, str)):
                                message = failure.args[1]
                                if isinstance(message, ast.Name):
                                    message = supplied.get(message.id, message)
                                text = message.value if isinstance(message, ast.Constant) and isinstance(message.value, str) else ast.unparse(message)
                                fixed.append({"code": failure.args[0].value, "precondition": text})
                    if fixed:
                        result.extend(fixed)
                    else:
                        pending.append(target)
    unique = {(item["code"], item["precondition"]): item for item in result}
    return [unique[key] for key in sorted(unique)]


def _parameter_constraints(function):
    """Expose literal argument choices enforced by the trusted implementation."""
    result = []
    for node in _nodes(_definition(function)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_choice" or len(node.args) < 2:
            continue
        try:
            choices = ast.literal_eval(node.args[1])
        except (ValueError, TypeError):
            continue
        if not isinstance(choices, (tuple, list)) or not all(isinstance(value, (str, int, float, bool)) or value is None for value in choices):
            continue
        result.append({"argument": ast.unparse(node.args[0]), "allowed": list(choices)})
    return result


def operation_contract(function):
    """Read the explicit contract declared on the implementation; never infer it."""
    contract = getattr(function, "__wrangle_contract__", None)
    if contract is None:
        raise WrangleError("UNDOCUMENTED_OPERATION", "The operation definition must declare its output and recipe contract.", {"operation": function.__name__})
    return contract


def recipe_eligibility(function):
    """Return the declared yes/conditional/never recipe eligibility."""
    return operation_contract(function)["recipe"]


def catalog():
    """Return executable signatures, output kinds, validation codes, and source navigation."""
    root = Path(__file__).parent
    result = []
    for name, function in operations():
        arguments = []
        for parameter in inspect.signature(function).parameters.values():
            item = {"name": parameter.name, "required": parameter.default is inspect.Parameter.empty, "kind": parameter.kind.name.lower()}
            if parameter.default is not inspect.Parameter.empty:
                default = parameter.default
                item["default"] = default if default is None or isinstance(default, (str, int, float, bool, list, dict)) else str(default)
            if parameter.annotation is not inspect.Parameter.empty:
                item["type"] = str(parameter.annotation)
            arguments.append(item)
        validation = _validation(function)
        contract = operation_contract(function)
        source_lines, line = inspect.getsourcelines(function)
        line += next(index for index, source in enumerate(source_lines) if source.lstrip().startswith("def "))
        result.append({"name": name, "group": function.__module__.split(".")[1] if function.__module__ != "wrangle._api" else "recipe", "symbol": f"{function.__module__}.{function.__name__}", "manual": f"docs/operations/{name}.json", "source": str(Path(inspect.getfile(function)).relative_to(root)), "line": line, "arguments": arguments, "description": inspect.getdoc(function) or "", "returns": sorted(contract["returns"]), "recipe": {"eligibility": contract["recipe"], "aggregates": contract["aggregates"]}, "validation": validation, "argument_constraints": _parameter_constraints(function), "boundary_validation": {"$ref": "#/boundary_validation"}, "retired": contract["retired"]})
    from ._core import as_series, frame, require_columns, reject_destructive, require_native
    common = []
    for boundary in (as_series, frame, require_columns, reject_destructive, require_native):
        common.extend(_validation(boundary, include_boundaries=True))
    unique = {(item["code"], item["precondition"]): item for item in common}
    from ._api import prepare, inspect as inspect_source
    from ._execution import GRAIN_OPERATIONS
    from ._api import Prepared
    from ._storage import DiskWorkspace, verify_evidence, verify_report
    from ._start import QUESTION_DEFINITIONS, draft, publish
    shared = _validation(draft) + _validation(publish) + _validation(prepare) + _validation(inspect_source) + _validation(DiskWorkspace.snapshot) + _validation(DiskWorkspace.record) + _validation(DiskWorkspace.publish) + _validation(Prepared._source_table) + _validation(verify_evidence) + _validation(verify_report)
    errors = {(item["code"], item["precondition"]): item for item in shared}
    from ._recipe_expressions import EXPRESSION_DEFINITIONS
    return {
        "version": 1,
        "recipe_contract": "Author recipe files in strict YAML (.yaml/.yml) using mappings, lists, strings, Boolean/null values and finite numbers; programmatic Python mappings share the same contract. Canonical JSON is internal receipt/hash serialization. Guided pending_decisions block execution before source access; recorded research_decisions must be complete and agree with controlled preparation steps. Conditional legacy operations require table-returning arguments. Existing keys retain identity; declare step.key for an explicit checked transition. Canonical grain operations require step.key even without an input key. Exclusions require reasons, expansion requires allow_expand, and incompatible measurement units cannot be reconciled by annotations.",
        "guided_start": {
            "symbols": {"draft": "wrangle._start.draft", "publish": "wrangle._start.publish", "decision_validation": "wrangle._start.validate_decisions"},
            "manual": "docs/recipes.md#guided-start", "workflow": "docs/start_workflow.py",
            "questions": list(QUESTION_DEFINITIONS),
            "scope": "One local observation file plus optional metadata. Drafting observes but never prepares data. Answers supply meaning, identity, measurement representations/units, exact missing codes, matching and QC; no scientific choice is inferred.",
            "readiness": "ready means required answers are complete; data_validated is always false. prepare performs native data checks and execution.",
            "recovery": "Resolve pending questions in answers.yaml and rerun start. Never remove blockers; recorded answers and controlled steps must agree.",
        },
        "grain_operations": sorted(GRAIN_OPERATIONS),
        "expression_grammar": EXPRESSION_DEFINITIONS,
        "recovery": "Read operation validation plus shared_validation and boundary_validation. Use WrangleError.code/details to correct that declared precondition and rerun the same protocol; never silently substitute a method or discard observations.",
        "shared_validation": [errors[key] for key in sorted(errors)],
        "boundary_validation": [unique[key] for key in sorted(unique)],
        "navigation": {
            "paths_relative_to": "Path(wrangle.__file__).parent",
            "entrypoint": "AGENTS.md", "recipe_grammar": "docs/recipes.md",
            "draft_protocol": "_start.py:draft", "publish_protocol": "_start.py:publish", "decision_validation": "_start.py:validate_decisions", "guided_workflow": "docs/start_workflow.py",
            "human_start": "docs/getting_started.md", "human_workflow": "docs/human_workflow.py", "starter_recipe": "docs/starter/recipe.yaml", "recipe_loader": "_protocol.py:load_recipe",
            "migration": "docs/migration.md", "executable_workflow": "docs/research_batch.py",
            "preparation_workflows": "docs/preparation_workflows.py",
            "inspection_workflow": "docs/inspection_workflow.py",
            "expression_workflow": "docs/expression_workflow.py",
            "statistics_workflow": "docs/frozen_parameters.py",
            "disk_workflow": "docs/disk_preparation.py", "disk_execution": "_storage.py",
            "execution_contracts": "_execution.py", "source_contracts": "_sources.py",
            "checks": "_contracts.py", "cli": "_cli.py:main",
            "architecture": "docs/architecture.md", "security": "docs/security.md",
            "security_assurance": "docs/security/assurance.md", "release_verification": "docs/security/releases.md",
            "contributing": "docs/project/CONTRIBUTING.md", "governance": "docs/project/GOVERNANCE.md",
            "security_reporting": "docs/project/SECURITY.md", "openssf_evidence": "docs/security/openssf-evidence.yaml",
            "select_operation": "Choose the task in docs/recipes.md. Read returns/recipe eligibility, then exact arguments, literal argument_constraints and validation. Resolve symbol or open source at line. Minimal workflows in navigation assert expected data and receipts. Signatures, choice constraints and failures come from engine definitions.",
        },
        "operations": result,
    }


def operation_document(name, document=None):
    """Return one operation's compact, file-equivalent agent contract."""
    document = catalog() if document is None else document
    entries = [entry for entry in document["operations"] if entry["name"] == name]
    if not entries:
        raise WrangleError("UNKNOWN_OPERATION", "Choose an operation from the shipped catalog.", {"operation": name})
    return {
        "version": document["version"], "navigation": document["navigation"],
        "recipe_contract": document["recipe_contract"], "recovery": document["recovery"],
        "operations": entries, "boundary_validation": document["boundary_validation"],
        "shared_validation": {"$ref": "docs/operations.json#/shared_validation"},
        "expression_grammar": {"$ref": "docs/operations.json#/expression_grammar"},
        "guided_start": {"$ref": "docs/operations.json#/guided_start"},
    }
