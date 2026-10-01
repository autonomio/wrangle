"""Strict, readable YAML recipes; execution and canonical evidence stay separate."""
from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
import math
from pathlib import Path
import re
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.events import (
    AliasEvent, DocumentStartEvent, MappingEndEvent, MappingStartEvent,
    SequenceEndEvent, SequenceStartEvent,
)
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
from ruamel.yaml.tokens import DirectiveToken

from ._core import WrangleError


_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_LEADING_ZERO = re.compile(r"[+-]?0[0-9_]+\Z")
_NONFINITE = re.compile(r"[+-]?\.(?:inf|nan)\Z", re.IGNORECASE)
_MAX_DEPTH = 64


def _yaml() -> YAML:
    parser = YAML(typ="safe", pure=True)
    parser.version = (1, 2)
    parser.allow_duplicate_keys = False
    return parser


def _error(code: str, message: str, path: Path | None = None, mark=None, **details):
    if path is not None:
        details["path"] = str(path)
    if mark is not None:
        details.update(line=mark.line + 1, column=mark.column + 1)
    raise WrangleError(code, message, details)


def _scalar(node: ScalarNode, path: Path) -> str | int | float | bool | None:
    value = node.value
    # Quotes and block scalars always carry text; resolver tags never coerce it.
    if node.style is not None:
        return value
    if _LEADING_ZERO.fullmatch(value):
        _error("YAML_AMBIGUOUS_SCALAR", "Quote identifiers with leading zeros, such as \"001\"; use ordinary decimal notation for numbers.", path, node.start_mark, value=value)
    if _NONFINITE.fullmatch(value):
        _error("YAML_NONFINITE", "Recipes require finite numbers. Use null for declared missingness, or quote this value if it is text.", path, node.start_mark, value=value)
    if value in {"", "null"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        if _INTEGER.fullmatch(value):
            return int(value)
        if _NUMBER.fullmatch(value):
            number = float(value)
            if not math.isfinite(number):
                _error("YAML_NONFINITE", "This number exceeds finite numeric range; correct it or quote it if it is text.", path, node.start_mark, value=value)
            return number
    except WrangleError:
        raise
    except ValueError:
        _error("YAML_AMBIGUOUS_SCALAR", "This numeric scalar cannot be represented; correct it or quote it if it is text.", path, node.start_mark, value=value)
    # In particular yes/no/on/off and dates remain strings, unlike YAML 1.1.
    return value


def _construct(node, path: Path):
    if isinstance(node, ScalarNode):
        return _scalar(node, path)
    if isinstance(node, SequenceNode):
        return [_construct(child, path) for child in node.value]
    if isinstance(node, MappingNode):
        result = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode):
                _error("INVALID_RECIPE", "Recipe field names must be strings; use a simple field name or quote it.", path, key_node.start_mark)
            key = _scalar(key_node, path)
            if not isinstance(key, str):
                _error("INVALID_RECIPE", "Recipe field names must be strings; quote this field name.", path, key_node.start_mark, key=key)
            if key == "<<" and key_node.style is None:
                _error("YAML_UNSUPPORTED_FEATURE", "Merge keys are not allowed. Write each recipe declaration explicitly.", path, key_node.start_mark, feature="merge_key")
            if key in result:
                _error("INVALID_YAML", "Each recipe field must be declared once; remove or rename this duplicate field.", path, key_node.start_mark, key=key, reason="duplicate_key")
            result[key] = _construct(value_node, path)
        return result
    _error("INVALID_YAML", "Use only mappings, lists, and ordinary scalar values in recipes.", path, node.start_mark)


def _load_text(text: str, path: Path) -> dict[str, Any]:
    """Check syntax features before constructing any recipe values."""
    try:
        # Scan directives first: ruamel's resolver asserts on unknown minor
        # versions before emitting the document event with its source location.
        for token in _yaml().scan(text):
            if isinstance(token, DirectiveToken):
                if token.name == "YAML" and token.value != (1, 2):
                    _error("YAML_UNSUPPORTED_FEATURE", "Recipe files use YAML 1.2; remove this version directive or declare %YAML 1.2.", path, token.start_mark, feature="yaml_version")
                if token.name == "TAG":
                    _error("YAML_UNSUPPORTED_FEATURE", "Tag directives are not allowed; write ordinary recipe values.", path, token.start_mark, feature="tag")
        documents = depth = 0
        for event in _yaml().parse(text):
            if isinstance(event, DocumentStartEvent):
                documents += 1
                if documents > 1:
                    _error("YAML_UNSUPPORTED_FEATURE", "A recipe file must contain one YAML document; put each protocol in its own file.", path, event.start_mark, feature="multiple_documents")
                if event.version not in {None, (1, 2)}:
                    _error("YAML_UNSUPPORTED_FEATURE", "Recipe files use YAML 1.2; remove this version directive or declare %YAML 1.2.", path, event.start_mark, feature="yaml_version")
                if event.tags:
                    _error("YAML_UNSUPPORTED_FEATURE", "Tag directives are not allowed; write ordinary recipe values.", path, event.start_mark, feature="tag")
            if isinstance(event, AliasEvent):
                _error("YAML_UNSUPPORTED_FEATURE", "Aliases are not allowed. Write each recipe declaration explicitly.", path, event.start_mark, feature="alias")
            if getattr(event, "anchor", None) is not None:
                _error("YAML_UNSUPPORTED_FEATURE", "Anchors are not allowed. Write each recipe declaration explicitly.", path, event.start_mark, feature="anchor")
            if getattr(event, "tag", None) is not None:
                _error("YAML_UNSUPPORTED_FEATURE", "Explicit tags are not allowed. Use ordinary values, and quote values that must remain text.", path, event.start_mark, feature="tag")
            if isinstance(event, (MappingStartEvent, SequenceStartEvent)):
                depth += 1
                if depth > _MAX_DEPTH:
                    _error("YAML_UNSUPPORTED_FEATURE", "The recipe is nested too deeply; simplify the expression or separate preparation steps.", path, event.start_mark, feature="nesting", maximum=_MAX_DEPTH)
            elif isinstance(event, (MappingEndEvent, SequenceEndEvent)):
                depth -= 1
        node = _yaml().compose(text)
        if not isinstance(node, MappingNode):
            _error("INVALID_RECIPE", "A recipe must be one YAML mapping with fields such as version, key, and steps.", path, getattr(node, "start_mark", None))
        return _construct(node, path)
    except WrangleError:
        raise
    except (YAMLError, UnicodeError, ValueError, RecursionError, AssertionError) as error:
        mark = getattr(error, "problem_mark", None) or getattr(error, "context_mark", None)
        _error("INVALID_YAML", "The YAML recipe cannot be read. Correct its indentation, quotes, or brackets at the reported location.", path, mark, error=str(error))


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Read one .yaml/.yml recipe as JSON-compatible values, without inference.

    Lowercase true/false/null and decimal JSON numbers are typed. Other bare
    scalars, including dates and yes/no, remain strings. Quotes always mean text.
    Failures include a stable code and one-based line/column where available.
    """
    try:
        source = Path(path)
    except TypeError as error:
        raise WrangleError("RECIPE_FORMAT", "Use the path to a .yaml or .yml recipe file.") from error
    if source.suffix.lower() not in {".yaml", ".yml"}:
        _error("RECIPE_FORMAT", "Recipe files must use .yaml or .yml. Convert the recipe to YAML before preparing data.", source, suffix=source.suffix)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        _error("RECIPE_IO", "The recipe file could not be read; check its path, access permissions, and UTF-8 encoding.", source, error=str(error))
    return _load_text(text, source)


def _plain(value, active: set[int], location: str = "$", depth: int = 0):
    """Copy only JSON values, removing shared identities that could emit aliases."""
    if depth > _MAX_DEPTH:
        _error("INVALID_RECIPE", "The recipe is nested too deeply; simplify the expression or separate preparation steps.", location=location)
    if type(value) in {str, int, bool, type(None)}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _error("YAML_NONFINITE", "Recipes require finite numbers; use null for declared missingness.", location=location)
        return value
    if type(value) not in {dict, list}:
        _error("INVALID_RECIPE", "Recipes support only mappings, lists, strings, finite numbers, booleans, and null.", location=location, type=type(value).__name__)
    identity = id(value)
    if identity in active:
        _error("INVALID_RECIPE", "Recipes cannot contain circular references; write each declaration explicitly.", location=location)
    active.add(identity)
    try:
        if type(value) is list:
            return [_plain(child, active, f"{location}[{index}]", depth + 1) for index, child in enumerate(value)]
        result = {}
        for key, child in value.items():
            if type(key) is not str:
                _error("INVALID_RECIPE", "Recipe field names must be strings.", location=location, type=type(key).__name__)
            result[key] = _plain(child, active, f"{location}.{key}", depth + 1)
        return result
    finally:
        active.remove(identity)


def normalize_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a Python recipe to plain finite JSON values, without type coercion.

    Python mappings at the outer boundary are supported. Nested values must be
    plain dictionaries/lists/scalars; callbacks and custom objects are rejected.
    """
    if not isinstance(recipe, Mapping):
        _error("INVALID_RECIPE", "A recipe must be one mapping with fields such as version, key, and steps.")
    return _plain(recipe if type(recipe) is dict else dict(recipe), set())


def dump_recipe(recipe: Mapping[str, Any]) -> str:
    """Write readable YAML with the same values; comments are not execution state."""
    plain = normalize_recipe(recipe)
    emitter = YAML(typ="safe", pure=True)
    emitter.default_flow_style = False
    emitter.sort_base_mapping_type_on_output = False
    emitter.allow_unicode = True
    emitter.width = 88
    emitter.indent(mapping=2, sequence=4, offset=2)
    stream = StringIO()
    emitter.dump(plain, stream)
    return stream.getvalue()
