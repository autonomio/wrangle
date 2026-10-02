"""Bounded adversarial properties shared by the Atheris targets and corpus replay."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import wrangle as wr
from wrangle._core import WrangleError
from wrangle._protocol import _load_text, dump_recipe, normalize_recipe


MAX_INPUT_BYTES = 8192
INPUT_PATH = Path("fuzz-input.yaml")


def _canonical(value):
    # JSON distinguishes booleans from integers and preserves signed float zero.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _accepted(data: bytes):
    if len(data) > MAX_INPUT_BYTES:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        return _load_text(text, INPUT_PATH)
    except WrangleError as error:
        _canonical(error.to_dict())
        return None


def fuzz_protocol(data: bytes):
    """Every accepted YAML mapping must survive a deterministic exact round trip."""
    recipe = _accepted(data)
    if recipe is None:
        return
    expected = _canonical(recipe)
    _ensure(_canonical(normalize_recipe(recipe)) == expected, "Normalization changed YAML values.")
    serialized = dump_recipe(recipe)
    _ensure(serialized == dump_recipe(recipe), "YAML serialization was not deterministic.")
    _ensure(_canonical(_load_text(serialized, INPUT_PATH)) == expected, "YAML round trip changed values.")


def fuzz_expression(data: bytes):
    """Mutated expression trees must fail safely or preserve keys, data and evidence."""
    expression = _accepted(data)
    if expression is None:
        return
    source = pl.DataFrame({
        "id": ["001", "002", "003"], "x": [0, -1, 2],
        "measurement": [0.25, None, 2.5], "text": [" A ", None, "β"],
        "flag": [True, False, None],
    })
    original = source.clone()
    recipe = {"version": 1, "key": "id", "steps": [
        {"op": "derive", "columns": {"derived": expression}},
    ]}
    declared = _canonical(recipe)
    try:
        first = wr.prepare(source, recipe)
    except WrangleError as error:
        expected_error = _canonical(error.to_dict())
        try:
            wr.prepare(source, recipe)
        except WrangleError as repeated:
            _ensure(_canonical(repeated.to_dict()) == expected_error, "Expression errors were not repeatable.")
        else:
            raise AssertionError("The same expression unexpectedly succeeded after rejection.")
        _ensure(source.equals(original), "Preparation mutated its source table.")
        _ensure(_canonical(recipe) == declared, "Preparation mutated its declared recipe.")
        return
    second = wr.prepare(source, recipe)
    _ensure(source.equals(original), "Preparation mutated its source table.")
    _ensure(_canonical(recipe) == declared, "Preparation mutated its declared recipe.")
    _ensure(first.data.select(original.columns).equals(original), "Derivation changed input columns or keys.")
    _ensure(first.data.equals(second.data), "Expression output was not repeatable.")
    _ensure(_canonical(first.receipt) == _canonical(second.receipt), "Preparation evidence was not repeatable.")
