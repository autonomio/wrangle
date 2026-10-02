"""YAML protocols preserve scientific declarations without hidden coercion."""
from collections import UserDict
from datetime import date
import json
from pathlib import Path

import pytest

from wrangle._core import WrangleError
from wrangle._protocol import dump_recipe, load_recipe, normalize_recipe


def recipe_file(tmp_path, content, name="protocol.yaml"):
    source = tmp_path / name
    source.write_text(content, encoding="utf-8")
    return source


def fails(tmp_path, content, code, **details):
    with pytest.raises(WrangleError) as caught:
        load_recipe(recipe_file(tmp_path, content))
    assert caught.value.code == code
    for key, expected in details.items():
        assert caught.value.details[key] == expected
    assert caught.value.details["path"].endswith("protocol.yaml")
    return caught.value


def test_human_recipe_comments_and_lists_preserve_values(tmp_path):
    source = recipe_file(tmp_path, '''# Instrument protocol; missingness was agreed with the researcher.
version: 1
name: Assay batch
key: sample_id
units:
  concentration: mg/L
steps:
  - op: normalize_missing
    columns:
      concentration: ["NA", "001", "null"]
  - op: cast
    columns:
      concentration: Float64
checks:
  row_count:
    min: 0
  ranges:
    concentration:
      min: 0.001
''')
    assert load_recipe(source) == {
        "version": 1, "name": "Assay batch", "key": "sample_id",
        "units": {"concentration": "mg/L"},
        "steps": [
            {"op": "normalize_missing", "columns": {"concentration": ["NA", "001", "null"]}},
            {"op": "cast", "columns": {"concentration": "Float64"}},
        ],
        "checks": {"row_count": {"min": 0}, "ranges": {"concentration": {"min": 0.001}}},
    }


@pytest.mark.parametrize("name", ["protocol.yaml", "protocol.yml", "protocol.YAML"])
def test_yaml_extensions_are_accepted(tmp_path, name):
    assert load_recipe(recipe_file(tmp_path, "steps: []\n", name)) == {"steps": []}


@pytest.mark.parametrize("name", ["protocol.json", "protocol.txt", "protocol"])
def test_authored_json_recipe_files_are_rejected(tmp_path, name):
    with pytest.raises(WrangleError) as caught:
        load_recipe(recipe_file(tmp_path, '{"steps": []}', name))
    assert caught.value.code == "RECIPE_FORMAT"
    assert ".yaml" in str(caught.value)


def test_missing_recipe_and_invalid_utf8_are_stable_failures(tmp_path):
    with pytest.raises(WrangleError) as caught:
        load_recipe(tmp_path / "absent.yaml")
    assert caught.value.code == "RECIPE_IO"
    source = tmp_path / "bad.yaml"
    source.write_bytes(b"name: \xff")
    with pytest.raises(WrangleError) as caught:
        load_recipe(source)
    assert caught.value.code == "RECIPE_IO"


def test_yaml_json_scalar_semantics_do_not_infer_dates_or_yes_no(tmp_path):
    result = load_recipe(recipe_file(tmp_path, '''values:
  - yes
  - no
  - on
  - off
  - true
  - false
  - null
  -
  - TRUE
  - False
  - Null
  - 2026-10-01
  - 2026-10-01T13:14:15Z
  - "001"
  - "true"
  - 'null'
  - 0
  - -3
  - 1.5
  - 1e3
  - -0.0
  - .5
  - 0xFF
  - 1_000
'''))
    assert result == {"values": [
        "yes", "no", "on", "off", True, False, None, None,
        "TRUE", "False", "Null", "2026-10-01", "2026-10-01T13:14:15Z",
        "001", "true", "null", 0, -3, 1.5, 1000.0, -0.0, ".5", "0xFF", "1_000",
    ]}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", ["001", "00", "-001", "+001", "0_01"])
def test_leading_zero_identifiers_require_quotes(tmp_path, value):
    error = fails(tmp_path, f"values: [{value}]\n", "YAML_AMBIGUOUS_SCALAR", line=1, column=10, value=value)
    assert "Quote" in str(error)
    assert load_recipe(recipe_file(tmp_path, f'values: ["{value}"]\n')) == {"values": [value]}


@pytest.mark.parametrize("value", [".nan", ".NaN", ".inf", "+.INF", "-.inf", "1e999"])
def test_nonfinite_numbers_require_an_explicit_correction(tmp_path, value):
    fails(tmp_path, f"value: {value}\n", "YAML_NONFINITE", line=1, column=8)
    assert load_recipe(recipe_file(tmp_path, f'value: "{value}"\n')) == {"value": value}


@pytest.mark.parametrize("content,key,line,column", [
    ("version: 1\nversion: 1\n", "version", 2, 1),
    ("steps:\n  - op: cast\n    columns:\n      mass: Float64\n      mass: Int64\n", "mass", 5, 7),
    ("steps: []\n'steps': []\n", "steps", 2, 1),
])
def test_duplicate_keys_are_rejected_without_losing_a_declaration(tmp_path, content, key, line, column):
    fails(tmp_path, content, "INVALID_YAML", reason="duplicate_key", key=key, line=line, column=column)


@pytest.mark.parametrize("content", ["1: value\n", "true: value\n", "null: value\n", "? [key, other]\n: value\n"])
def test_mapping_keys_must_be_strings(tmp_path, content):
    fails(tmp_path, content, "INVALID_RECIPE")


def test_quoted_numeric_field_names_remain_strings(tmp_path):
    assert load_recipe(recipe_file(tmp_path, 'columns: {"1": String, "true": String}\n')) == {"columns": {"1": "String", "true": "String"}}


@pytest.mark.parametrize("content,feature", [
    ("steps: &steps []\n", "anchor"),
    ("steps: *steps\n", "alias"),
    ("value: !scientific-method abc\n", "tag"),
    ("value: !!python/object/apply:os.system ['echo unsafe']\n", "tag"),
    ("value: !!timestamp 2026-10-01\n", "tag"),
    ("value: !!str 001\n", "tag"),
    ("<<: {version: 1}\nsteps: []\n", "merge_key"),
    ("version: 1\n---\nsteps: []\n", "multiple_documents"),
    ("%YAML 1.1\n---\nsteps: []\n", "yaml_version"),
    ("%YAML 1.3\n---\nsteps: []\n", "yaml_version"),
    ("%YAML 2.0\n---\nsteps: []\n", "yaml_version"),
    ("%TAG !e! tag:example.com,2000:\n---\nsteps: []\n", "tag"),
])
def test_hidden_or_executable_yaml_features_are_rejected(tmp_path, content, feature):
    fails(tmp_path, content, "YAML_UNSUPPORTED_FEATURE", feature=feature)


def test_yaml_12_directive_is_accepted(tmp_path):
    assert load_recipe(recipe_file(tmp_path, "%YAML 1.2\n---\nsteps: []\n")) == {"steps": []}


@pytest.mark.parametrize("content", ["", "# comment only\n", "null\n", "- steps\n", "just text\n"])
def test_recipe_requires_a_single_mapping(tmp_path, content):
    fails(tmp_path, content, "INVALID_RECIPE")


def test_syntax_error_has_actionable_location(tmp_path):
    error = fails(tmp_path, "steps: [\n", "INVALID_YAML")
    assert error.details["line"] == 2
    assert error.details["column"] == 1
    assert "indentation" in str(error)


def test_deep_recipes_fail_without_python_recursion_failure(tmp_path):
    fails(tmp_path, "value: " + "[" * 65 + "0" + "]" * 65, "YAML_UNSUPPORTED_FEATURE", feature="nesting")


def test_dump_is_readable_and_preserves_json_values(tmp_path):
    recipe = {
        "version": 1, "name": "血液 assay", "key": "sample_id",
        "steps": [{"op": "recode", "column": "status", "mapping": {"yes": "001", "no": None, "on": False, "2026-10-01": True}}],
        "numeric": [0, -1, 1.5, 1000.0, -0.0], "multiline": "first\nsecond\n",
        "strings": ["true", "null", "001", "0", "1.0", ".nan", "1e999", "TRUE", "False", ".5", "1_000", "0xFF"],
    }
    content = dump_recipe(recipe)
    assert content.startswith("version: 1\n")
    assert "血液" in content
    assert "!!" not in content
    assert load_recipe(recipe_file(tmp_path, content)) == recipe
    assert json.dumps(load_recipe(recipe_file(tmp_path, content)), sort_keys=True) == json.dumps(recipe, sort_keys=True)


def test_dump_shared_values_never_emits_anchors(tmp_path):
    shared = [{"op": "select", "columns": ["id"]}]
    recipe = {"steps": shared, "copy": shared}
    content = dump_recipe(recipe)
    assert "&id" not in content
    assert "*id" not in content
    assert load_recipe(recipe_file(tmp_path, content)) == recipe


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_dump_rejects_nonfinite_values(value):
    with pytest.raises(WrangleError) as caught:
        dump_recipe({"value": value})
    assert caught.value.code == "YAML_NONFINITE"


@pytest.mark.parametrize("recipe", [{"value": date(2026, 10, 1)}, {"value": Path("local.csv")}, {"value": (1, 2)}, {1: "value"}, []])
def test_dump_rejects_objects_that_are_not_json_values(recipe):
    with pytest.raises(WrangleError) as caught:
        dump_recipe(recipe)
    assert caught.value.code in {"INVALID_RECIPE", "INVALID_RECIPE"}


def test_dump_rejects_recursive_python_values():
    recipe = {}
    recipe["recursive"] = recipe
    with pytest.raises(WrangleError) as caught:
        dump_recipe(recipe)
    assert caught.value.code == "INVALID_RECIPE"


def test_normalize_recipe_accepts_outer_mapping_and_preserves_identity():
    original = UserDict({"version": 1, "steps": [{"op": "select", "columns": ["id"]}]})
    normalized = normalize_recipe(original)
    assert type(normalized) is dict
    assert normalized == original
    normalized["steps"][0]["columns"].append("value")
    assert original["steps"][0]["columns"] == ["id"]


@pytest.mark.parametrize("value", [float("nan"), date(2026, 10, 1), (1, 2), lambda: None])
def test_python_recipe_rejects_unsupported_values_before_execution(value):
    with pytest.raises(WrangleError) as caught:
        normalize_recipe({"value": value})
    assert caught.value.code in {"INVALID_RECIPE", "YAML_NONFINITE"}


@pytest.mark.parametrize("length", [70, 90, 110, 128, 150, 1100])
def test_long_measurement_field_names_round_trip_without_wrapped_simple_keys(tmp_path, length):
    field = "sample_" + "a" * length + " [signal"
    recipe = {"descriptions": {field: "Observed signal"}}
    serialized = dump_recipe(recipe)
    assert load_recipe(recipe_file(tmp_path, serialized)) == recipe


@pytest.mark.parametrize("location", ["field", "name"])
def test_long_plain_text_retains_significant_repeated_spaces(tmp_path, location):
    value = "measurement_" + "a" * 90 + " " * 20 + "observed signal"
    recipe = {"descriptions": {value: "Meaning"}} if location == "field" else {"name": value}
    assert load_recipe(recipe_file(tmp_path, dump_recipe(recipe))) == recipe


@pytest.mark.parametrize("location", ["field", "name"])
@pytest.mark.parametrize("value", ["first\x85second", "a" * 90 + "\x85" + " " * 20 + "signal"])
def test_yaml_nel_characters_are_escaped_without_folding_values(tmp_path, location, value):
    recipe = {"descriptions": {value: "Meaning"}} if location == "field" else {"name": value}
    assert load_recipe(recipe_file(tmp_path, dump_recipe(recipe))) == recipe


@pytest.mark.parametrize("location", ["field", "name"])
def test_long_quoted_text_does_not_gain_spaces_after_escaped_bom(tmp_path, location):
    value = "sample_" + "a" * 90 + "\ufeff" + "b" * 60 + " [signal"
    recipe = {"descriptions": {value: "Meaning"}} if location == "field" else {"name": value}
    assert load_recipe(recipe_file(tmp_path, dump_recipe(recipe))) == recipe
