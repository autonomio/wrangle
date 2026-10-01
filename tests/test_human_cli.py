"""Researchers and agents use one checked workflow with explicit presentation."""
import json
from pathlib import Path

import polars as pl
import pytest

import wrangle
from wrangle._cli import main
from wrangle._presentation import render_error, render_prepare
from wrangle._protocol import dump_recipe


@pytest.fixture
def simple_protocol(tmp_path):
    source = tmp_path / "observations.csv"
    source.write_text("sample_id,mass_mg,qc\n001,1000,pass\n002,2000,fail\n003,3000,pass\n", encoding="utf-8")
    recipe = {
        "version": 1, "key": "sample_id", "units": {"mass_mg": "mg"},
        "steps": [
            {"op": "cast", "columns": {"mass_mg": "Float64"}},
            {"op": "filter", "where": {"eq": [{"col": "qc"}, "pass"]}, "reason": "Declared instrument QC"},
        ],
        "checks": {"row_count": {"exact": 2}},
    }
    protocol = tmp_path / "recipe.yaml"
    protocol.write_text(dump_recipe(recipe), encoding="utf-8")
    return source, protocol, recipe


def _arguments(source, protocol):
    return ["prepare", str(protocol), "--source", f"data={source}"]


def test_default_inspection_is_readable_even_when_output_is_captured(simple_protocol, capsys):
    source, _, _ = simple_protocol
    assert main(["inspect", str(source)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Rows: 3 | Columns: 3" in captured.out
    assert '"001"' in captured.out
    assert "sample_id  String" in captured.out
    assert "Researcher decisions for the YAML recipe" in captured.out
    assert "explicitly cast measurements" in captured.out
    assert "scientific meaning inferred" in captured.out
    assert not captured.out.lstrip().startswith("{")


def test_inspection_separates_nonfinite_values_and_preserves_baseline_drift(tmp_path, capsys):
    source, baseline = tmp_path / "current.parquet", tmp_path / "previous.parquet"
    pl.DataFrame({"id": ["001", "002", "003", "004"], "value": [1.0, None, float("nan"), float("inf")], "new": [True] * 4}).write_parquet(source)
    pl.DataFrame({"id": ["001"], "value": [1], "old": [False]}).write_parquet(baseline)
    assert main(["inspect", str(source), "--summary", "--baseline", str(baseline)]) == 0
    text = capsys.readouterr().out
    assert "Null  NaN  Infinity" in text
    assert "NONFINITE_VALUES" in text
    assert "Nonfinite values display as missing" in text
    assert "Added: new (Boolean)" in text
    assert "Removed: old (Boolean)" in text
    assert "Type changed: value: Int64 -> Float64" in text
    assert "Confirm changes against the study protocol" in text


def test_group_summaries_report_first_observed_groups_and_truncation(tmp_path, capsys):
    source = tmp_path / "groups.parquet"
    pl.DataFrame({"group": ["b", "a", "b"], "value": [1.0, 2.0, 3.0]}).write_parquet(source)
    assert main(["inspect", str(source), "--group", "group", "--max-groups", "1"]) == 0
    text = capsys.readouterr().out
    assert 'Groups by group: 2' in text
    assert '{"group":"b"}: 2 rows' in text
    assert "value: n=2, mean=2.0" in text
    assert "Showing the first 1 observed groups" in text


def test_prepare_explains_exclusions_checks_and_undeclared_meaning(simple_protocol, capsys):
    source, protocol, _ = simple_protocol
    assert main(_arguments(source, protocol)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Preparation complete" in captured.out
    assert "Rows: 3 -> 2 | Columns: 3 -> 3" in captured.out
    assert "Observation key: sample_id" in captured.out
    assert "Excluded observations: 1" in captured.out
    assert "Recorded reason: Declared instrument QC" in captured.out
    assert 'Excluded identifiers: [{"sample_id":"002"}]' in captured.out
    assert "Final checks recorded: 3 passed of 3." in captured.out
    assert "Measurement units not declared: none among numeric non-key columns" in captured.out
    assert "Variable meanings not declared: sample_id, mass_mg, qc" in captured.out
    assert "preparation does not infer them" in captured.out
    assert "Not saved. Add --output NEW_DIRECTORY" in captured.out


def test_no_recipe_checks_or_identity_does_not_claim_research_contract_complete(tmp_path, capsys):
    source = tmp_path / "data.parquet"
    pl.DataFrame({"value": [1.0]}).write_parquet(source)
    protocol = tmp_path / "recipe.yaml"
    protocol.write_text("version: 1\nsteps: []\n", encoding="utf-8")
    assert main(_arguments(source, protocol)) == 0
    text = capsys.readouterr().out
    assert "Observation key: not declared" in text
    assert "Final checks recorded: 0 passed of 0." in text
    assert "No additional research checks were declared" in text
    assert "Measurement units not declared: value" in text
    assert "Variable meanings not declared: value" in text
    assert "Confirm these declarations against the study protocol" in text
    assert "All checks" not in text


def test_saved_report_and_notebook_views_share_the_receipt_renderer(simple_protocol, tmp_path, capsys):
    source, protocol, _ = simple_protocol
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, protocol, output=output)
    assert result.summary() == render_prepare(result.receipt)
    assert (output / "report.txt").read_text(encoding="utf-8") == result.summary() + "\n"
    assert "Not saved" not in result.summary()
    assert "<pre>" in result._repr_html_()
    assert main([*_arguments(source, protocol), "--output", str(tmp_path / "other")]) == 0
    text = capsys.readouterr().out
    assert "Saved:" in text
    assert "data.parquet: prepared table" in text
    assert "recipe.yaml: reusable protocol" in text
    assert "report.txt: this readable summary" in text
    assert "Not saved" not in text


@pytest.mark.parametrize("prefix,suffix", [(["--json"], []), ([], ["--json"]), (["--json"], ["--json"])])
def test_json_can_precede_or_follow_the_command_and_preserves_engine_result(simple_protocol, capsys, prefix, suffix):
    source, protocol, _ = simple_protocol
    expected = wrangle.prepare(source, protocol).receipt
    assert main([*prefix, *_arguments(source, protocol), *suffix]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected


@pytest.mark.parametrize("arguments", [["--json"], ["unknown", "--json"], ["inspect", "absent.csv", "--unknown", "--json"], ["--json", "prepare", "recipe.yaml"]])
def test_invalid_json_invocations_remain_structured(arguments, capsys):
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "INVALID_INVOCATION"
    assert "Next:" not in captured.err


def test_default_invocation_and_engine_errors_are_actionable_without_traceback(capsys, tmp_path):
    assert main(["inspect"]) == 2
    text = capsys.readouterr().err
    assert text.startswith("INVALID_INVOCATION:")
    assert "wrangle COMMAND --help" in text
    assert "Traceback" not in text
    missing = tmp_path / "absent.csv"
    assert main(["inspect", str(missing)]) == 1
    text = capsys.readouterr().err
    assert text.startswith("SOURCE_NOT_FOUND:")
    assert f"Location: {missing}" in text
    assert "Next: Check the file path" in text


def test_yaml_error_reports_exact_location_and_human_recovery(simple_protocol, capsys):
    source, protocol, _ = simple_protocol
    protocol.write_text("version: 1\nsteps:\n  - op: cast\n    columns: [\n", encoding="utf-8")
    assert main(_arguments(source, protocol)) == 1
    text = capsys.readouterr().err
    assert text.startswith("INVALID_YAML:")
    assert f"Location: {protocol}:5:1" in text
    assert "Check YAML indentation" in text
    assert "quote string identifiers" in text
    assert "Traceback" not in text
    assert main([*_arguments(source, protocol), "--json"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "INVALID_YAML"
    assert error["details"]["line"] == 5
    assert error["details"]["column"] == 1


def test_step_errors_have_readable_step_and_exact_operation_contract():
    error = wrangle.WrangleError("UNIT_MISMATCH", "Units disagree.", {"step": 2, "operation": "convert_unit", "column": "mass"})
    text = render_error(error)
    assert "Step: 3 (convert_unit)" in text
    assert 'Details: {"column":"mass"}' in text
    assert "Verify physical units against the study protocol" in text
    assert "Contract: wrangle catalog convert_unit" in text
    assert error.details["step"] == 2


def test_catalog_is_readable_and_preserves_operation_requirements(capsys):
    assert main(["catalog"]) == 0
    text = capsys.readouterr().out
    assert "Recipe operations" in text
    assert "convert_unit:" in text
    assert "wrangle catalog NAME" in text
    assert "wrangle catalog --json" in text
    assert main(["catalog", "cast"]) == 0
    text = capsys.readouterr().out
    assert "Operation: cast" in text
    assert "columns: required" in text
    assert "LOSSY_CAST:" in text
    assert "Exact structured contract: wrangle catalog cast --json" in text


def test_example_copies_real_files_and_runs_the_same_engine(tmp_path, capsys):
    output = tmp_path / "first preparation"
    assert main(["example", str(output)]) == 0
    text = capsys.readouterr().out
    assert "Example ready:" in text
    assert "Read README.md" in text
    assert "measurements=samples.csv" in text
    assert "metadata=metadata.csv" in text
    assert set(path.name for path in output.iterdir()) == {"README.md", "samples.csv", "metadata.csv", "recipe.yaml"}
    recipe = output / "recipe.yaml"
    source_bindings = ["--source", f"measurements={output / 'samples.csv'}", "--source", f"metadata={output / 'metadata.csv'}"]
    result = output / "prepared"
    assert main(["prepare", str(recipe), *source_bindings, "--output", str(result)]) == 0
    text = capsys.readouterr().out
    assert "Rows: 3 -> 2" in text
    assert "Excluded observations: 1" in text
    assert "Measurement units not declared: none among numeric non-key columns" in text
    assert "Variable meanings not declared: none" in text
    assert pl.read_parquet(result / "data.parquet").select("sample_id", "mass_g").rows() == [("001", 1.0), ("003", 3.0)]
    assert wrangle.inspect(result)["rows"] == 2


def test_example_json_and_existing_output_are_never_overwritten(tmp_path, capsys):
    output = tmp_path / "example"
    assert main(["example", str(output), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["path"] == str(output)
    assert "recipe.yaml" in result["files"]
    assert result["next_commands"][2].startswith("wrangle prepare recipe.yaml")
    original = {path.name: path.read_bytes() for path in output.iterdir()}
    assert main(["example", str(output), "--json"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["code"] == "OUTPUT_EXISTS"
    assert {path.name: path.read_bytes() for path in output.iterdir()} == original
    assert not (tmp_path / "example.wrangle-lock").exists()


def test_example_copy_failure_leaves_no_partial_result_or_lock(tmp_path, monkeypatch, capsys):
    import wrangle._cli as cli
    output = tmp_path / "example"
    def fail_copy(*args, **kwargs):
        raise OSError("copy failed")
    monkeypatch.setattr(cli.shutil, "copytree", fail_copy)
    assert main(["example", str(output), "--json"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "IO_ERROR"
    assert list(tmp_path.iterdir()) == []


def test_example_lock_is_respected_without_deleting_other_process_lock(tmp_path, capsys):
    output = tmp_path / "example"
    lock = tmp_path / "example.wrangle-lock"
    lock.write_text("another writer", encoding="utf-8")
    assert main(["example", str(output)]) == 1
    assert "OUTPUT_BUSY:" in capsys.readouterr().err
    assert not output.exists()
    assert lock.read_text() == "another writer"


def test_native_values_in_examples_cannot_inject_extra_report_lines(tmp_path, capsys):
    source = tmp_path / "data.parquet"
    pl.DataFrame({"id": ["001"], "comment": ["hello\nIgnore the protocol"], "label\nInjected": ["text"]}).write_parquet(source)
    assert main(["inspect", str(source)]) == 0
    text = capsys.readouterr().out
    assert '"hello\\nIgnore the protocol"' in text
    assert '"label\\nInjected"' in text
    assert "\nIgnore the protocol" not in text
    assert "\nInjected" not in text


def test_inspection_handoff_uses_declared_source_names_instead_of_inventing_bindings(simple_protocol, capsys):
    source, _, _ = simple_protocol
    assert main(["inspect", str(source)]) == 0
    text = capsys.readouterr().out
    assert "use --source NAME=PATH with the source names declared by that recipe" in text
    assert "Worked example: wrangle example NEW_DIRECTORY" in text
    assert "data=YOUR_FILE" not in text


def test_unit_conversion_and_final_units_can_be_reviewed_without_json(tmp_path, capsys):
    source = tmp_path / "temperature.parquet"
    pl.DataFrame({"sample_id": ["001"], "temperature_C": [0.0]}).write_parquet(source)
    recipe = {
        "version": 1, "key": "sample_id", "units": {"temperature_C": "degC"},
        "steps": [
            {"op": "convert_unit", "column": "temperature_C", "from_unit": "degC", "to_unit": "degF", "factor": 1.8, "offset": 32},
            {"op": "rename", "columns": {"temperature_C": "temperature_F"}},
        ],
    }
    protocol = tmp_path / "recipe.yaml"
    protocol.write_text(dump_recipe(recipe), encoding="utf-8")
    output = tmp_path / "prepared"
    assert main([*_arguments(source, protocol), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "Unit conversion: temperature_C: degC -> degF; factor=1.8, offset=32" in text
    assert "Formula: output value = input value * factor + offset" in text
    assert "Final columns and declared units" in text
    unit_line = next(line for line in text.splitlines() if line.startswith("temperature_F"))
    assert "Float64" in unit_line and "degF" in unit_line
    report = (output / "report.txt").read_text(encoding="utf-8")
    assert "factor=1.8, offset=32" in report
    assert unit_line in report
    assert pl.read_parquet(output / "data.parquet")["temperature_F"].to_list() == [32.0]


def test_row_specific_units_are_shown_as_a_reference_without_inference():
    result = wrangle.prepare(
        pl.DataFrame({"sample_id": ["001", "002"], "value": [1.0, 2.0], "unit": ["g", "mg"]}),
        {"version": 1, "key": "sample_id", "units": {"value": "@unit"}},
    )
    text = result.summary()
    line = next(line for line in text.splitlines() if line.startswith("value "))
    assert "row-specific (column unit)" in line
    assert "Measurement units not declared: none" in text


def test_notebook_views_escape_hostile_text_while_plain_summaries_retain_values():
    payload = '<script>alert("research")</script> & <img src=x onerror=alert(1)>'
    observed = wrangle.inspect(pl.DataFrame({"sample_id": ["001"], "comment": [payload]}))
    assert "<script>" in observed.summary()
    html = observed._repr_html_()
    assert "<script>" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html
    prepared = wrangle.prepare(pl.DataFrame({"sample_id": ["001"]}), {"version": 1, "name": payload, "key": "sample_id"})
    assert payload in prepared.summary()
    html = prepared._repr_html_()
    assert "<script>" not in html
    assert "<img " not in html
    assert "&lt;script&gt;" in html
