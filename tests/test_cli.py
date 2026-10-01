"""The shell surface preserves Python engine semantics and publication evidence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle
from wrangle._catalog import catalog, operation_document
from wrangle._cli import main
from wrangle._protocol import dump_recipe, load_recipe


@pytest.fixture
def research_files(tmp_path):
    measurements = tmp_path / "raw samples=run.csv"
    measurements.write_text("sample_id,mass_mg,qc\n001,1000,pass\n002,2000,fail\n003,3000,pass\n", encoding="utf-8")
    metadata = tmp_path / "sample metadata.csv"
    metadata.write_text("sample_id,group\n003,treated\n001,control\n002,control\n", encoding="utf-8")
    recipe = {
        "version": 1, "input": "measurements", "key": "sample_id", "units": {"mass_mg": "mg"},
        "steps": [
            {"op": "cast", "columns": {"mass_mg": "Float64"}},
            {"op": "join", "source": "metadata", "on": "sample_id"},
            {"op": "filter", "where": {"eq": [{"col": "qc"}, "pass"]}, "reason": "Instrument quality control"},
            {"op": "convert_unit", "column": "mass_mg", "from_unit": "mg", "to_unit": "g", "factor": 0.001},
            {"op": "rename", "columns": {"mass_mg": "mass_g"}},
        ],
        "checks": {"required": ["group", "mass_g"], "row_count": {"exact": 2}},
    }
    protocol = tmp_path / "research protocol.yaml"
    protocol.write_text(dump_recipe(recipe), encoding="utf-8")
    return {"measurements": measurements, "metadata": metadata}, protocol, recipe


def _prepare_args(sources, protocol):
    arguments = ["--json", "prepare", str(protocol)]
    for name, path in sources.items():
        arguments.extend(["--source", f"{name}={path}"])
    return arguments


def test_inspect_matches_python_and_preserves_identifiers(research_files, capsys):
    sources, _, _ = research_files
    source = sources["measurements"]
    assert main(["--json", "inspect", str(source), "--sample-rows", "1"]) == 0
    output = capsys.readouterr()
    assert not output.err
    profile = json.loads(output.out)
    assert profile == wrangle.inspect(source, sample_rows=1)
    assert profile["examples"][0]["sample_id"] == "001"


def test_catalog_matches_python_and_retains_shared_contracts(capsys):
    expected = catalog()
    assert main(["--json", "catalog"]) == 0
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == expected
    assert main(["--json", "catalog", "cast"]) == 0
    filtered = json.loads(capsys.readouterr().out)
    assert filtered == operation_document("cast", expected)
    assert filtered["operations"][0]["name"] == "cast"
    assert filtered["shared_validation"] == {"$ref": "docs/operations.json#/shared_validation"}
    assert filtered["boundary_validation"]


def test_prepare_matches_python_data_receipt_and_evidence(research_files, tmp_path, capsys):
    sources, protocol, recipe = research_files
    original = {path: path.read_bytes() for path in [*sources.values(), protocol]}
    expected = wrangle.prepare(sources, protocol)
    output_path = tmp_path / "prepared batch"
    assert main(["--json", *_prepare_args(sources, protocol), "--output", str(output_path)]) == 0
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == expected.receipt
    assert json.loads((output_path / "receipt.json").read_text()) == expected.receipt
    assert load_recipe(output_path / "recipe.yaml") == recipe
    assert_frame_equal(pl.read_parquet(output_path / "data.parquet"), expected.data)
    assert expected.data["sample_id"].to_list() == ["001", "003"]
    assert expected.data["mass_g"].to_list() == [1.0, 3.0]
    assert {path: path.read_bytes() for path in original} == original


def test_prepare_without_output_runs_checks_and_does_not_publish(research_files, tmp_path, capsys):
    sources, protocol, _ = research_files
    before = set(tmp_path.rglob("*"))
    assert main(_prepare_args(sources, protocol)) == 0
    output = capsys.readouterr()
    assert not output.err
    assert json.loads(output.out) == wrangle.prepare(sources, protocol).receipt
    assert set(tmp_path.rglob("*")) == before


def test_engine_failure_preserves_code_details_and_prevents_publication(research_files, tmp_path, capsys):
    sources, protocol, recipe = research_files
    recipe["checks"]["row_count"] = {"exact": 3}
    protocol.write_text(dump_recipe(recipe), encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as expected:
        wrangle.prepare(sources, protocol)
    destination = tmp_path / "must not publish"
    assert main(["--json", *_prepare_args(sources, protocol), "--output", str(destination)]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == expected.value.to_dict()
    assert not destination.exists()
    assert main(_prepare_args(sources, protocol)) == 1
    assert json.loads(capsys.readouterr().err) == expected.value.to_dict()


def test_existing_output_is_never_overwritten(research_files, tmp_path, capsys):
    sources, protocol, _ = research_files
    destination = tmp_path / "existing result"
    arguments = [*_prepare_args(sources, protocol), "--output", str(destination)]
    assert main(["--json", *arguments]) == 0
    capsys.readouterr()
    original = {path.name: path.read_bytes() for path in destination.iterdir()}
    assert main(["--json", *arguments]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["code"] == "OUTPUT_EXISTS"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original
    assert not (tmp_path / "existing result.wrangle-lock").exists()


@pytest.mark.parametrize("arguments", [
    [], ["unknown"], ["inspect"], ["prepare", "recipe.yaml"],
    ["inspect", "data.csv", "--unknown"], ["inspect", "data.csv", "--sample-rows", "many"],
    ["prepare", "recipe.yaml", "--source", "data"],
    ["prepare", "recipe.yaml", "--source", "=data.csv"],
    ["prepare", "recipe.yaml", "--source", "data="],
    ["prepare", "recipe.yaml", "--source", "data=a.csv", "--source", "data=b.csv"],
    ["prepare", "recipe.yaml", "--source", "data=a.csv", "--dry-run"],
    ["prepare", "recipe.yaml", "--source", "data=a.csv", "--out", "batch"],
])
def test_invalid_invocations_are_json_and_status_2(arguments, capsys):
    assert main(["--json", *arguments]) == 2
    output = capsys.readouterr()
    assert not output.out
    error = json.loads(output.err)
    assert error["code"] == "INVALID_INVOCATION"
    assert error["message"]
    assert isinstance(error["details"], dict)


def test_missing_source_unknown_operation_and_invalid_sample_count(tmp_path, capsys):
    assert main(["--json", "inspect", str(tmp_path / "absent.csv")]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "SOURCE_NOT_FOUND"
    assert main(["--json", "catalog", "import_anything"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "UNKNOWN_OPERATION"
    assert error["details"] == {"operation": "import_anything"}
    assert main(["--json", "inspect", str(tmp_path / "absent.csv"), "--sample-rows", "101"]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "INVALID_ARGUMENT"


def test_output_io_failure_is_structured(research_files, tmp_path, capsys):
    sources, protocol, _ = research_files
    blocked = tmp_path / "file is not a directory"
    blocked.write_text("preserve", encoding="utf-8")
    assert main(["--json", *_prepare_args(sources, protocol), "--output", str(blocked / "batch")]) == 1
    output = capsys.readouterr()
    assert not output.out
    error = json.loads(output.err)
    assert error["code"] == "IO_ERROR"
    assert error["details"]["path"] == str(blocked)
    assert blocked.read_text() == "preserve"


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_help_and_version_are_human_readable(flag, capsys):
    with pytest.raises(SystemExit) as result:
        main([flag])
    assert result.value.code == 0
    output = capsys.readouterr()
    assert "wrangle" in output.out
    assert not output.err


def test_module_and_installed_console_entrypoints(research_files, tmp_path):
    sources, protocol, _ = research_files
    module = subprocess.run([sys.executable, "-m", "wrangle", *_prepare_args(sources, protocol)], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert module.returncode == 0, module.stderr
    assert not module.stderr
    assert json.loads(module.stdout) == wrangle.prepare(sources, protocol).receipt
    executable = shutil.which("wrangle", path=str(Path(sys.executable).parent))
    assert executable is not None, "Install the package to verify its console entrypoint."
    console = subprocess.run([executable, "--json", "inspect", str(sources["measurements"])], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert console.returncode == 0, console.stderr
    assert not console.stderr
    assert json.loads(console.stdout) == wrangle.inspect(sources["measurements"])


def test_inspect_summary_group_and_baseline_controls_match_python(tmp_path, capsys):
    data_path = tmp_path / "current batch.parquet"
    baseline_path = tmp_path / "prior batch.parquet"
    current = pl.DataFrame({"id": ["001", "002", "003"], "group": ["b", "a", "b"], "batch": [2, 1, 2], "value": [1.0, None, 3.0]})
    current.write_parquet(data_path)
    pl.DataFrame({"id": ["001"], "value": [1], "old": [True]}).write_parquet(baseline_path)
    before = {path: path.read_bytes() for path in (data_path, baseline_path)}
    arguments = ["--json", "inspect", str(data_path), "--summary", "--group", "group", "--group", "batch", "--max-groups", "1", "--baseline", str(baseline_path), "--sample-rows", "1"]
    assert main(["--json", *arguments]) == 0
    output = capsys.readouterr()
    assert not output.err
    expected = wrangle.inspect(data_path, summary=True, groups=["group", "batch"], max_groups=1, baseline=baseline_path, sample_rows=1)
    assert json.loads(output.out) == expected
    assert expected["groups"]["count"] == 2
    assert expected["groups"]["truncated"] is True
    assert expected["groups"]["items"][0]["values"] == {"group": "b", "batch": 2}
    assert expected["schema_changes"]["type_changed"] == [{"name": "value", "before": "Int64", "after": "Float64"}]
    assert {path: path.read_bytes() for path in before} == before
    executable = shutil.which("wrangle", path=str(Path(sys.executable).parent))
    assert executable is not None
    console = subprocess.run([executable, *arguments], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert console.returncode == 0, console.stderr
    assert not console.stderr
    assert json.loads(console.stdout) == expected


def test_inspect_summary_does_not_infer_measurements_from_csv_strings(research_files, capsys):
    sources, _, _ = research_files
    assert main(["--json", "inspect", str(sources["measurements"]), "--summary"]) == 0
    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    mass = next(field for field in result["fields"] if field["name"] == "mass_mg")
    assert mass["dtype"] == "String"
    assert mass["summary"]["reason"] == "NON_NUMERIC"
    assert result["examples"][0]["sample_id"] == "001"


@pytest.mark.parametrize("controls,status,code", [
    (["--max-groups", "101"], 1, "INVALID_ARGUMENT"),
    (["--group", "group", "--group", "group"], 1, "INVALID_ARGUMENT"),
    (["--group", "absent"], 1, "UNKNOWN_COLUMN"),
    (["--max-groups", "many"], 2, "INVALID_INVOCATION"),
    (["--max-g", "1"], 2, "INVALID_INVOCATION"),
])
def test_inspect_profile_controls_have_stable_error_status(tmp_path, capsys, controls, status, code):
    path = tmp_path / "batch.parquet"
    pl.DataFrame({"group": ["a"], "value": [1.0]}).write_parquet(path)
    assert main(["--json", "inspect", str(path), *controls]) == status
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["code"] == code


def test_inspect_repeatable_columns_project_examples_and_fields_only(tmp_path, capsys):
    path = tmp_path / "batch.parquet"
    data = pl.DataFrame({"id": ["001"], "value": [1.0], "other": [True]})
    data.write_parquet(path)
    assert main(["--json", "inspect", str(path), "--column", "value", "--column", "id", "--summary"]) == 0
    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    assert result == wrangle.inspect(path, columns=["value", "id"], summary=True)
    assert [field["name"] for field in result["fields"]] == ["value", "id"]
    assert result["examples"] == [{"value": 1.0, "id": "001"}]
    assert result["columns"] == {"id": "String", "value": "Float64", "other": "Boolean"}
