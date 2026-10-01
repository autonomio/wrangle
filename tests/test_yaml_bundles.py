"""YAML authoring and readable evidence preserve the scientific execution contract."""
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

import wrangle
from wrangle._protocol import dump_recipe, load_recipe
from wrangle._storage import verify_report


@pytest.fixture
def protocol(tmp_path):
    source = tmp_path / "samples.csv"
    source.write_text("z_id,a_id,z_mass,a_mass,qc\n001,visit1,NA,1,pass\n002,visit2,2,NA,fail\n", encoding="utf-8")
    recipe = {
        "version": 1, "key": ["z_id", "a_id"],
        "units": {"z_mass": "g", "a_mass": "g"},
        "descriptions": {"z_id": "Sample identifier", "a_id": "Visit identifier", "z_mass": "First mass", "a_mass": "Second mass", "qc": "Instrument QC"},
        "steps": [
            {"op": "normalize_missing", "columns": {"z_mass": ["NA"], "a_mass": ["NA"]}},
            {"op": "cast", "columns": {"z_mass": "Float64", "a_mass": "Float64"}},
            {"op": "filter", "where": {"eq": [{"col": "qc"}, "pass"]}, "reason": "Predeclared QC rule"},
        ],
        "checks": {"required": ["z_id", "a_id", "qc"], "row_count": {"exact": 1}},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(dump_recipe(recipe), encoding="utf-8")
    return source, path, recipe


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_yaml_and_python_match_and_report_hash_uses_canonical_receipt_order(protocol, tmp_path, execution):
    source, path, recipe = protocol
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, path, execution=execution, output=output)
    direct = wrangle.prepare(source, recipe)
    assert result.receipt["recipe_sha256"] == direct.receipt["recipe_sha256"]
    assert result.receipt["output"] == direct.receipt["output"]
    assert load_recipe(output / "recipe.yaml") == recipe
    assert not (output / "recipe.json").exists()
    report = (output / "report.txt").read_bytes()
    assert hashlib.sha256(report).hexdigest() == result.receipt["report_sha256"]
    assert verify_report(output, result.receipt)
    assert report.decode("utf-8") == result.summary() + "\n"
    assert wrangle.prepare(output, {}).receipt["output"] == result.receipt["output"]
    assert pl.read_parquet(output / "data.parquet").select("z_id", "a_id").rows() == [("001", "visit1")]


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_report_bytes_ignore_platform_text_newline_translation(protocol, tmp_path, monkeypatch, execution):
    source, _, recipe = protocol
    recipe["steps"][-1]["reason"] = "Instrument µ-QC exclusion"
    native_write_text = Path.write_text

    def platform_text_write(path, text, *args, **kwargs):
        if path.name == "report.txt":
            # Emulate Windows text mode even when this test runs on POSIX.
            return path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        return native_write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", platform_text_write)
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, recipe, execution=execution, output=output)
    report = (output / "report.txt").read_bytes()
    assert "µ-QC".encode("utf-8") in report
    assert b"\r\n" not in report
    assert report == (result.summary() + "\n").encode("utf-8")
    assert hashlib.sha256(report).hexdigest() == result.receipt["report_sha256"]
    assert wrangle.prepare(output, {}).receipt["output"] == result.receipt["output"]


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_public_prepare_rejects_json_authoring_without_publication(protocol, tmp_path, execution):
    source, _, recipe = protocol
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe), encoding="utf-8")
    output = tmp_path / "failed"
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(source, path, execution=execution, output=output)
    assert caught.value.code == "RECIPE_FORMAT"
    assert not output.exists()
    assert not (tmp_path / "failed.wrangle-lock").exists()


@pytest.mark.parametrize("execution", ["memory", "disk"])
@pytest.mark.parametrize("change", ["edit", "delete"])
def test_saved_report_is_verified_before_reuse(protocol, tmp_path, execution, change):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, path, output=output)
    report = output / "report.txt"
    if change == "edit":
        report.write_text("Incorrect exclusion summary\n", encoding="utf-8")
    else:
        report.unlink()
    failed = tmp_path / "failed"
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(output, {}, execution=execution, output=failed)
    assert caught.value.code == "BUNDLE_MISMATCH"
    assert caught.value.details["path"] == "report.txt"
    assert not failed.exists()
    assert wrangle.prepare(source, path).receipt == result.receipt


@pytest.mark.parametrize("action", ["summary", "reuse", "write"])
def test_disk_result_rechecks_its_readable_report(protocol, tmp_path, action):
    source, path, _ = protocol
    result = wrangle.prepare(source, path, execution="disk", output=tmp_path / "prepared")
    (tmp_path / "prepared" / "report.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        if action == "summary":
            result.summary()
        elif action == "reuse":
            wrangle.prepare(result, {})
        else:
            result.write(tmp_path / "copied")
    assert caught.value.code == "BUNDLE_MISMATCH"
    assert not (tmp_path / "copied").exists()


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_legacy_json_bundle_is_readable_but_new_publications_use_yaml(protocol, tmp_path, execution):
    source, path, _ = protocol
    legacy = tmp_path / "legacy"
    parent = wrangle.prepare(source, path, output=legacy)
    (legacy / "recipe.yaml").unlink()
    (legacy / "recipe.json").write_text(json.dumps(parent.receipt["recipe"]), encoding="utf-8")
    (legacy / "report.txt").unlink()
    receipt = dict(parent.receipt)
    receipt.pop("report_sha256")
    (legacy / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "current"
    result = wrangle.prepare(legacy, {}, execution=execution, output=output)
    assert result.receipt["output"] == parent.receipt["output"]
    assert result.receipt["sources"]["data"]["parent"]["recipe_sha256"] == receipt["recipe_sha256"]
    assert (output / "recipe.yaml").is_file() and (output / "report.txt").is_file()
    assert not (output / "recipe.json").exists()


@pytest.mark.parametrize("additional", ["recipe.yml", "recipe.json"])
def test_saved_bundle_never_selects_between_multiple_recipe_files(protocol, tmp_path, additional):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, path, output=output)
    content = json.dumps(result.receipt["recipe"]) if additional.endswith("json") else dump_recipe(result.receipt["recipe"])
    (output / additional).write_text(content, encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.inspect(output)
    assert caught.value.code == "INVALID_BUNDLE"
    assert sorted(caught.value.details["recipes"]) == sorted(["recipe.yaml", additional])


@pytest.mark.parametrize("change", ["data", "receipt", "nonfinite_receipt"])
def test_summary_refuses_changed_memory_result(protocol, change):
    source, path, _ = protocol
    result = wrangle.prepare(source, path)
    if change == "data":
        result.data.replace_column(0, pl.Series("z_id", ["new"]))
    elif change == "receipt":
        result.receipt["units"]["z_mass"] = "kg"
    else:
        result.receipt["unexpected"] = float("nan")
    with pytest.raises(wrangle.WrangleError) as caught:
        result.summary()
    assert caught.value.code == "RESULT_CHANGED"


def test_report_change_during_bundle_scan_is_detected(protocol, tmp_path, monkeypatch):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    wrangle.prepare(source, path, output=output)
    original = pl.read_parquet
    def changed_scan(*args, **kwargs):
        data = original(*args, **kwargs)
        (output / "report.txt").write_text("changed during scan", encoding="utf-8")
        return data
    monkeypatch.setattr(pl, "read_parquet", changed_scan)
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.inspect(output)
    assert caught.value.code == "SOURCE_CHANGED"


@pytest.mark.parametrize("value", ["true", "1.0"])
@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_saved_recipe_identity_distinguishes_booleans_integers_and_floats(protocol, tmp_path, value, execution):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    wrangle.prepare(source, path, output=output)
    saved = output / "recipe.yaml"
    saved.write_text(saved.read_text().replace("version: 1\n", f"version: {value}\n"), encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(output, {}, execution=execution, output=tmp_path / "failed")
    assert caught.value.code == "BUNDLE_MISMATCH"
    assert not (tmp_path / "failed").exists()


def test_saved_yaml_error_names_original_file_after_disk_snapshot_cleanup(protocol, tmp_path):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    wrangle.prepare(source, path, output=output)
    saved = output / "recipe.yaml"
    saved.write_text("steps: [\n", encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(output, {}, execution="disk", output=tmp_path / "failed")
    assert caught.value.code == "INVALID_BUNDLE"
    assert caught.value.details["recipe_code"] == "INVALID_YAML"
    assert caught.value.details["path"] == str(saved)
    assert caught.value.details["line"] == 2


@pytest.mark.parametrize("metadata", ["recipe.yaml", "receipt.json"])
@pytest.mark.parametrize("action", ["summary", "reuse", "write"])
def test_disk_result_rechecks_saved_protocol_and_receipt(protocol, tmp_path, metadata, action):
    source, path, _ = protocol
    output = tmp_path / "prepared"
    result = wrangle.prepare(source, path, execution="disk", output=output)
    target = output / metadata
    if metadata.endswith("yaml"):
        target.write_text(target.read_text().replace("version: 1\n", "version: true\n"), encoding="utf-8")
    else:
        receipt = json.loads(target.read_text())
        receipt["units"]["z_mass"] = "kg"
        target.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        if action == "summary":
            result.summary()
        elif action == "reuse":
            wrangle.prepare(result, {})
        else:
            result.write(tmp_path / "copied")
    assert caught.value.code == "RESULT_CHANGED"
    assert not (tmp_path / "copied").exists()
