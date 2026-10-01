"""Cross-operation regressions for scientific identity, units and captured state."""
from copy import deepcopy
import functools
import hashlib
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle as wr


def _fails(data, recipe, code):
    original = data.clone()
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, recipe)
    assert caught.value.code == code
    assert_frame_equal(data, original)


@pytest.mark.parametrize("step", [
    {"op": "recode", "column": "unit", "mapping": {"mg": "g"}, "dtype": "String"},
    {"op": "derive", "columns": {"unit": "g"}, "overwrite": True},
])
def test_relabeling_dynamic_unit_field_does_not_reinterpret_unchanged_measurements(step):
    data = pl.DataFrame({"id": ["a", "b"], "value": [1000.0, 2000.0], "unit": ["mg", "mg"]})
    _fails(data, {"key": "id", "units": {"value": "@unit"}, "steps": [step]}, "UNIT_MISMATCH")


@pytest.mark.parametrize("op,controls", [("standardize", {"ddof": 1}), ("impute", {"method": "mean"})])
def test_statistics_do_not_pool_measurements_with_different_row_specific_units(op, controls):
    data = pl.DataFrame({"id": ["a", "b", "c"], "value": [1000.0, 1.0, None], "unit": ["mg", "g", "mg"]})
    _fails(data, {"key": "id", "units": {"value": "@unit"}, "steps": [{"op": op, "columns": ["value"], **controls}]}, "UNIT_MISMATCH")


@pytest.mark.parametrize("op,controls", [("standardize", {"ddof": 1}), ("impute", {"method": "mean"})])
def test_frozen_statistics_require_the_same_physical_measurement_unit(op, controls):
    fitted = wr.prepare(pl.DataFrame({"id": ["a", "b"], "value": [1000.0, 2000.0]}), {"key": "id", "units": {"value": "mg"}, "steps": [{"op": op, "columns": ["value"], **controls}]})
    parameters = fitted.receipt["steps"][0]["parameters"]["parameters"]
    target = pl.DataFrame({"id": ["c", "d"], "value": [1.0, None]})
    _fails(target, {"key": "id", "units": {"value": "g"}, "steps": [{"op": op, "columns": ["value"], **controls, "parameters": parameters}]}, "UNIT_MISMATCH")


@pytest.mark.parametrize("op,controls", [("standardize", {"ddof": 1}), ("impute", {"method": "mean"})])
def test_frozen_statistics_replay_without_refitting_or_changing_inputs(op, controls):
    training = pl.DataFrame({"id": ["a", "b"], "value": [1.0, 3.0]})
    original = training.clone()
    fitted = wr.prepare(training, {"key": "id", "units": {"value": "mg"}, "steps": [{"op": op, "columns": ["value"], **controls}]})
    parameters = fitted.receipt["steps"][0]["parameters"]["parameters"]
    target = pl.DataFrame({"id": ["c", "d"], "value": [10.0, None]})
    target_original = target.clone()
    recipe = {"key": "id", "units": {"value": "mg"}, "steps": [{"op": op, "columns": ["value"], **controls, "parameters": deepcopy(parameters)}]}
    supplied = deepcopy(recipe)
    result = wr.prepare(target, recipe)
    replay = wr.prepare(target.rechunk(), recipe)
    assert result.receipt == replay.receipt
    assert result.receipt["steps"][0]["parameters"]["parameters"] == parameters
    assert recipe == supplied
    assert result.data["value"].to_list() == ([pytest.approx(8 / 2**0.5), None] if op == "standardize" else [10.0, 2.0])
    assert_frame_equal(training, original)
    assert_frame_equal(target, target_original)


def test_named_file_sources_are_captured_once_before_steps_and_joins(tmp_path, monkeypatch):
    import wrangle._api as api
    left_path, right_path = tmp_path / "observations.parquet", tmp_path / "metadata.parquet"
    left = pl.DataFrame({"id": ["a", "b"], "value": [1.0, 2.0]})
    right = pl.DataFrame({"id": ["a", "b"], "label": ["original a", "original b"]})
    left.write_parquet(left_path)
    right.write_parquet(right_path)
    captured_files = {"left": left_path.read_bytes(), "right": right_path.read_bytes()}
    original = api._SIMPLE["derive"]
    @functools.wraps(original)
    def alter_files_after_capture(*args, **kwargs):
        pl.DataFrame({"id": ["z"], "value": [999.0]}).write_parquet(left_path)
        pl.DataFrame({"id": ["a", "b"], "label": ["changed a", "changed b"]}).write_parquet(right_path)
        return original(*args, **kwargs)
    monkeypatch.setitem(api._SIMPLE, "derive", alter_files_after_capture)
    result = wr.prepare({"left": left_path, "right": right_path}, {"input": "left", "key": "id", "steps": [{"op": "derive", "columns": {"marker": 1}}, {"op": "join", "source": "right", "on": "id"}]})
    assert result.data["id"].to_list() == ["a", "b"]
    assert result.data["value"].to_list() == [1.0, 2.0]
    assert result.data["label"].to_list() == ["original a", "original b"]
    for name, encoded in captured_files.items():
        assert result.receipt["sources"][name]["sha256"] == hashlib.sha256(encoded).hexdigest()
    assert result.receipt["steps"][0]["input_sha256"] == result.receipt["sources"]["left"]["snapshot_sha256"]


def test_full_join_with_explicit_union_identity_retains_both_source_relations():
    left = pl.DataFrame({"left_id": ["b", "a"], "value": [2.0, 1.0]})
    right = pl.DataFrame({"right_id": ["c", "b"], "value": [30.0, 20.0]})
    originals = left.clone(), right.clone()
    recipe = {"input": "left", "key": "left_id", "units": {"value": "mg"}, "source_contracts": {"right": {"key": "right_id", "units": {"value": "g"}}}, "steps": [{"op": "join", "source": "right", "left_on": "left_id", "right_on": "right_id", "coalesce": True, "how": "full", "maintain_order": "left_right", "unmatched": {"left": "keep", "right": "keep"}, "allow_expand": True, "key": "left_id"}]}
    result = wr.prepare({"left": left, "right": right}, recipe)
    assert result.data["left_id"].to_list() == ["b", "a", "c"]
    assert result.data["value"].to_list() == [2.0, 1.0, None]
    assert result.data["value_right"].to_list() == [20.0, None, 30.0]
    assert result.receipt["units"] == {"value": "mg", "value_right": "g"}
    transition = result.receipt["steps"][0]["observation_transition"]
    assert transition["input_key"] == transition["output_key"] == ["left_id"]
    assert transition["introduced_source"] == "right"
    assert transition["introduced_rows"] == 1
    assert json.loads(json.dumps(result.receipt)) == result.receipt
    assert result.receipt == wr.prepare({"left": left, "right": right}, recipe).receipt
    assert_frame_equal(left, originals[0])
    assert_frame_equal(right, originals[1])


def test_right_join_can_adopt_right_identity_and_records_lost_left_observations():
    left = pl.DataFrame({"left_id": ["b", "a"], "value": [2.0, 1.0]})
    right = pl.DataFrame({"right_id": ["c", "b"], "value": [30.0, 20.0]})
    recipe = {"input": "left", "key": "left_id", "source_contracts": {"right": {"key": "right_id"}}, "steps": [{"op": "join", "source": "right", "left_on": "left_id", "right_on": "right_id", "how": "right", "maintain_order": "right_left", "unmatched": {"left": "drop", "right": "keep"}, "allow_expand": True, "key": "right_id", "reason": "Retain authoritative registry observations"}]}
    result = wr.prepare({"left": left, "right": right}, recipe)
    assert result.data["right_id"].to_list() == ["c", "b"]
    assert result.data["left_id"].to_list() == [None, "b"]
    step = result.receipt["steps"][0]
    assert step["excluded_keys"] == [{"left_id": "a"}]
    assert step["observation_counts"]["excluded_observations"] == 1
    assert step["observation_transition"]["introduced_source"] == "right"
    assert step["observation_transition"]["introduced_rows"] == 1
    assert json.loads(json.dumps(result.receipt)) == result.receipt


def test_different_join_keys_keep_right_descriptions_and_dynamic_unit_pointer():
    left = pl.DataFrame({"id": ["a", "b"], "unit": ["left annotation", "left annotation"]})
    right = pl.DataFrame({"right_id": ["b", "a"], "unit": ["g", "mg"], "value": [2.0, 1000.0]})
    recipe = {"input": "left", "key": "id", "source_contracts": {"right": {"key": "right_id", "units": {"value": "@unit"}, "descriptions": {"right_id": "Registry specimen identifier", "value": "Measured mass"}}}, "steps": [{"op": "join", "source": "right", "left_on": "id", "right_on": "right_id"}]}
    result = wr.prepare({"left": left, "right": right}, recipe)
    assert result.data["id"].to_list() == ["a", "b"]
    assert result.data["right_id"].to_list() == ["a", "b"]
    assert result.data["unit_right"].to_list() == ["mg", "g"]
    assert result.receipt["units"]["value"] == "@unit_right"
    assert result.receipt["variables"]["right_id"]["description"] == "Registry specimen identifier"
    assert result.receipt["variables"]["value"]["description"] == "Measured mass"


def test_right_coalesced_different_key_preserves_coverage_evidence_after_left_axis_disappears():
    left = pl.DataFrame({"left_id": ["b", "a"], "value": [2.0, 1.0]})
    right = pl.DataFrame({"right_id": ["c", "b"], "metadata": [30, 20]})
    recipe = {"input": "left", "key": "left_id", "steps": [{"op": "join", "source": "right", "left_on": "left_id", "right_on": "right_id", "coalesce": True, "how": "right", "maintain_order": "right_left", "unmatched": {"left": "drop", "right": "keep"}, "key": "right_id", "reason": "Retain authoritative registry observations", "allow_expand": True}]}
    result = wr.prepare({"left": left, "right": right}, recipe)
    assert result.data["right_id"].to_list() == ["c", "b"]
    assert "left_id" not in result.data.columns
    step = result.receipt["steps"][0]
    assert step["excluded_keys"] == [{"left_id": "a"}]
    assert step["observation_counts"]["excluded_observations"] == 1
    assert step["observation_transition"]["introduced_source"] == "right"
    assert step["observation_transition"]["introduced_rows"] == 1
    assert json.loads(json.dumps(result.receipt)) == result.receipt


def test_right_coalesced_unit_axis_remaps_left_measurement_reference_to_surviving_field():
    left = pl.DataFrame({"id": ["a"], "unit_left": ["mg"], "value": [1000.0]})
    right = pl.DataFrame({"registry_id": ["x", "y"], "unit_right": ["mg", "g"]})
    originals = left.clone(), right.clone()
    recipe = {"input": "left", "key": "id", "units": {"value": "@unit_left"}, "source_contracts": {"right": {"key": "registry_id"}}, "steps": [{"op": "join", "source": "right", "left_on": "unit_left", "right_on": "unit_right", "coalesce": True, "how": "right", "maintain_order": "right_left", "unmatched": {"left": "drop", "right": "keep"}, "key": "registry_id", "reason": "Retain authoritative registry observations", "allow_expand": True}]}
    result = wr.prepare({"left": left, "right": right}, recipe)
    assert result.data["registry_id"].to_list() == ["x", "y"]
    assert result.data["value"].to_list() == [1000.0, None]
    assert result.data["unit_right"].to_list() == ["mg", "g"]
    assert result.receipt["units"] == {"value": "@unit_right"}
    assert "unit_left" not in result.data.columns
    assert_frame_equal(left, originals[0])
    assert_frame_equal(right, originals[1])
