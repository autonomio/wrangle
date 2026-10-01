"""Observed profiles preserve data and expose unresolved scientific boundaries."""
from decimal import Decimal
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle
from wrangle._profile import profile, validate_options


def _field(result, name):
    return next(field for field in result["fields"] if field["name"] == name)


def test_default_profile_retains_source_parent_counts_and_bounded_examples():
    data = pl.DataFrame({"id": ["001", "002", "003"], "value": [1.0, None, 3.0]})
    source = {"rows": 3, "columns": {"id": "String", "value": "Float64"}, "parent": {"recipe_sha256": "parent", "units": {"value": "mg"}, "descriptions": {"value": "measured mass"}}}
    original = data.clone()
    result = profile(data, source, sample_rows=1)
    assert result["parent"] == source["parent"]
    assert result["columns"] == source["columns"]
    assert result["examples"] == [{"id": "001", "value": 1.0}]
    assert _field(result, "value") == {"name": "value", "dtype": "Float64", "nulls": 1, "nans": 0, "infinities": 0, "distinct": 3}
    assert "summary_policy" not in result
    assert_frame_equal(data, original)


def test_numeric_summary_uses_non_null_denominator_and_sample_standard_deviation():
    data = pl.DataFrame({"value": [1.0, None, 3.0], "label": ["a", "b", "c"]})
    result = profile(data, {}, summary=True, sample_rows=0)
    summary = _field(result, "value")["summary"]
    assert summary == {"status": "resolved", "count": 2, "min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0, "std": pytest.approx(2**0.5)}
    assert result["summary_policy"]["ddof"] == 1
    assert result["summary_policy"]["denominator"] == "non-null observations"
    assert result["examples"] == []
    assert _field(result, "label")["summary"]["reason"] == "NON_NUMERIC"


@pytest.mark.parametrize("values", [[], [None, None]])
def test_absent_numeric_observations_have_zero_count_and_undefined_statistics(values):
    data = pl.DataFrame({"value": pl.Series(values, dtype=pl.Float64)})
    summary = _field(profile(data, {}, summary=True), "value")["summary"]
    assert summary == {"status": "unresolved", "reason": "NO_OBSERVATIONS", "count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}


def test_one_observation_defines_location_but_not_sample_standard_deviation():
    summary = _field(profile(pl.DataFrame({"value": [4, None]}), {}, summary=True), "value")["summary"]
    assert summary == {"status": "resolved", "count": 1, "min": 4, "max": 4, "mean": 4.0, "median": 4.0, "std": None}


def test_nonfinite_observations_are_counted_and_never_removed_from_summaries():
    data = pl.DataFrame({"value": [1.0, float("nan"), None, float("inf"), float("-inf")]})
    original = data.clone()
    field = _field(profile(data, {}, summary=True), "value")
    assert (field["nulls"], field["nans"], field["infinities"]) == (1, 1, 2)
    assert field["summary"] == {"status": "unresolved", "count": 4, "reason": "NONFINITE_VALUES", "min": None, "max": None, "mean": None, "median": None, "std": None}
    assert_frame_equal(data, original)


@pytest.mark.parametrize("dtype,values,nans,infinities", [
    (pl.List(pl.Float64), [[float("nan"), float("inf")], None, [1.0, float("-inf")]], 1, 2),
    (pl.Array(pl.Float64, 2), [[float("nan"), float("inf")], None, [1.0, float("-inf")]], 1, 2),
    (pl.Struct({"value": pl.Float64}), [{"value": float("nan")}, None, {"value": float("inf")}], 1, 1),
])
def test_nested_float_observations_are_counted_instead_of_hidden(dtype, values, nans, infinities):
    data = pl.DataFrame({"nested": pl.Series(values, dtype=dtype)})
    field = _field(profile(data, {}, summary=True), "nested")
    assert field["nulls"] == 1
    assert field["nans"] == nans
    assert field["infinities"] == infinities
    assert field["summary"]["reason"] == "NONFINITE_VALUES"
    assert field["summary"]["count"] == 2


@pytest.mark.parametrize("dtype,values", [
    (pl.Int64, [2**53 + 1, 2**53 + 3]),
    (pl.UInt64, [2**64 - 1]),
    (pl.Decimal(38, 3), [Decimal("9007199254740993.125")]),
    (pl.Decimal(38, 20), [Decimal("0.12345678901234567890")]),
])
def test_summary_does_not_round_integer_or_decimal_measurements(dtype, values):
    data = pl.DataFrame({"value": pl.Series(values, dtype=dtype)})
    original = data.clone()
    field = _field(profile(data, {}, summary=True), "value")
    assert field["summary"]["status"] == "unresolved"
    assert field["summary"]["reason"] == "LOSSY_FLOAT64"
    assert field["summary"]["mean"] is None
    assert_frame_equal(data, original)


def test_exact_decimal_observations_keep_exact_extrema_and_defined_summary():
    data = pl.DataFrame({"value": pl.Series([Decimal("1.25"), None, Decimal("2.50")], dtype=pl.Decimal(38, 2))})
    summary = _field(profile(data, {}, summary=True), "value")["summary"]
    assert summary["status"] == "resolved"
    assert summary["min"] == "1.25"
    assert summary["max"] == "2.50"
    assert summary["mean"] == 1.875
    assert summary["std"] == pytest.approx((1.25**2 / 2)**0.5)


def test_finite_observations_with_overflowing_statistics_remain_unresolved():
    data = pl.DataFrame({"value": [1e308, -1e308]})
    field = _field(profile(data, {}, summary=True), "value")
    assert field["nans"] == field["infinities"] == 0
    assert field["summary"]["reason"] == "NONFINITE_STATISTIC"
    assert field["summary"]["std"] is None
    json.dumps(field, allow_nan=False)


def test_group_summaries_use_first_appearance_order_and_report_truncation():
    data = pl.DataFrame({"group": ["b", "a", "b", None, "c"], "value": [1.0, None, 3.0, 4.0, 5.0]})
    original = data.clone()
    result = profile(data, {}, columns=["value"], groups=["group"], max_groups=2)
    groups = result["groups"]
    assert groups["count"] == 4
    assert groups["max_groups"] == 2
    assert groups["truncated"] is True
    assert [item["values"] for item in groups["items"]] == [{"group": "b"}, {"group": "a"}]
    assert groups["items"][0]["rows"] == 2
    assert groups["items"][0]["fields"][0]["summary"]["mean"] == 2.0
    assert groups["items"][1]["fields"][0]["summary"]["reason"] == "NO_OBSERVATIONS"
    assert list(result["examples"][0]) == ["value"]
    assert_frame_equal(data, original)


def test_composite_groups_include_null_keys_without_dropping_observations():
    data = pl.DataFrame({"subject": ["b", "a", "b", None], "batch": [2, 1, 2, 1], "value": [1, 2, 3, 4]})
    groups = profile(data, {}, groups=["subject", "batch"])["groups"]
    assert groups["count"] == 3
    assert groups["truncated"] is False
    assert [item["values"] for item in groups["items"]] == [{"subject": "b", "batch": 2}, {"subject": "a", "batch": 1}, {"subject": None, "batch": 1}]
    assert sum(item["rows"] for item in groups["items"]) == data.height


def test_nonfinite_values_only_leave_their_observed_group_unresolved():
    data = pl.DataFrame({"group": ["a", "b", "a"], "value": [1.0, 2.0, float("nan")]})
    items = profile(data, {}, groups=["group"])["groups"]["items"]
    assert items[0]["fields"][1]["summary"]["reason"] == "NONFINITE_VALUES"
    assert items[0]["rows"] == 2
    assert items[1]["fields"][1]["summary"]["mean"] == 2.0


def test_nonfinite_group_keys_are_not_conflated_with_json_null_labels():
    data = pl.DataFrame({"group": [None, float("nan"), float("inf"), float("-inf")], "value": [1, 2, 3, 4]})
    groups = profile(data, {}, groups=["group"])["groups"]
    assert groups["count"] == 4
    assert groups["items"] == []
    assert groups["status"] == "unresolved"
    assert groups["reason"] == "NONFINITE_GROUP_KEY"


def test_schema_comparison_reports_complete_drift_without_casting_or_projection():
    data = pl.DataFrame({"id": ["001"], "value": [1.5], "extra": [True]})
    original = data.clone()
    result = profile(data, {}, columns=["value"], baseline_columns={"id": "String", "value": "Int64", "old": "Date"})
    assert result["schema_changes"] == {"added": [{"name": "extra", "dtype": "Boolean"}], "removed": [{"name": "old", "dtype": "Date"}], "type_changed": [{"name": "value", "before": "Int64", "after": "Float64"}]}
    assert_frame_equal(data, original)


@pytest.mark.parametrize("options", [
    {"sample_rows": -1}, {"sample_rows": 101}, {"sample_rows": True},
    {"summary": 1}, {"summary": "true"},
    {"max_groups": 0}, {"max_groups": 101}, {"max_groups": True},
    {"columns": []}, {"columns": "value"}, {"columns": ["value", "value"]},
    {"groups": []}, {"groups": ["value", 1]},
])
def test_malformed_profile_controls_are_stable_errors(options):
    with pytest.raises(wrangle.WrangleError) as caught:
        validate_options(**options)
    assert caught.value.code == "INVALID_ARGUMENT"


@pytest.mark.parametrize("options", [{"columns": ["absent"]}, {"groups": ["absent"]}])
def test_absent_profile_columns_are_stable_errors(options):
    with pytest.raises(wrangle.WrangleError) as caught:
        profile(pl.DataFrame({"value": [1]}), {}, **options)
    assert caught.value.code == "UNKNOWN_COLUMN"


def test_group_output_limit_is_enforced_for_hundred_observed_groups():
    data = pl.DataFrame({"group": list(range(101)), "value": [1.0] * 101})
    result = profile(data, {}, groups=["group"], max_groups=100)
    assert result["groups"]["count"] == 101
    assert result["groups"]["truncated"] is True
    assert len(result["groups"]["items"]) == 100
    assert result["groups"]["items"][-1]["values"] == {"group": 99}


def test_profile_internal_aliases_cannot_shadow_group_labels():
    name = "__wrangle_profile_0_nulls"
    data = pl.DataFrame({name: ["a", "b"], "__wrangle_profile_rows": [1, 2]})
    groups = profile(data, {}, groups=[name])["groups"]
    assert [item["values"] for item in groups["items"]] == [{name: "a"}, {name: "b"}]
    assert groups["items"][1]["fields"][1]["summary"]["mean"] == 2.0


def test_public_inspect_preserves_saved_parent_lineage_and_compares_baseline(tmp_path):
    source = pl.DataFrame({"id": ["001", "002"], "value": [1.0, 3.0]})
    prepared = wrangle.prepare(source, {"key": "id", "units": {"value": "mg"}, "descriptions": {"value": "measured mass"}}, output=tmp_path / "batch")
    baseline = pl.DataFrame({"id": ["001"], "value": [1], "old": [True]})
    result = wrangle.inspect(tmp_path / "batch", summary=True, columns=["value"], baseline=baseline)
    assert result["parent"]["recipe_sha256"] == prepared.receipt["recipe_sha256"]
    assert result["parent"]["units"] == {"value": "mg"}
    assert result["parent"]["descriptions"]["value"] == "measured mass"
    assert result["schema_changes"] == {"added": [], "removed": [{"name": "old", "dtype": "Boolean"}], "type_changed": [{"name": "value", "before": "Int64", "after": "Float64"}]}
    assert _field(result, "value")["summary"]["mean"] == 2.0
    assert_frame_equal(prepared.data, source)
    assert_frame_equal(pl.read_parquet(tmp_path / "batch" / "data.parquet"), source)


def test_invalid_public_controls_never_read_source_or_baseline(monkeypatch):
    import wrangle._api as api
    read = []
    def unexpected_read(source):
        read.append(source)
        raise AssertionError("invalid controls must fail before source ingestion")
    monkeypatch.setattr(api, "_source", unexpected_read)
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.inspect("unread.csv", max_groups=101, baseline="unread baseline.csv")
    assert caught.value.code == "INVALID_ARGUMENT"
    assert read == []


@pytest.mark.parametrize("dtype,values,expected", [
    (pl.Binary, [b"\x00\xff", None, b""], ["00ff", None, ""]),
    (pl.List(pl.Binary), [[b"\x00\xff", None], None, []], [["00ff", None], None, []]),
    (pl.Array(pl.Binary, 2), [[b"\x00\xff", None], None, [b"", b"\x01"]], [["00ff", None], None, ["", "01"]]),
    (pl.Struct({"payload": pl.List(pl.Binary)}), [{"payload": [b"\x00\xff", None]}, None, {"payload": []}], [{"payload": ["00ff", None]}, None, {"payload": []}]),
])
def test_binary_display_is_explicit_native_hex_without_changing_source(dtype, values, expected):
    data = pl.DataFrame({"payload": pl.Series(values, dtype=dtype)})
    original = data.clone()
    plain = wrangle.inspect(data, sample_rows=0)
    result = wrangle.inspect(data, summary=True)
    assert [example["payload"] for example in result["examples"]] == expected
    assert result["columns"]["payload"] == str(dtype)
    assert _field(result, "payload")["dtype"] == str(dtype)
    assert result["example_encoding"]["binary"] == "lowercase hexadecimal strings"
    assert result["example_encoding"]["binary_columns"] == ["payload"]
    assert result["snapshot_sha256"] == plain["snapshot_sha256"]
    assert_frame_equal(data, original)


def test_binary_group_labels_keep_explicit_encoding_when_profile_columns_exclude_key():
    data = pl.DataFrame({"group": [b"\x00\xff", None, b"\x00\xff"], "value": [1, 2, 3]})
    result = wrangle.inspect(data, columns=["value"], groups=["group"])
    assert result["example_encoding"]["binary_columns"] == ["group"]
    assert [item["values"] for item in result["groups"]["items"]] == [{"group": "00ff"}, {"group": None}]
    assert result["groups"]["items"][0]["fields"][0]["summary"]["mean"] == 2.0


def test_json_nonfinite_display_is_declared_and_does_not_change_observed_counts():
    result = wrangle.inspect(pl.DataFrame({"value": [float("nan"), float("inf"), None]}))
    assert result["examples"] == [{"value": None}, {"value": None}, {"value": None}]
    assert "JSON null" in result["example_encoding"]["nonfinite_floats"]
    assert _field(result, "value") == {"name": "value", "dtype": "Float64", "nulls": 1, "nans": 1, "infinities": 1, "distinct": 3}


def test_empty_sources_never_fabricate_observed_groups():
    data = pl.DataFrame(schema={"group": pl.String, "value": pl.Float64})
    result = wrangle.inspect(data, summary=True, groups=["group"])
    assert result["rows"] == 0
    assert result["examples"] == []
    assert result["groups"] == {"by": ["group"], "count": 0, "max_groups": 20, "truncated": False, "items": []}
    assert _field(result, "value")["summary"]["count"] == 0


def test_shipped_and_checkout_inspection_examples_execute_identically(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    shipped = Path(wrangle.__file__).parent / "docs" / "inspection_workflow.py"
    checkout = Path(__file__).resolve().parents[1] / "examples" / "inspection_workflow.py"
    assert shipped.read_bytes() == checkout.read_bytes()
    expected = {"rows": 4, "groups": 2, "groups_truncated": True, "mean_mass_mg": 3.0, "sources_unchanged": True}
    for script in (shipped, checkout):
        result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert not result.stderr
        assert json.loads(result.stdout) == expected
