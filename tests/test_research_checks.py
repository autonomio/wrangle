"""Research release checks assert declared meaning without changing observations."""
from datetime import date
import json

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle


def test_complete_release_protocol_uses_native_checks_without_changing_data():
    data = pl.DataFrame({
        "id": ["a1", "b1", "a2", "b2"], "subject": ["a", "b", "a", "b"],
        "day": [date(2026, 10, 1), date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 2)],
        "start": [1.0, 100.0, 2.0, 101.0], "stop": [2.0, 101.0, 3.0, 102.0],
        "status": ["pass", "pass", "fail", "pass"], "mass": [0.0, None, 2.0, 3.0],
    })
    registry = pl.DataFrame({"participant": ["b", "a"]})
    checks = {
        "schema": {"id": "String", "subject": "String", "day": "Date", "start": "Float64", "stop": "Float64", "status": "String", "mass": "Float64"},
        "allowed": {"status": ["pass", "fail"]}, "patterns": {"id": "^[ab][12]$"},
        "ranges": {"mass": {"min": 0}}, "missing": {"mass": {"max": 0.25}},
        "temporal_ranges": {"day": {"min": "2026-10-01", "max": "2026-10-02"}},
        "assertions": [{"name": "positive acquisition duration", "where": {"ge": [{"col": "stop"}, {"col": "start"}]}}],
        "ordering": {"column": "start", "groups": ["subject"], "ties": "error"},
        "foreign_keys": [{"source": "registry", "columns": ["subject"], "reference": ["participant"]}],
        "group_counts": [{"by": ["subject"], "exact": 2}], "row_count": {"exact": 4},
        "protocol": {"key": True, "units": ["mass", "start", "stop"], "descriptions": ["status"]},
    }
    recipe = {"input": "observations", "key": "id", "units": {"mass": "g", "start": "s", "stop": "s"}, "descriptions": {"status": "Instrument quality status"}, "checks": checks}
    prepared = wrangle.prepare({"observations": data.lazy(), "registry": registry}, recipe)
    assert_frame_equal(prepared.data, data)
    assert_frame_equal(registry, pl.DataFrame({"participant": ["b", "a"]}))
    kinds = {check["check"] for check in prepared.receipt["checks"]}
    assert {"schema", "allowed", "pattern", "missing", "range", "assertion", "ordering", "foreign_key", "group_count", "protocol", "row_count"} <= kinds
    assert all(check["passed"] for check in prepared.receipt["checks"] if "passed" in check)
    missing = next(check for check in prepared.receipt["checks"] if check["check"] == "missing")
    assert missing["fraction"] == 0.25
    json.dumps(prepared.receipt, allow_nan=False)


def test_schema_checks_do_not_drop_extra_fields_or_recast_values():
    data = pl.DataFrame({"id": ["001"], "value": [1]})
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"schema": {"id": "String"}}})
    assert caught.value.code == "SCHEMA_MISMATCH"
    assert caught.value.details["extra_columns"] == ["value"]
    result = wrangle.prepare(data, {"checks": {"schema": {"id": "String"}, "extra_columns": "allow"}})
    assert_frame_equal(result.data, data)
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"schema": {"id": "String", "value": "Float64"}}})
    assert caught.value.code == "SCHEMA_MISMATCH"
    assert caught.value.details["columns"]["value"] == {"actual": "Int64", "expected": "Float64"}


@pytest.mark.parametrize("values,domain,code", [
    (["pass", "other"], ["pass", "fail"], "DOMAIN_VIOLATION"),
    (["pass", None], ["pass"], "DOMAIN_VIOLATION"),
    ([1, 2], ["1", "2"], "INVALID_RECIPE"),
])
def test_typed_domain_failures_are_stable(values, domain, code, tmp_path):
    destination = tmp_path / "must not publish"
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(pl.DataFrame({"status": values}), {"checks": {"allowed": {"status": domain}}}, output=destination)
    assert caught.value.code == code
    assert not destination.exists()


def test_null_domain_membership_must_be_declared():
    data = pl.DataFrame({"status": ["pass", None]})
    assert_frame_equal(wrangle.prepare(data, {"checks": {"allowed": {"status": ["pass", None]}}}).data, data)


def test_pattern_boundaries_and_invalid_native_regular_expression():
    data = pl.DataFrame({"id": ["S001", "prefix-S002"]})
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"patterns": {"id": "^S[0-9]{3}$"}}})
    assert caught.value.code == "PATTERN_VIOLATION"
    assert caught.value.details["affected_rows"] == 1
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"patterns": {"id": "["}}})
    assert caught.value.code == "INVALID_RECIPE"


def test_missing_fraction_uses_all_observations_and_preserves_nulls():
    data = pl.DataFrame({"value": [1.0, None, None, 4.0]})
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"missing": {"value": {"max": 0.49}}}})
    assert caught.value.code == "MISSINGNESS_VIOLATION"
    assert caught.value.details == {"column": "value", "missing": 2, "rows": 4, "fraction": 0.5}
    prepared = wrangle.prepare(data, {"checks": {"missing": {"value": {"min": 0.5, "max": 0.5}}}})
    assert_frame_equal(prepared.data, data)


def test_missing_fraction_with_zero_denominator_is_not_reported_as_zero():
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(pl.DataFrame(schema={"value": pl.Float64}), {"checks": {"missing": {"value": {"max": 0}}}})
    assert caught.value.code == "UNDEFINED_MISSINGNESS"


def test_temporal_range_checks_typed_boundaries():
    data = pl.DataFrame({"day": [date(2026, 10, 1), date(2026, 10, 2)]})
    result = wrangle.prepare(data, {"checks": {"temporal_ranges": {"day": {"min": "2026-10-01", "max": "2026-10-02"}}}})
    assert_frame_equal(result.data, data)
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"temporal_ranges": {"day": {"max": "2026-10-01"}}}})
    assert caught.value.code == "RANGE_VIOLATION"
    assert caught.value.details["affected_rows"] == 1
    for limits in ({"max": "not-a-date"}, {"min": "2026-10-03", "max": "2026-10-01"}):
        with pytest.raises(wrangle.WrangleError) as caught:
            wrangle.prepare(data, {"checks": {"temporal_ranges": {"day": limits}}})
        assert caught.value.code == "INVALID_RECIPE"


@pytest.mark.parametrize("policy,code", [("error", "UNRESOLVED_ASSERTION"), ("fail", "ASSERTION_FAILED"), ("pass", None)])
def test_cross_field_assertion_has_explicit_unknown_truth_policy(policy, code):
    data = pl.DataFrame({"start": [1.0, None], "stop": [2.0, 3.0]})
    assertion = {"name": "duration", "where": {"ge": [{"col": "stop"}, {"col": "start"}]}, "nulls": policy}
    if code:
        with pytest.raises(wrangle.WrangleError) as caught:
            wrangle.prepare(data, {"checks": {"assertions": [assertion]}})
        assert caught.value.code == code
        assert caught.value.details["affected_rows"] == 1
    else:
        assert_frame_equal(wrangle.prepare(data, {"checks": {"assertions": [assertion]}}).data, data)


@pytest.mark.parametrize("where", [{"add": [{"col": "start"}, 1]}, {"python": "run arbitrary callback"}])
def test_assertions_compile_only_declared_native_boolean_expressions(where):
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(pl.DataFrame({"start": [1]}).lazy(), {"checks": {"assertions": [{"name": "must be Boolean", "where": where}]}})
    assert caught.value.code == "INVALID_EXPRESSION"


def test_ordering_is_within_subject_even_when_subjects_are_interleaved():
    data = pl.DataFrame({"subject": ["a", "b", "a", "b"], "time": [1, 100, 2, 101]})
    rule = {"column": "time", "groups": ["subject"], "ties": "error"}
    assert_frame_equal(wrangle.prepare(data, {"checks": {"ordering": rule}}).data, data)
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data.with_columns(pl.when(pl.int_range(pl.len()) == 2).then(0).otherwise(pl.col("time")).alias("time")), {"checks": {"ordering": rule}})
    assert caught.value.code == "ORDER_VIOLATION"
    assert caught.value.details["affected_rows"] == 1


@pytest.mark.parametrize("values,rule,code", [
    ([1, 1], {"ties": "error"}, "ORDER_VIOLATION"),
    ([1, 2], {"descending": True}, "ORDER_VIOLATION"),
    ([1, None, 2], {}, "MISSING_REQUIRED"),
    ([1, None, 2], {"nulls": "skip"}, None),
    ([3, 2, 1], {"descending": True}, None),
])
def test_ordering_direction_ties_and_missing_values_are_declared(values, rule, code):
    data = pl.DataFrame({"time": values})
    recipe = {"checks": {"ordering": [{"column": "time", **rule}]}}
    if code:
        with pytest.raises(wrangle.WrangleError) as caught:
            wrangle.prepare(data, recipe)
        assert caught.value.code == code
    else:
        assert_frame_equal(wrangle.prepare(data, recipe).data, data)


@pytest.mark.parametrize("reference,code", [
    (pl.DataFrame({"participant": ["a"]}), "FOREIGN_KEY_VIOLATION"),
    (pl.DataFrame({"participant": ["a", "a", "b"]}), "FOREIGN_KEY_CARDINALITY"),
    (pl.DataFrame({"participant": ["a", None]}), "FOREIGN_KEY_CARDINALITY"),
    (pl.DataFrame({"participant": [1, 2]}), "DTYPE_MISMATCH"),
])
def test_foreign_keys_assert_membership_and_parent_cardinality(reference, code):
    data = pl.DataFrame({"subject": ["a", "b"]})
    rule = {"source": "registry", "columns": ["subject"], "reference": ["participant"]}
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare({"observations": data, "registry": reference}, {"input": "observations", "checks": {"foreign_keys": [rule]}})
    assert caught.value.code == code


def test_foreign_key_missing_policy_does_not_fabricate_or_drop_records():
    data = pl.DataFrame({"subject": ["a", None]})
    sources = {"observations": data, "registry": pl.DataFrame({"participant": ["a"]})}
    rule = {"source": "registry", "columns": ["subject"], "reference": ["participant"]}
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(sources, {"input": "observations", "checks": {"foreign_keys": [rule]}})
    assert caught.value.code == "NULL_KEY"
    result = wrangle.prepare(sources, {"input": "observations", "checks": {"foreign_keys": [{**rule, "missing": "allow"}]}})
    assert_frame_equal(result.data, data)


def test_group_counts_check_observed_groups_and_explicit_null_group_policy():
    data = pl.DataFrame({"group": ["a", "a", "b"]})
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(data, {"checks": {"group_counts": [{"by": ["group"], "min": 2}]}})
    assert caught.value.code == "GROUP_COUNT_VIOLATION"
    assert caught.value.details["affected_groups"] == 1
    assert_frame_equal(wrangle.prepare(data, {"checks": {"group_counts": [{"by": ["group"], "min": 1, "max": 2}]}}).data, data)
    missing_group = pl.DataFrame({"group": ["a", None]})
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(missing_group, {"checks": {"group_counts": [{"by": ["group"], "exact": 1}]}})
    assert caught.value.code == "NULL_KEY"
    assert_frame_equal(wrangle.prepare(missing_group, {"checks": {"group_counts": [{"by": ["group"], "exact": 1, "nulls": "group"}]}}).data, missing_group)


@pytest.mark.parametrize("protocol,declarations", [
    ({"key": True}, {}), ({"units": ["value"]}, {}), ({"descriptions": ["value"]}, {}),
])
def test_required_protocol_declarations_fail_when_unresolved(protocol, declarations):
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(pl.DataFrame({"value": [1]}), {"checks": {"protocol": protocol}, **declarations})
    assert caught.value.code == "UNRESOLVED_PROTOCOL"


def test_protocol_can_require_declared_dimensionless_meaning():
    result = wrangle.prepare(pl.DataFrame({"id": ["001"], "ratio": [1.0]}), {"key": "id", "units": {"ratio": "1"}, "descriptions": {"ratio": "Signal divided by baseline"}, "checks": {"protocol": {"key": True, "units": ["ratio"], "descriptions": ["ratio"]}}})
    assert result.receipt["units"] == {"ratio": "1"}


@pytest.mark.parametrize("rules", [
    {"extra_columns": []}, {"allowed": {"value": "allowed"}}, {"patterns": {"value": 12}},
    {"missing": {"value": {"max": 1.1}}}, {"missing": {"value": {}}},
    {"ranges": {"value": {"min": 2, "max": 1}}}, {"temporal_ranges": {"value": {}}},
    {"assertions": [{"name": "bad policy", "where": True, "nulls": []}]},
    {"ordering": {"column": "value", "ties": []}}, {"ordering": {"column": "value", "groups": "subject"}},
    {"foreign_keys": [{"source": "registry", "columns": ["value"], "reference": ["a", "b"]}]},
    {"group_counts": [{"by": ["value"]}]}, {"group_counts": [{"by": [], "exact": 1}]},
    {"protocol": {"key": "yes"}}, {"row_count": {"exact": -1}}, {"unknown_check": True},
])
def test_malformed_rules_fail_with_stable_code_before_publication(rules, tmp_path):
    destination = tmp_path / "must not publish"
    with pytest.raises(wrangle.WrangleError) as caught:
        wrangle.prepare(pl.DataFrame({"value": [1]}), {"checks": rules}, output=destination)
    assert caught.value.code == "INVALID_RECIPE"
    assert not destination.exists()
