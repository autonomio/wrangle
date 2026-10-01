"""Join mode/cardinality contracts and bounded declarative expression semantics."""
from datetime import date, datetime, timezone
import json
import polars as pl
import pytest
from wrangle._core import WrangleError, require_native
from wrangle._recipe_join import join, output_mapping, resolved_parameters
from wrangle._recipe_expressions import compile_expression, EXPRESSION_DEFINITIONS
from wrangle._api import _expression


def table(**columns):
    return pl.DataFrame(columns).lazy()


def compiler(value, data):
    extended = compile_expression(value, data, compiler)
    return extended if extended is not None else _expression(value, data)


def evaluate(value, data):
    expr = compiler(value, data)
    require_native(expr)
    return data.select(expr.alias("result")).collect()["result"].to_list()


def test_legacy_join_alias_and_output_suffix_are_preserved():
    left, right = table(id=[1, 2], x=[10, 11]), table(id=[1, 2], x=[20, 21])
    result = join(left, right, on="id", validate="m:1").collect()
    assert result.to_dict(as_series=False) == {"id": [1, 2], "x": [10, 11], "x_right": [20, 21]}
    assert output_mapping(["id", "x"], ["id", "x"], {"on": "id"}) == {"id": "id", "x": "x_right"}
    with pytest.raises(WrangleError, match="agree"):
        join(left, right, on="id", validate="1:1", cardinality="m:m")


def test_distinct_join_keys_and_coalescing_have_explicit_output_mapping():
    left, right = table(sample=["a"], x=[10]), table(subject=["a"], x=[20])
    options = dict(left_on="sample", right_on="subject", cardinality="1:1", unmatched="error")
    assert join(left, right, **options).collect().columns == ["sample", "x", "subject", "x_right"]
    assert output_mapping(["sample", "x"], ["subject", "x"], options) == {"subject": "subject", "x": "x_right"}
    assert output_mapping(["sample", "x"], ["subject", "x"], {**options, "coalesce": True}) == {"subject": "sample", "x": "x_right"}


def test_full_join_declares_both_unmatched_sides_and_source_order():
    left, right = table(id=[2, 1], l=[20, 10]), table(id=[3, 1], r=[30, 11])
    options = dict(on="id", how="full", cardinality="1:1", unmatched={"left": "keep", "right": "keep"})
    with pytest.raises(WrangleError, match="order"):
        join(left, right, **options)
    result = join(left, right, **options, maintain_order="left_right").collect()
    assert result["id"].to_list() == [2, 1, 3]
    assert result["r"].to_list() == [None, 11, 30]
    result = join(left, right, **{**options, "unmatched": {"left": "drop", "right": "keep"}}, maintain_order="right_left").collect()
    assert result["id"].to_list() == [3, 1]
    with pytest.raises(WrangleError) as caught:
        join(left, right, **{**options, "unmatched": {"left": "keep", "right": "error"}}, maintain_order="left_right")
    assert caught.value.code == "UNMATCHED_KEYS" and caught.value.details["side"] == "right"


def test_right_join_cardinality_uses_native_preflight_for_unsupported_engine_validation():
    left, right = table(id=[1, 1], x=[2, 3]), table(id=[1, 2], y=[4, 5])
    opts = dict(on="id", how="right", unmatched={"left": "drop", "right": "keep"}, maintain_order="right_left")
    result = join(left, right, **opts, cardinality="m:1").collect()
    assert result["id"].to_list() == [1, 1, 2]
    with pytest.raises(WrangleError) as caught:
        join(left, right, **opts, cardinality="1:1")
    assert caught.value.code == "JOIN_CARDINALITY" and caught.value.details["side"] == "left"


def test_semi_anti_return_left_payload_once_under_declared_side_policy():
    left, right = table(id=[3, 1, 2], x=[30, 10, 20]), table(id=[1, 1, 2], y=[5, 6, 7])
    assert join(left, right, on="id", how="semi", cardinality="m:m", unmatched="drop").collect()["id"].to_list() == [1, 2]
    assert join(left, right, on="id", how="anti", cardinality="m:m", unmatched="keep").collect().to_dict(as_series=False) == {"id": [3], "x": [30]}
    assert output_mapping(["id", "x"], ["id", "y"], dict(on="id", how="anti", unmatched="keep")) == {}
    with pytest.raises(WrangleError) as caught:
        join(left, right, on="id", how="semi", cardinality="m:1", unmatched="drop")
    assert caught.value.code == "JOIN_CARDINALITY"


def test_null_keys_never_match_and_drop_is_explicit():
    left, right = table(id=[None, 1], x=[10, 11]), table(id=[None, 1], y=[20, 21])
    with pytest.raises(WrangleError) as caught:
        join(left, right, on="id", unmatched="keep")
    assert caught.value.code == "NULL_KEY"
    assert join(left, right, on="id", unmatched="keep", nulls="drop").collect().height == 1
    with pytest.raises(WrangleError, match="dtypes"):
        join(left, table(id=[1.0]), on="id")


def test_join_checks_overlap_and_suffix_collisions_before_execution():
    left, right = table(id=[1], x=[10]), table(id=[1], x=[20], x_right=[30])
    with pytest.raises(WrangleError) as caught:
        join(left, right, on="id")
    assert caught.value.code == "OUTPUT_COLLISION"
    with pytest.raises(WrangleError) as caught:
        join(left, table(id=[1], x=[20]), on="id", overlap="error")
    assert caught.value.code == "OVERLAPPING_COLUMNS"


def test_join_parameters_are_resolved_json_without_lazy_sources():
    params = resolved_parameters({"on": "id", "validate": "m:1", "unmatched": "keep"})
    assert params["cardinality"] == "m:1" and params["coalesce"] is True
    assert params["unmatched"] == {"left": "keep", "right": "drop"}
    json.dumps(params, allow_nan=False)
    with pytest.raises(WrangleError):
        resolved_parameters({"on": "id", "cardinality": False})


def test_when_case_coalesce_have_declared_null_semantics():
    data = table(flag=[True, False, None], x=[2, None, 4])
    node = {"when": {"if": {"col": "flag"}, "then": "yes", "else": "no", "nulls": "null"}}
    assert evaluate(node, data) == ["yes", "no", None]
    with pytest.raises(WrangleError) as caught:
        evaluate({"when": {**node["when"], "nulls": "error"}}, data)
    assert caught.value.code == "UNRESOLVED_EXPRESSION"
    assert evaluate({"coalesce": [{"col": "x"}, 0]}, data) == [2, 0, 4]
    node = {"case": {"branches": [{"when": {"col": "flag"}, "then": 1}, {"when": True, "then": 2}], "else": 3, "nulls": "null"}}
    assert evaluate(node, data) == [1, 2, None]


def test_branch_float_promotion_cannot_collapse_neighboring_integer_observations():
    data = table(x=[2**53 + 1], flag=[True])
    with pytest.raises(WrangleError) as caught:
        evaluate({"when": {"if": {"col": "flag"}, "then": {"col": "x"}, "else": 0.0, "nulls": "error"}}, data)
    assert caught.value.code == "LOSSY_CAST"


def test_exact_membership_and_missing_truth_are_explicit():
    data = table(x=[1, None, 3])
    assert evaluate({"is_in": {"value": {"col": "x"}, "values": [1, None], "nulls": "match"}}, data) == [True, True, False]
    assert evaluate({"is_in": {"value": {"col": "x"}, "values": [1, None], "nulls": "false"}}, data) == [True, False, False]
    with pytest.raises(WrangleError) as caught:
        evaluate({"is_in": {"value": {"col": "x"}, "values": [1.5], "nulls": "false"}}, data)
    assert caught.value.code == "LOSSY_CAST"


def test_cast_invalid_values_and_precision_choices_are_explicit():
    data = table(x=["2", "bad"])
    node = {"cast": {"value": {"col": "x"}, "dtype": "Int64", "invalid": "null", "precision": "exact"}}
    assert evaluate(node, data) == [2, None]
    with pytest.raises(WrangleError) as caught:
        evaluate({"cast": {**node["cast"], "invalid": "error"}}, data)
    assert caught.value.code == "INVALID_CAST"
    with pytest.raises(WrangleError) as caught:
        evaluate({"literal": {"value": 2**53 + 1, "dtype": "Float64"}}, table(id=[1]))
    assert caught.value.code == "LOSSY_CAST"
    assert evaluate({"literal": {"value": [1, 2], "dtype": {"List": "Int64"}}}, table(id=[1])) == [[1, 2]]


def test_temporal_literal_and_components_use_explicit_units_and_iso_calendar():
    data = table(day=[date(2024, 1, 1), date(2024, 1, 7)])
    assert evaluate({"date_part": {"value": {"col": "day"}, "part": "weekday"}}, data) == [1, 7]
    assert evaluate({"date_part": {"value": {"literal": {"value": "2024-02-29", "dtype": "Date"}}, "part": "day"}}, data) == [29]
    with pytest.raises(WrangleError):
        evaluate({"literal": {"value": "2500-01-01T00:00:00", "dtype": {"Datetime": {"time_unit": "ns"}}}}, data)


def test_text_clean_slicing_contains_and_literal_replacement_are_predictable():
    data = table(s=[" A.a ", None])
    assert evaluate({"lower": {"trim": {"col": "s"}}}, data) == ["a.a", None]
    assert evaluate({"slice": {"value": {"col": "s"}, "offset": 1, "length": 3}}, data) == ["A.a", None]
    assert evaluate({"contains": {"value": {"col": "s"}, "pattern": ".", "literal": True}}, data) == [True, None]
    replacement = {"replace": {"value": {"col": "s"}, "pattern": ".", "replacement": "$5", "literal": True, "all": True, "capture_groups": False}}
    assert evaluate(replacement, data) == [" A$5a ", None]
    replacement["replace"].update(pattern="A", literal=False)
    assert evaluate(replacement, data) == [" $5.a ", None]


def test_regex_extract_and_replacement_reject_absent_or_invalid_captures():
    data = table(s=["a-12"])
    assert evaluate({"extract": {"value": {"col": "s"}, "pattern": r"([a-z]+)-([0-9]+)", "group": 2}}, data) == ["12"]
    node = {"replace": {"value": {"col": "s"}, "pattern": r"([a-z]+)-([0-9]+)", "replacement": "${2}:${1}", "literal": False, "all": True, "capture_groups": True}}
    assert evaluate(node, data) == ["12:a"]
    with pytest.raises(WrangleError):
        evaluate({"replace": {**node["replace"], "replacement": "${3}"}}, data)
    with pytest.raises(WrangleError):
        evaluate({"extract": {"value": {"col": "s"}, "pattern": "([", "group": 1}}, data)
    with pytest.raises(WrangleError):
        evaluate({"extract": {"value": {"col": "s"}, "pattern": "(a)", "group": 2}}, data)


def test_string_concat_has_explicit_missing_and_separator_policies():
    data = table(a=["x", None], b=[None, None])
    node = {"concat": {"values": [{"col": "a"}, {"col": "b"}], "separator": ":", "nulls": "ignore"}}
    assert evaluate(node, data) == ["x", ""]
    assert evaluate({"concat": {**node["concat"], "nulls": "propagate"}}, data) == [None, None]
    with pytest.raises(WrangleError):
        evaluate({"concat": {**node["concat"], "nulls": "error"}}, data)


def test_rounding_ties_and_clip_domains_are_explicit():
    data = table(x=[1.5, 2.5, None])
    assert evaluate({"round": {"value": {"col": "x"}, "decimals": 0, "mode": "half_to_even"}}, data) == [2.0, 2.0, None]
    assert evaluate({"round": {"value": {"col": "x"}, "decimals": 0, "mode": "half_away_from_zero"}}, data) == [2.0, 3.0, None]
    assert evaluate({"clip": {"value": {"col": "x"}, "min": 2.0, "max": 3.0, "nulls": "keep"}}, data) == [2.0, 2.5, None]
    with pytest.raises(WrangleError) as caught:
        evaluate({"clip": {"value": {"col": "x"}, "min": 3.0, "max": 2.0, "nulls": "keep"}}, data)
    assert caught.value.code == "INVALID_DOMAIN"


@pytest.mark.parametrize("node", [{"when": {"if": True, "then": 1, "else": 2}}, {"coalesce": []}, {"cast": {"value": 1, "dtype": "Float64"}}, {"contains": {"value": "a", "pattern": "a", "literal": "yes"}}, {"clip": {"value": 1, "min": None, "max": None}}])
def test_incomplete_or_unknown_scientific_choices_fail(node):
    with pytest.raises(WrangleError):
        evaluate(node, table(id=[1]))


def test_extension_metadata_is_machine_readable_and_unknown_operators_are_not_executed():
    assert len(EXPRESSION_DEFINITIONS) == 21
    json.dumps(EXPRESSION_DEFINITIONS)
    assert compile_expression({"eval": "1+1"}, table(id=[1]), compiler) is None


def test_right_coalesced_key_mapping_matches_actual_retained_axis_and_collisions():
    left, right = table(sample=[1], subject=["payload"]), table(subject=[1, 2], sample=["p", "q"])
    opts = dict(left_on="sample", right_on="subject", how="right", coalesce=True, cardinality="1:1", unmatched={"left": "drop", "right": "keep"}, maintain_order="left_right")
    assert output_mapping(["sample", "subject"], ["subject", "sample"], opts) == {"subject": "subject_right", "sample": "sample"}
    assert join(left, right, **opts).collect().to_dict(as_series=False) == {"subject": ["payload", None], "subject_right": [1, 2], "sample": ["p", "q"]}
    with pytest.raises(WrangleError) as caught:
        join(left, table(subject=[1], subject_right=["p"]), **opts)
    assert caught.value.code == "OUTPUT_COLLISION"
    with pytest.raises(WrangleError) as caught:
        join(table(id=[1]), table(id=[1.0]), on="id")
    assert caught.value.code == "DTYPE_MISMATCH" and caught.value.details["columns"] == ["id"]


def shift(value, offset, mode="calendar", precision="exact", month_end="error"):
    return {"datetime_shift": {"value": value, "offset": offset, "mode": mode, "precision": precision, "month_end": month_end}}


def test_bounded_split_preserves_empty_tokens_remainder_and_missingness():
    data = table(s=["a:b:c", "a:", ":", None])
    node = {"split": {"value": {"col": "s"}, "separator": ":", "max_splits": 1, "nulls": "keep"}}
    assert evaluate(node, data) == [["a", "b:c"], ["a", ""], ["", ""], None]
    assert evaluate({"split": {**node["split"], "max_splits": 0}}, data) == [["a:b:c"], ["a:"], [":"], None]
    with pytest.raises(WrangleError):
        evaluate({"split": {**node["split"], "nulls": "error"}}, data)


def test_timezone_conversion_preserves_instants_and_refuses_naive_time():
    data = table(time=[datetime(2024, 3, 31, 0, 30, tzinfo=timezone.utc), None])
    node = {"convert_time_zone": {"value": {"col": "time"}, "time_zone": "Europe/Helsinki"}}
    result = data.select(compiler(node, data).alias("time")).collect()
    assert result["time"].cast(pl.Int64).to_list() == data.collect()["time"].cast(pl.Int64).to_list()
    assert result["time"][0].hour == 2
    with pytest.raises(WrangleError):
        evaluate(node, table(time=[datetime(2024, 3, 31)]))
    with pytest.raises(WrangleError):
        evaluate({"convert_time_zone": {**node["convert_time_zone"], "time_zone": "Bad/Zone"}}, data)


def test_calendar_month_end_and_timestamp_wrap_are_explicit_failures():
    data = table(time=[datetime(2024, 1, 31), None])
    with pytest.raises(WrangleError) as caught:
        evaluate(shift({"col": "time"}, "1mo"), data)
    assert caught.value.code == "INVALID_DATETIME_SHIFT"
    assert evaluate(shift({"col": "time"}, "1mo", month_end="clip"), data) == [datetime(2024, 2, 29), None]
    ns = table(time=[datetime(2024, 1, 1)]).with_columns(pl.col("time").cast(pl.Datetime("ns")))
    for offset in ("500y", "6000y", "-1000y"):
        with pytest.raises(WrangleError) as caught:
            evaluate(shift({"col": "time"}, offset), ns)
        assert caught.value.code == "DATETIME_OVERFLOW"
    assert evaluate(shift({"col": "day"}, "-1mo", month_end="clip"), table(day=[date(2024, 3, 31)])) == [date(2024, 2, 29)]


def test_elapsed_precision_and_overflow_do_not_fabricate_instants():
    data = table(time=[datetime(2024, 1, 1)])
    with pytest.raises(WrangleError) as caught:
        evaluate(shift({"col": "time"}, "1ns", mode="elapsed"), data)
    assert caught.value.code == "LOSSY_CAST"
    assert evaluate(shift({"col": "time"}, "-1001ns", mode="elapsed", precision="allow"), data) == [datetime(2023, 12, 31, 23, 59, 59, 999999)]
    edge = pl.DataFrame({"time": pl.Series([2**63 - 1], dtype=pl.Int64).cast(pl.Datetime("ns"))}).lazy()
    with pytest.raises(WrangleError) as caught:
        evaluate(shift({"col": "time"}, "1ns", mode="elapsed"), edge)
    assert caught.value.code == "DATETIME_OVERFLOW"


def test_calendar_day_differs_from_elapsed_day_across_dst_and_ambiguity_fails():
    data = table(time=[datetime(2024, 3, 30, 10, tzinfo=timezone.utc)]).with_columns(pl.col("time").dt.convert_time_zone("Europe/Helsinki"))
    calendar = evaluate(shift({"col": "time"}, "1d"), data)[0]
    elapsed = evaluate(shift({"col": "time"}, "1d", mode="elapsed"), data)[0]
    assert calendar.hour == 12 and elapsed.hour == 13
    before = table(time=[datetime(2024, 3, 30, 1, 30, tzinfo=timezone.utc)]).with_columns(pl.col("time").dt.convert_time_zone("Europe/Helsinki"))
    with pytest.raises(WrangleError) as caught:
        evaluate(shift({"col": "time"}, "1d"), before)
    assert caught.value.code == "INVALID_DATETIME_SHIFT"


def test_calendar_shift_does_not_choose_an_ambiguous_dst_occurrence():
    data = table(time=[datetime(2024, 10, 26, 0, 30, tzinfo=timezone.utc)]).with_columns(pl.col("time").dt.convert_time_zone("Europe/Helsinki"))
    with pytest.raises(WrangleError) as caught:
        evaluate(shift({"col": "time"}, "1d"), data)
    assert caught.value.code == "INVALID_DATETIME_SHIFT"


def test_declared_literal_precision_is_checked_even_for_zero_observations():
    with pytest.raises(WrangleError) as caught:
        evaluate({"literal": {"value": 2**53 + 1, "dtype": "Float64"}}, pl.DataFrame(schema={"id": pl.Int64}).lazy())
    assert caught.value.code == "LOSSY_CAST"
    with pytest.raises(WrangleError) as caught:
        evaluate({"cast": {"value": {"col": "x"}, "dtype": "Boolean", "invalid": "error", "precision": "exact"}}, table(x=[0, 2]))
    assert caught.value.code == "LOSSY_CAST"
    assert evaluate({"cast": {"value": {"col": "x"}, "dtype": "Boolean", "invalid": "error", "precision": "allow"}}, table(x=[0, 2])) == [False, True]


def test_verified_expression_workflow_is_packaged_and_executes():
    from wrangle.docs.expression_workflow import run
    assert run().data.height == 2
