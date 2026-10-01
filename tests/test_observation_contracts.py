"""Scientific grain, lineage, exclusion, and semantic propagation contracts."""
from datetime import date
import polars as pl
import pytest
import wrangle as wr


def table(**columns):
    return pl.DataFrame(columns)


def reduction(column, method, **options):
    return {"column": column, "method": method, "nulls": "ignore", "min_count": 1, **options}


def fails(data, recipe, codes):
    with pytest.raises(wr.WrangleError) as caught:
        wr.prepare(data, recipe)
    assert caught.value.code in ({codes} if isinstance(codes, str) else set(codes))
    return caught.value


def test_raw_duplicate_cleanup_can_establish_the_first_observation_key():
    raw = table(sample=["a", "a", "b"], x=[2, 2, 3])
    recipe = {"steps": [{"op": "deduplicate", "by": ["sample"], "keep": "first", "conflicts": "error", "ties": "source", "reason": "Repeated exported records", "key": ["sample"]}]}
    result = wr.prepare(raw, recipe)
    assert result.data["sample"].to_list() == ["a", "b"]
    assert result.receipt["key"] == ["sample"]
    assert result.receipt["steps"][0]["observation_transition"]["input_key"] == []
    fails(raw, {"key": ["sample"], "steps": recipe["steps"]}, "DUPLICATE_KEY")
    fails(raw, {"steps": [{key: value for key, value in recipe["steps"][0].items() if key != "reason"}]}, "UNDECLARED_LOSS")


def test_raw_duplicate_conflicts_cannot_be_hidden_by_new_key():
    raw = table(sample=["a", "a"], x=[2, 3])
    fails(raw, {"steps": [{"op": "deduplicate", "by": ["sample"], "keep": "first", "conflicts": "error", "ties": "source", "key": "sample", "reason": "Export duplicates"}]}, "DUPLICATE_CONFLICT")


def test_aggregate_requires_and_records_exact_output_group_key():
    raw = table(id=[1, 2, 3], subject=["a", "a", "b"], x=[2, 4, 6])
    step = {"op": "aggregate", "by": ["subject"], "metrics": {"mean": reduction("x", "mean")}, "null_keys": "error"}
    fails(raw, {"key": "id", "steps": [step]}, "OBSERVATION_UNIT_CHANGED")
    fails(raw, {"key": "id", "steps": [{**step, "key": ["mean"]}]}, "OBSERVATION_UNIT_CHANGED")
    result = wr.prepare(raw, {"key": "id", "units": {"x": "mg"}, "steps": [{**step, "key": "subject"}]})
    assert result.data["mean"].to_list() == [3.0, 6.0]
    assert result.receipt["key"] == ["subject"]
    assert result.receipt["units"] == {"mean": "mg"}
    assert result.receipt["steps"][0]["observation_transition"]["group_by"] == ["subject"]


def test_pivot_requires_declared_index_grain_and_reports_new_measurements():
    raw = table(id=[1, 2, 3, 4], subject=["a", "a", "b", "b"], assay=["x", "y", "x", "y"], amount=[2.0, 4.0, 6.0, 8.0])
    step = {"op": "pivot", "index": ["subject"], "on": "assay", "values": ["amount"], "domain": ["x", "y"], "duplicate_cells": "error", "missing_cells": "error"}
    fails(raw, {"key": "id", "steps": [step]}, "OBSERVATION_UNIT_CHANGED")
    fails(raw, {"key": "id", "steps": [{**step, "key": ["amount__x"]}]}, "OBSERVATION_UNIT_CHANGED")
    result = wr.prepare(raw, {"key": "id", "units": {"amount": "mg"}, "steps": [{**step, "key": ["subject"]}]})
    assert result.receipt["units"] == {"amount__x": "mg", "amount__y": "mg"}
    assert result.receipt["steps"][0]["observation_transition"]["group_by"] == ["subject"]


def test_unpivot_explicit_parent_variable_identity_and_expansion():
    raw = table(id=[1, 2], a=[2.0, 3.0], b=[4.0, 5.0])
    step = {"op": "unpivot", "index": ["id"], "on": ["a", "b"], "variable": "assay", "value": "amount", "nulls": "keep", "key": ["id", "assay"]}
    fails(raw, {"key": "id", "steps": [step]}, "UNDECLARED_EXPANSION")
    result = wr.prepare(raw, {"key": "id", "units": {"a": "mg", "b": "mg"}, "steps": [{**step, "allow_expand": True}]})
    assert result.receipt["key"] == ["id", "assay"]
    assert result.receipt["units"] == {"amount": "mg"}
    assert result.receipt["steps"][0]["observation_transition"]["parent_key"] == ["id"]
    fails(raw, {"key": "id", "steps": [{**step, "allow_expand": True, "key": ["assay", "amount"]}]}, "OBSERVATION_UNIT_CHANGED")


def test_unpivot_heterogeneous_units_require_explicit_unit_column():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "unpivot", "index": ["id"], "on": ["mass", "volume"], "variable": "variable", "value": "value", "nulls": "keep", "allow_expand": True, "key": ["id", "variable"]}
    recipe = {"key": "id", "units": {"mass": "mg", "volume": "mL"}, "steps": [step]}
    fails(raw, recipe, "UNIT_MISMATCH")
    result = wr.prepare(raw, {**recipe, "steps": [{**step, "unit_column": "unit"}]})
    assert result.data["unit"].to_list() == ["mg", "mL"]
    assert result.receipt["units"] == {"value": "@unit"}
    assert result.receipt["variable_units"][0]["mapping"] == {"mass": "mg", "volume": "mL"}
    fails(raw, {**recipe, "units": {"mass": "mg"}, "steps": [{**step, "unit_column": "unit"}]}, "UNDECLARED_UNITS")


def test_unpivot_cannot_drop_parent_identity_even_for_unique_new_values():
    raw = table(id=[1], subject=["a"], a=[2.0], b=[4.0])
    step = {"op": "unpivot", "index": ["subject"], "on": ["a", "b"], "variable": "assay", "value": "amount", "nulls": "keep", "allow_expand": True, "key": ["subject", "assay"]}
    fails(raw, {"key": "id", "steps": [step]}, "OBSERVATION_UNIT_CHANGED")


def test_explode_keeps_parent_identity_and_absent_list_sentinel():
    raw = table(id=[1, 2, 3], x=[[10, 11], [], None])
    step = {"op": "explode", "columns": ["x"], "index": "element", "empty": "null", "nulls": "null", "allow_expand": True, "key": ["id", "element"]}
    result = wr.prepare(raw, {"key": "id", "steps": [step]})
    assert result.data["element"].to_list() == [0, 1, -1, -1]
    assert result.receipt["key"] == ["id", "element"]
    assert result.receipt["steps"][0]["observation_transition"]["element_index"] == "element"
    fails(raw, {"key": "id", "steps": [{**step, "key": ["element"]}]}, {"DUPLICATE_KEY", "OBSERVATION_UNIT_CHANGED"})


def test_concat_provenance_disambiguates_reused_batch_identifiers():
    sources = {"left": table(id=[1], x=[2.0]), "right": table(id=[1], x=[3.0])}
    step = {"op": "concat", "sources": ["right"], "schema": "strict", "provenance": "batch", "labels": ["before", "after"], "allow_expand": True, "key": ["batch", "id"]}
    recipe = {"input": "left", "key": "id", "units": {"x": "mg"}, "source_contracts": {"right": {"units": {"x": "mg"}}}, "steps": [step]}
    result = wr.prepare(sources, recipe)
    assert result.data["batch"].to_list() == ["before", "after"]
    assert result.receipt["steps"][0]["observation_transition"]["sources"] == ["left", "right"]
    fails(sources, {**recipe, "steps": [{**step, "key": "id"}]}, "DUPLICATE_KEY")
    fails(sources, {**recipe, "source_contracts": {"right": {"units": {"x": "g"}}}}, "SOURCE_CONTRACT_MISMATCH")
    fails(sources, {**recipe, "source_contracts": {}}, "UNRESOLVED_SOURCE_UNITS")
    fails(sources, {**recipe, "steps": [{**step, "provenance": "id"}]}, "OUTPUT_COLLISION")


def test_replacement_sample_requires_draw_key_and_keeps_original_parent_key():
    raw = table(id=[1, 2], weight=[0.0, 1.0], x=[2.0, 3.0])
    step = {"op": "sample", "unit": "row", "groups": [], "strata": [], "weights": "weight", "n": 3, "seed": 1, "replacement": True, "draw_id": "draw", "allow_expand": True, "reason": "Predeclared weighted resampling"}
    fails(raw, {"key": "id", "steps": [step]}, "OBSERVATION_UNIT_CHANGED")
    result = wr.prepare(raw, {"key": "id", "steps": [{**step, "key": ["id", "draw"]}]})
    assert result.data["id"].to_list() == [2, 2, 2]
    assert result.receipt["steps"][0]["observation_transition"]["parent_key"] == ["id"]
    fails(raw, {"key": "id", "steps": [{**step, "key": ["draw"]}]}, "OBSERVATION_UNIT_CHANGED")


@pytest.mark.parametrize("step", [
    {"op": "clean_text", "columns": ["id"]},
    {"op": "cast", "columns": {"id": "Int64"}},
])
def test_identifier_cleanup_requires_explicit_identity_mapping(step):
    raw = table(id=[" 001 ", "002"], x=[2.0, 3.0]) if step["op"] == "clean_text" else table(id=["001", "002"], x=[2.0, 3.0])
    fails(raw, {"key": "id", "steps": [step]}, "KEY_CHANGED")
    result = wr.prepare(raw, {"key": "id", "steps": [{**step, "key": ["id"]}]})
    mapping = result.receipt["steps"][0]["observation_transition"]["identity_map"]
    assert mapping[0]["input"]["id"] == raw["id"][0]
    assert mapping[0]["output"]["id"] == result.data["id"][0]
    assert len(mapping) == 2


def test_identifier_cleanup_cannot_merge_different_observations():
    raw = table(id=["a", " a "], x=[2.0, 3.0])
    fails(raw, {"key": "id", "steps": [{"op": "clean_text", "columns": ["id"], "key": ["id"]}]}, "DUPLICATE_KEY")


@pytest.mark.parametrize("metadata", [{"observation_key": ["x"]}, {"unsafe_allow_key_change": True}, {"unit_column": "units"}, {"allow_expand": "true"}])
def test_unknown_step_metadata_cannot_bypass_identity_or_semantics(metadata):
    raw = table(id=[" a "], x=[2.0])
    fails(raw, {"key": "id", "steps": [{"op": "clean_text", "columns": ["id"], **metadata}]}, {"STEP_FAILED", "INVALID_RECIPE"})


@pytest.mark.parametrize("op", ["join", "join_asof"])
def test_join_suffix_semantics_follow_actual_input_columns(op):
    sources = {"left": table(id=[1], time=[3], x=[10.0]), "right": table(id=[1], time=[2], x=[20.0])}
    step = {"op": "join", "source": "right", "on": "id"} if op == "join" else {"op": "join_asof", "source": "right", "on": "time", "by": ["id"], "strategy": "backward", "tolerance": 2, "exact": True, "ties": "error", "unmatched": "keep"}
    result = wr.prepare(sources, {"input": "left", "key": "id", "source_contracts": {"right": {"units": {"x": "mg"}, "descriptions": {"x": "Right-hand assay"}}}, "steps": [step]})
    output = "x_right" if op == "join" else "x__right"
    assert result.receipt["units"] == {output: "mg"}
    assert result.receipt["variables"][output]["description"] == "Right-hand assay"
    assert result.receipt["variables"]["x"]["unit"] is None
    assert result.receipt["variables"]["x"]["description"] is None


def test_concat_cannot_relabel_known_appended_units_when_input_is_unresolved():
    sources = {"left": table(id=[1], x=[10.0]), "right": table(id=[2], x=[20.0])}
    recipe = {"input": "left", "key": "id", "source_contracts": {"right": {"units": {"x": "mg"}}}, "steps": [{"op": "concat", "sources": ["right"], "schema": "strict", "provenance": "batch", "labels": ["left", "right"], "key": ["batch", "id"], "allow_expand": True, "units": {"x": "g"}}]}
    fails(sources, recipe, {"UNRESOLVED_SOURCE_UNITS", "SOURCE_CONTRACT_MISMATCH", "UNIT_MISMATCH"})


def test_aggregate_reusing_input_name_cannot_inherit_raw_description_or_variance_unit():
    raw = table(id=[1, 2], subject=["a", "a"], x=[10.0, 20.0])
    recipe = {"key": "id", "units": {"x": "mg"}, "descriptions": {"x": "Individual raw assay amount"}, "steps": [{"op": "aggregate", "by": ["subject"], "metrics": {"x": reduction("x", "var", ddof=1)}, "null_keys": "error", "key": ["subject"]}]}
    result = wr.prepare(raw, recipe)
    assert result.receipt["variables"]["x"]["description"] is None
    assert result.receipt["variables"]["x"]["unit"] != "mg"


def test_aggregate_reusing_input_name_requires_a_new_group_description():
    raw = table(id=[1, 2], subject=["a", "a"], x=[10.0, 20.0])
    recipe = {"key": "id", "units": {"x": "mg"}, "descriptions": {"x": "Individual raw assay amount"}, "steps": [{"op": "aggregate", "by": ["subject"], "metrics": {"x": reduction("x", "mean")}, "null_keys": "error", "key": ["subject"]}]}
    result = wr.prepare(raw, recipe)
    assert result.receipt["variables"]["x"]["unit"] == "mg"
    assert result.receipt["variables"]["x"]["description"] is None


@pytest.mark.parametrize("step", [
    {"op": "sort", "by": ["id"], "descending": False, "nulls": "error", "ties": "error", "units": {"x": "g"}},
    {"op": "convert_unit", "column": "x", "from_unit": "mg", "to_unit": "g", "factor": 0.001, "units": {"x": "kg"}},
])
def test_step_metadata_cannot_override_verified_or_computed_units(step):
    raw = table(id=[1, 2], x=[1000.0, 2000.0])
    fails(raw, {"key": "id", "units": {"x": "mg"}, "steps": [step]}, {"UNIT_MISMATCH", "SOURCE_CONTRACT_MISMATCH"})


def test_units_for_a_genuinely_derived_measurement_remain_supported():
    raw = table(id=[1, 2], x=[1000.0, 2000.0])
    result = wr.prepare(raw, {"key": "id", "units": {"x": "mg"}, "steps": [{"op": "derive", "columns": {"x": {"mul": [{"col": "x"}, 0.001]}}, "overwrite": True, "units": {"x": "g"}, "descriptions": {"x": "Derived amount in grams"}}]})
    assert result.data["x"].to_list() == [1.0, 2.0]
    assert result.receipt["units"] == {"x": "g"}


def test_unpivot_unit_column_cannot_be_overridden_by_a_single_measurement_unit():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "unpivot", "index": ["id"], "on": ["mass", "volume"], "variable": "variable", "value": "value", "nulls": "keep", "allow_expand": True, "key": ["id", "variable"], "unit_column": "unit", "units": {"value": "g"}}
    fails(raw, {"key": "id", "units": {"mass": "mg", "volume": "mL"}, "steps": [step]}, {"UNIT_MISMATCH", "SOURCE_CONTRACT_MISMATCH"})


def balanced_join_recipe(**metadata):
    return {"input": "left", "key": "id", "steps": [{"op": "join", "source": "right", "on": "group", "how": "inner", "validate": "1:m", "unmatched": "drop", "key": ["id", "replicate"], **metadata}]}


def balanced_join_sources():
    return {"left": table(id=[1, 2], group=["a", "b"]), "right": table(group=["a", "a"], replicate=[1, 2])}


def test_balanced_join_loss_and_expansion_are_not_hidden_by_equal_row_count():
    fails(balanced_join_sources(), balanced_join_recipe(), {"UNDECLARED_LOSS", "UNDECLARED_EXPANSION"})
    fails(balanced_join_sources(), balanced_join_recipe(reason="Eligible assay replicates"), "UNDECLARED_EXPANSION")
    fails(balanced_join_sources(), balanced_join_recipe(allow_expand=True), "UNDECLARED_LOSS")
    result = wr.prepare(balanced_join_sources(), balanced_join_recipe(reason="Eligible assay replicates", allow_expand=True))
    assert result.data.height == 2
    assert result.receipt["steps"][0]["excluded_keys"] == [{"id": 2}]


def test_union_concat_propagates_semantics_only_for_fields_present_in_each_batch():
    sources = {"left": table(id=[1], mass=[10.0]), "right": table(id=[2], volume=[20.0])}
    recipe = {"input": "left", "key": "id", "units": {"mass": "mg"}, "descriptions": {"mass": "Assay mass"}, "source_contracts": {"right": {"units": {"volume": "mL"}, "descriptions": {"volume": "Assay volume"}}}, "steps": [{"op": "concat", "sources": ["right"], "schema": "union", "provenance": "batch", "labels": ["left", "right"], "key": ["batch", "id"], "allow_expand": True}]}
    result = wr.prepare(sources, recipe)
    assert result.data["mass"].to_list() == [10.0, None]
    assert result.data["volume"].to_list() == [None, 20.0]
    assert result.receipt["units"] == {"mass": "mg", "volume": "mL"}
    assert result.receipt["variables"]["volume"]["description"] == "Assay volume"


def test_step_unit_assertion_can_agree_with_inferred_measurement_units():
    raw = table(id=[1, 2], subject=["a", "a"], x=[10.0, 20.0])
    result = wr.prepare(raw, {"key": "id", "units": {"x": "mg"}, "steps": [{"op": "aggregate", "by": ["subject"], "metrics": {"average": reduction("x", "mean")}, "null_keys": "error", "key": "subject", "units": {"average": "mg"}, "descriptions": {"average": "Mean subject assay amount"}}]})
    assert result.receipt["variables"]["average"]["unit"] == "mg"
    assert result.receipt["variables"]["average"]["description"] == "Mean subject assay amount"


def heterogeneous_measurement_recipe(*steps):
    reshape = {"op": "unpivot", "index": ["id"], "on": ["mass", "volume"], "variable": "variable", "value": "value", "nulls": "keep", "allow_expand": True, "key": ["id", "variable"], "unit_column": "unit"}
    return {"key": "id", "units": {"mass": "mg", "volume": "mL"}, "steps": [reshape, *steps]}


def test_renaming_dynamic_unit_field_updates_the_verified_reference():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    result = wr.prepare(raw, heterogeneous_measurement_recipe({"op": "rename", "columns": {"unit": "measurement_unit"}}))
    assert result.receipt["units"] == {"value": "@measurement_unit"}
    assert result.data["measurement_unit"].to_list() == ["mg", "mL"]


def test_select_cannot_publish_a_dangling_dynamic_unit_reference():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    with pytest.raises(wr.WrangleError):
        wr.prepare(raw, heterogeneous_measurement_recipe({"op": "select", "columns": ["id", "variable", "value"]}))


def test_aggregation_cannot_mix_row_specific_physical_units():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "aggregate", "by": ["id"], "metrics": {"mean": reduction("value", "mean")}, "null_keys": "error", "key": ["id"]}
    with pytest.raises(wr.WrangleError):
        wr.prepare(raw, heterogeneous_measurement_recipe(step))


def test_aggregation_by_dynamic_unit_retains_compatible_measurements():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "aggregate", "by": ["id", "unit"], "metrics": {"mean": reduction("value", "mean")}, "null_keys": "error", "key": ["id", "unit"]}
    result = wr.prepare(raw, heterogeneous_measurement_recipe(step))
    assert result.data["mean"].to_list() == [2.0, 3.0]
    assert result.receipt["units"] == {"mean": "@unit"}


def test_subject_window_cannot_average_distinct_row_specific_units():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "window", "by": ["id"], "order_by": ["variable"], "ties": "error", "metrics": {"rolling": {"column": "value", "method": "rolling_mean", "size": 2, "min_count": 2, "nulls": "keep"}}}
    with pytest.raises(wr.WrangleError):
        wr.prepare(raw, heterogeneous_measurement_recipe(step))


def test_counting_heterogeneous_measurements_is_dimensionless():
    raw = table(id=[1], mass=[2.0], volume=[3.0])
    step = {"op": "aggregate", "by": ["id"], "metrics": {"n": {"column": "value", "method": "count", "nulls": "ignore"}}, "null_keys": "error", "key": ["id"]}
    result = wr.prepare(raw, heterogeneous_measurement_recipe(step))
    assert result.data["n"].to_list() == [2]
    assert result.receipt["units"] == {"n": "1"}


def test_calendar_aggregation_does_not_reuse_an_observed_timestamp_description():
    raw = table(id=[1, 2], time=[date(2024, 1, 10), date(2024, 1, 20)], x=[2.0, 4.0])
    step = {"op": "aggregate", "by": [], "time": "time", "every": "1mo", "period": "1mo", "closed": "left", "label": "left", "metrics": {"mean": reduction("x", "mean")}, "null_keys": "error", "key": ["time"]}
    result = wr.prepare(raw, {"key": "id", "descriptions": {"time": "Observed sample collection date"}, "steps": [step]})
    assert result.data["time"].to_list() == [date(2024, 1, 1)]
    assert result.receipt["variables"]["time"]["description"] is None


def test_asof_preserves_left_timestamp_role_when_right_role_differs():
    sources = {"left": table(id=[1], time=[3]), "right": table(id=[1], time=[2], dose=[20.0])}
    recipe = {"input": "left", "key": "id", "units": {"time": "day"}, "descriptions": {"time": "Sample collection day after baseline"}, "source_contracts": {"right": {"units": {"time": "day", "dose": "mg"}, "descriptions": {"time": "Medication administration day after baseline"}}}, "steps": [{"op": "join_asof", "source": "right", "on": "time", "by": ["id"], "strategy": "backward", "tolerance": 2, "exact": True, "ties": "error", "unmatched": "keep"}]}
    result = wr.prepare(sources, recipe)
    assert result.data["dose"].to_list() == [20.0]
    assert result.receipt["variables"]["time"]["description"] == "Sample collection day after baseline"
    assert result.receipt["variables"]["time"]["unit"] == "day"
