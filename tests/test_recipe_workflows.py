"""Verify public preparation workflows and scientific failure boundaries."""
from pathlib import Path
import runpy

import polars as pl
import pytest
import wrangle as wr
from wrangle.docs import preparation_workflows as workflows


def recipe(steps, *, key="id", **declarations):
    return {"version": 1, "key": key, "steps": steps, **declarations}


def fails(code, sources, protocol):
    with pytest.raises(wr.WrangleError) as captured:
        wr.prepare(sources, protocol)
    assert captured.value.code == code
    return captured.value


def test_installed_package_workflows_verify_every_new_preparation_operation():
    results = workflows.run()
    covered = {step["op"] for result in results.values() for step in result.receipt["steps"]}
    assert covered >= {
        "normalize_missing", "clean_text", "parse_datetime", "recode", "encode",
        "sort", "deduplicate", "concat", "pivot", "unpivot", "explode", "unnest",
        "aggregate", "join_asof", "window", "sample", "partition",
    }
    assert results["long"].receipt["key"] == ["specimen", "assay"]
    assert results["wide"].receipt["key"] == ["specimen"]
    assert results["summary"].receipt["key"] == ["assay"]
    assert results["resampled"].receipt["key"] == ["draw", "visit"]


def test_checkout_example_is_identical_to_installed_agent_example():
    repository = Path(__file__).resolve().parents[1]
    installed = Path(wr.__file__).parent / "docs" / "preparation_workflows.py"
    assert installed.read_bytes() == (repository / "examples" / "preparation_workflows.py").read_bytes()
    assert callable(runpy.run_path(str(installed))["run"])


def test_fixed_one_hot_schema_and_meaning_replay_across_independent_batches():
    protocol = recipe([{"op": "encode", "column": "group", "mode": "one_hot", "categories": ["control", "treated"], "names": ["control", "treated"], "nulls": "preserve"}])
    first = wr.prepare(pl.DataFrame({"id": ["001", "002"], "group": ["treated", None]}), protocol)
    second = wr.prepare(pl.DataFrame({"id": ["003"], "group": ["control"]}), protocol)
    assert first.data.schema == second.data.schema
    assert first.data.to_dict(as_series=False) == {"id": ["001", "002"], "control": [0, None], "treated": [1, None]}
    assert second.data.to_dict(as_series=False) == {"id": ["003"], "control": [1], "treated": [0]}
    error = fails("UNKNOWN_CATEGORY", pl.DataFrame({"id": ["004"], "group": ["new cohort"]}), protocol)
    assert error.details["operation"] == "encode"


def test_cleaning_an_observation_identifier_cannot_change_identity():
    protocol = recipe([{"op": "clean_text", "columns": ["id"], "case": "lower"}])
    fails("KEY_CHANGED", pl.DataFrame({"id": ["S1"], "x": [1.0]}), protocol)


def test_explicit_concat_key_still_rejects_duplicate_observation_identities():
    sources = {"first": pl.DataFrame({"id": ["001"], "x": [1.0]}), "second": pl.DataFrame({"id": ["001"], "x": [2.0]})}
    protocol = recipe([{"op": "concat", "sources": ["second"], "schema": "strict", "provenance": "batch", "labels": ["first", "second"], "allow_expand": True, "key": ["id"]}], input="first")
    fails("DUPLICATE_KEY", sources, protocol)


def test_secondary_sources_are_bound_by_name_and_never_serialized_as_tables():
    sources = {"first": pl.DataFrame({"id": ["001"], "x": [1.0]}), "second": pl.DataFrame({"id": ["002"], "x": [2.0]})}
    step = {"op": "concat", "sources": ["second"], "schema": "strict", "provenance": "batch", "labels": ["first", "second"], "allow_expand": True, "key": ["id"]}
    result = wr.prepare(sources, recipe([step], input="first"))
    assert result.receipt["steps"][0]["parameters"]["sources"] == ["second"]
    step["sources"] = ["absent"]
    fails("UNKNOWN_SOURCE", sources, recipe([step], input="first"))


def test_unpivot_requires_deliberate_expansion_and_an_observed_composite_key():
    source = pl.DataFrame({"id": ["001", "002"], "a": [1.0, 2.0], "b": [3.0, 4.0]})
    step = {"op": "unpivot", "index": ["id"], "on": ["a", "b"], "variable": "assay", "value": "amount", "nulls": "keep", "key": ["id", "assay"]}
    fails("UNDECLARED_EXPANSION", source, recipe([step]))
    step["allow_expand"] = True
    result = wr.prepare(source, recipe([step]))
    assert result.data.select("id", "assay").to_dict(as_series=False) == {"id": ["001", "001", "002", "002"], "assay": ["a", "b", "a", "b"]}
    step["key"] = ["id"]
    fails("DUPLICATE_KEY", source, recipe([step]))


def test_missing_reshape_measurements_remain_observations_without_exclusion():
    source = pl.DataFrame({"id": ["001"], "a": [1.0], "b": pl.Series([None], dtype=pl.Float64)})
    step = {"op": "unpivot", "index": ["id"], "on": ["a", "b"], "variable": "assay", "value": "amount", "nulls": "keep", "key": ["id", "assay"], "allow_expand": True}
    result = wr.prepare(source, recipe([step]))
    assert result.data["amount"].to_list() == [1.0, None]
    assert result.receipt["steps"][0]["excluded_keys"] is None


def test_deduplicating_raw_measurements_requires_an_exclusion_reason():
    source = pl.DataFrame({"id": ["001", "001"], "revision": [1, 2], "x": [1.0, 2.0]})
    step = {"op": "deduplicate", "by": ["id"], "keep": "last", "conflicts": "allow", "order_by": ["revision"], "key": ["id"]}
    fails("UNDECLARED_LOSS", source, recipe([step], key=[]))
    step["reason"] = "Latest validated instrument revision"
    assert wr.prepare(source, recipe([step], key=[])).data["x"].to_list() == [2.0]


def test_pivot_reduction_requires_declared_output_identity_and_fixed_domain():
    source = pl.DataFrame({"id": ["001", "001"], "assay": ["a", "b"], "x": [1.0, 2.0]})
    step = {"op": "pivot", "index": ["id"], "on": "assay", "values": ["x"], "domain": ["a", "b"], "duplicate_cells": "error", "missing_cells": "null", "key": ["id"]}
    result = wr.prepare(source, recipe([step], key=["id", "assay"]))
    assert result.data.to_dict(as_series=False) == {"id": ["001"], "x__a": [1.0], "x__b": [2.0]}
    assert result.receipt["steps"][0]["reason"] is None
    step["domain"] = ["a"]
    fails("PIVOT_DOMAIN", source, recipe([step], key=["id", "assay"]))


def test_subject_temporal_matching_never_borrows_other_subjects():
    sources = {"visits": pl.DataFrame({"id": ["a1"], "subject": ["a"], "time": [3]}), "exposure": pl.DataFrame({"subject": ["b"], "time": [2], "dose": [9.0]})}
    step = {"op": "join_asof", "source": "exposure", "on": "time", "by": ["subject"], "strategy": "backward", "tolerance": 2, "exact": True, "ties": "error", "unmatched": "error"}
    fails("UNMATCHED_ROWS", sources, recipe([step], input="visits"))
    step["unmatched"] = "keep"
    assert wr.prepare(sources, recipe([step], input="visits")).data["dose"].to_list() == [None]


def test_subject_sampling_needs_constant_stratum_within_each_subject():
    source = pl.DataFrame({"id": ["a1", "a2"], "subject": ["a", "a"], "region": ["north", "south"]})
    step = {"op": "sample", "unit": "group", "groups": ["subject"], "strata": ["region"], "n": 1, "seed": 11, "replacement": False, "reason": "Predeclared subject subset"}
    fails("GROUP_CONFLICT", source, recipe([step]))


def test_replacement_sampling_requires_new_draw_identity_even_when_row_count_shrinks():
    source = pl.DataFrame({"id": ["a1", "a2", "b1", "b2"], "subject": ["a", "a", "b", "b"], "weight": [0.0, 0.0, 1.0, 1.0]})
    step = {"op": "sample", "unit": "group", "groups": ["subject"], "strata": [], "n": 1, "seed": 11, "replacement": True, "weights": "weight", "draw_id": "draw", "key": ["draw", "id"], "reason": "Predeclared one-subject weighted draw"}
    result = wr.prepare(source, recipe([step]))
    assert result.receipt["key"] == ["draw", "id"]
    assert result.data["id"].to_list() == ["b1", "b2"]


@pytest.mark.parametrize("kind", ["unpivot", "pivot", "aggregate", "explode", "concat", "replacement"])
def test_new_observation_grain_always_requires_an_explicit_output_key(kind):
    source = pl.DataFrame({"id": ["001"], "a": [1.0], "b": [2.0]})
    if kind == "unpivot":
        step = {"op": "unpivot", "index": ["id"], "on": ["a", "b"], "variable": "assay", "value": "x", "nulls": "keep", "allow_expand": True}
    elif kind == "pivot":
        source = pl.DataFrame({"id": ["001"], "assay": ["a"], "x": [1.0]})
        step = {"op": "pivot", "index": ["id"], "on": "assay", "values": ["x"], "domain": ["a"], "duplicate_cells": "error", "missing_cells": "null"}
    elif kind == "aggregate":
        step = {"op": "aggregate", "by": ["id"], "metrics": {"n": {"method": "len"}}, "null_keys": "error"}
    elif kind == "explode":
        source = pl.DataFrame({"id": ["001"], "x": [[1.0]]})
        step = {"op": "explode", "columns": ["x"], "index": "replicate", "empty": "error", "nulls": "error", "allow_expand": True}
    elif kind == "concat":
        source = {"data": source, "extra": pl.DataFrame({"id": ["002"], "a": [2.0], "b": [3.0]})}
        step = {"op": "concat", "sources": ["extra"], "schema": "strict", "provenance": "batch", "labels": ["first", "second"], "allow_expand": True}
    else:
        step = {"op": "sample", "unit": "row", "groups": [], "strata": [], "n": 1, "seed": 11, "replacement": True, "draw_id": "draw"}
    protocol = recipe([step], input="data") if kind == "concat" else recipe([step])
    fails("OBSERVATION_UNIT_CHANGED", source, protocol)


def test_heterogeneous_unpivot_requires_variable_units_to_retain_meaning():
    source = pl.DataFrame({"id": ["001"], "mass": [1.0], "temperature": [20.0]})
    step = {"op": "unpivot", "index": ["id"], "on": ["mass", "temperature"], "variable": "measurement", "value": "value", "nulls": "keep", "allow_expand": True, "key": ["id", "measurement"]}
    protocol = recipe([step], units={"mass": "kg", "temperature": "degC"})
    fails("UNIT_MISMATCH", source, protocol)
    step["unit_column"] = "unit"
    result = wr.prepare(source, protocol)
    assert result.data["unit"].to_list() == ["kg", "degC"]
    assert result.receipt["units"]["value"] == "@unit"
    assert result.receipt["variable_units"][0]["mapping"] == {"mass": "kg", "temperature": "degC"}


def test_verified_parent_lineage_retains_the_original_protocol_digest():
    results = workflows.reshape_and_summarize()
    parent = results["wide"].receipt["sources"]["data"]["parent"]
    assert parent["recipe_sha256"] == results["long"].receipt["recipe_sha256"]
    assert parent["key"] == ["specimen", "assay"]
    assert parent["units"] == {"concentration": "mmol/L"}


def test_heterogeneous_measurements_cannot_be_averaged_under_one_specimen_key():
    source = pl.DataFrame({"id": ["001"], "mass": [1.0], "temperature": [20.0]})
    long = wr.prepare(source, recipe([{"op": "unpivot", "index": ["id"], "on": ["mass", "temperature"], "variable": "measurement", "value": "value", "nulls": "keep", "allow_expand": True, "key": ["id", "measurement"], "unit_column": "unit"}], units={"mass": "kg", "temperature": "degC"}))
    reduction = {"op": "aggregate", "by": ["id"], "metrics": {"mean": {"column": "value", "method": "mean", "nulls": "ignore", "min_count": 1}}, "null_keys": "error", "key": ["id"]}
    fails("UNIT_MISMATCH", long, {"steps": [reduction]})
