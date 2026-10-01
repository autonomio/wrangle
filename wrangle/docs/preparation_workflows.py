"""Executable scientific preparation protocols; run this installed-package file.

Each function contains input tables, a Python recipe mapping, and expected observations.
Keys describe the current observation unit; reshape/reduction steps declare the
new key. No example infers identifiers, category domains, missing codes or units.
"""
from datetime import datetime

import polars as pl
import wrangle as wr


def _execute(sources, recipe):
    result = wr.prepare(sources, recipe)
    replay = wr.prepare(sources, recipe)
    assert result.data.equals(replay.data)
    assert result.receipt == replay.receipt
    return result


def routine_fields():
    """Explicit missing codes, selected text, timestamp format and fixed categories."""
    source = pl.DataFrame({
        "specimen": ["003", "001", "002"],
        "mass_mg": ["3000", "1000", "-99"],
        "group": [" Control ", "TREATED", "control"],
        "collected": ["30/09/2026 09:00", "28/09/2026 09:00", "29/09/2026 09:00"],
    })
    recipe = {
        "version": 1, "name": "Specimen field preparation", "key": ["specimen"],
        "units": {"mass_mg": "mg"},
        "steps": [
            {"op": "normalize_missing", "columns": {"mass_mg": ["-99"]}, "units": {"mass_mg": "mg"}},
            {"op": "cast", "columns": {"mass_mg": "Float64"}},
            {"op": "clean_text", "columns": ["group"], "trim": True, "case": "lower", "unicode": "NFC"},
            {"op": "parse_datetime", "columns": ["collected"], "format": "%d/%m/%Y %H:%M", "time_zone": "UTC"},
            {"op": "recode", "column": "group", "mapping": {"control": "reference", "treated": "intervention"}, "dtype": "String", "domain": ["reference", "intervention"]},
            {"op": "encode", "column": "group", "mode": "one_hot", "categories": ["reference", "intervention", "unassigned"], "names": ["reference", "intervention", "unassigned"], "nulls": "error"},
            {"op": "convert_unit", "column": "mass_mg", "from_unit": "mg", "to_unit": "g", "factor": 0.001},
            {"op": "rename", "columns": {"mass_mg": "mass_g"}},
            {"op": "sort", "by": ["specimen"], "descending": False, "nulls": "error", "ties": "error"},
        ],
        "checks": {
            "patterns": {"specimen": "^[0-9]{3}$"},
            "ranges": {"mass_g": {"min": 0}}, "missing": {"mass_g": {"max": 0.34}},
            "row_count": {"exact": 3}, "protocol": {"key": True, "units": ["mass_g"]},
        },
    }
    result = _execute(source, recipe)
    assert result.data["specimen"].to_list() == ["001", "002", "003"]
    assert result.data["mass_g"].to_list() == [1.0, None, 3.0]
    assert result.data["reference"].to_list() == [0, 1, 1]
    assert result.data["intervention"].to_list() == [1, 0, 0]
    assert result.data["unassigned"].to_list() == [0, 0, 0]
    assert result.data.schema["collected"] == pl.Datetime("us", "UTC")
    assert result.receipt["units"] == {"mass_g": "g"}
    assert source["mass_mg"].to_list() == ["3000", "1000", "-99"]
    return result


def batches():
    """Append batches with declared union fields and an explicit provenance column."""
    sources = {
        "morning": pl.DataFrame({"specimen": ["001"], "mass_g": [1.0]}),
        "evening": pl.DataFrame({"specimen": ["002"], "mass_g": [2.0], "temperature": [20.0]}),
    }
    recipe = {
        "version": 1, "name": "Instrument batch union", "input": "morning", "key": ["specimen"],
        "units": {"mass_g": "g"},
        "source_contracts": {"evening": {"units": {"mass_g": "g", "temperature": "degC"}}},
        "steps": [{"op": "concat", "sources": ["evening"], "schema": "union", "provenance": "batch", "labels": ["morning", "evening"], "allow_expand": True, "key": ["specimen"], "units": {"temperature": "degC"}}],
        "checks": {"row_count": {"exact": 2}, "allowed": {"batch": ["morning", "evening"]}},
    }
    result = _execute(sources, recipe)
    assert result.data.to_dict(as_series=False) == {
        "specimen": ["001", "002"], "mass_g": [1.0, 2.0], "temperature": [None, 20.0], "batch": ["morning", "evening"],
    }
    assert result.receipt["steps"][0]["parameters"]["sources"] == ["evening"]
    return result


def duplicate_resolution():
    """Establish an identity after resolving raw duplicate observations explicitly."""
    source = pl.DataFrame({"specimen": ["001", "001", "002"], "revision": [1, 2, 1], "mass_g": [1.0, 1.2, 2.0]})
    recipe = {
        "version": 1, "name": "Keep latest instrument revision", "units": {"mass_g": "g"},
        "steps": [{"op": "deduplicate", "by": ["specimen"], "keep": "last", "conflicts": "allow", "order_by": ["revision"], "key": ["specimen"], "reason": "Instrument revision supersedes earlier reading"}],
        "checks": {"row_count": {"exact": 2}},
    }
    result = _execute(source, recipe)
    assert result.data["mass_g"].to_list() == [1.2, 2.0]
    assert result.receipt["key"] == ["specimen"]
    return result


def reshape_and_summarize():
    """Change between specimen and specimen-assay units, then summarize assays."""
    source = pl.DataFrame({"specimen": ["001", "002"], "sodium": [1.0, 3.0], "chloride": [2.0, None]})
    long_recipe = {
        "version": 1, "name": "Specimen-assay observations", "key": ["specimen"],
        "units": {"sodium": "mmol/L", "chloride": "mmol/L"},
        "steps": [{"op": "unpivot", "index": ["specimen"], "on": ["sodium", "chloride"], "variable": "assay", "value": "concentration", "nulls": "keep", "allow_expand": True, "key": ["specimen", "assay"], "units": {"concentration": "mmol/L"}}],
    }
    long = _execute(source, long_recipe)
    assert long.data["concentration"].to_list() == [1.0, 2.0, 3.0, None]
    wide_recipe = {
        "version": 1, "name": "Return to specimen observations", "key": ["specimen", "assay"],
        "units": {"concentration": "mmol/L"},
        "steps": [{"op": "pivot", "index": ["specimen"], "on": "assay", "values": ["concentration"], "domain": ["sodium", "chloride"], "duplicate_cells": "error", "missing_cells": "null", "key": ["specimen"], "units": {"concentration__sodium": "mmol/L", "concentration__chloride": "mmol/L"}}],
    }
    wide = _execute(long, wide_recipe)
    assert wide.data["concentration__sodium"].to_list() == [1.0, 3.0]
    assert wide.data["concentration__chloride"].to_list() == [2.0, None]
    summary_recipe = {
        "version": 1, "name": "Observed assay means", "key": ["specimen", "assay"],
        "units": {"concentration": "mmol/L"},
        "steps": [{"op": "aggregate", "by": ["assay"], "metrics": {
            "n_observed": {"column": "concentration", "method": "count", "nulls": "ignore"},
            "mean": {"column": "concentration", "method": "mean", "nulls": "ignore", "min_count": 1},
        }, "null_keys": "error", "key": ["assay"], "units": {"mean": "mmol/L", "n_observed": "1"}}],
    }
    summary = _execute(long, summary_recipe)
    assert summary.data["assay"].to_list() == ["sodium", "chloride"]
    assert summary.data["n_observed"].to_list() == [2, 1]
    assert summary.data["mean"].to_list() == [2.0, 2.0]
    assert summary.receipt["steps"][0]["excluded_keys"] is None
    return {"long": long, "wide": wide, "summary": summary}


def nested_fields():
    """Expose declared struct fields and identify each replicate in aligned lists."""
    source = pl.DataFrame({"specimen": ["001", "002"], "metadata": [{"lab": "north"}, {"lab": "south"}], "mass_g": [[1.0, 1.1], [2.0]]})
    recipe = {
        "version": 1, "name": "List-valued replicate measurements", "key": ["specimen"],
        "steps": [
            {"op": "unnest", "columns": ["metadata"], "separator": "__"},
            {"op": "explode", "columns": ["mass_g"], "index": "replicate", "empty": "error", "nulls": "error", "allow_expand": True, "key": ["specimen", "replicate"], "units": {"mass_g": "g"}},
        ],
    }
    result = _execute(source, recipe)
    assert result.data.to_dict(as_series=False) == {"specimen": ["001", "001", "002"], "mass_g": [1.0, 1.1, 2.0], "metadata__lab": ["north", "north", "south"], "replicate": [0, 1, 0]}
    return result


def temporal_alignment():
    """Within-subject windows and preceding exposure, with explicit tolerance."""
    d1, d2 = datetime(2026, 9, 28), datetime(2026, 9, 29)
    source = pl.DataFrame({"visit": ["b2", "a2", "a1", "b1"], "subject": ["b", "a", "a", "b"], "time": [d2, d2, d1, d1], "mass_g": [12.0, 20.0, 10.0, 8.0]})
    exposure = pl.DataFrame({"subject": ["a", "b"], "time": [d1, d1], "dose_mg": [5.0, 2.0]})
    recipe = {
        "version": 1, "name": "Subject-local longitudinal context", "input": "visits", "key": ["visit"], "units": {"mass_g": "g"},
        "source_contracts": {"exposure": {"units": {"dose_mg": "mg"}}},
        "steps": [
            {"op": "window", "by": ["subject"], "order_by": ["time"], "metrics": {"previous_mass": {"column": "mass_g", "method": "lag", "n": 1, "nulls": "keep"}, "mean_mass": {"column": "mass_g", "method": "rolling_mean", "size": 2, "min_count": 2, "nulls": "keep"}}, "ties": "error", "units": {"previous_mass": "g", "mean_mass": "g"}},
            {"op": "join_asof", "source": "exposure", "on": "time", "by": ["subject"], "strategy": "backward", "tolerance": "1d", "exact": True, "ties": "error", "unmatched": "error", "units": {"dose_mg": "mg"}},
        ],
    }
    result = _execute({"visits": source, "exposure": exposure}, recipe)
    assert result.data["visit"].to_list() == ["b2", "a2", "a1", "b1"]
    assert result.data["previous_mass"].to_list() == [8.0, 10.0, None, None]
    assert result.data["mean_mass"].to_list() == [10.0, 15.0, None, None]
    assert result.data["dose_mg"].to_list() == [2.0, 5.0, 5.0, 2.0]
    daily_recipe = {
        "version": 1, "name": "Subject-day observations", "key": ["visit"],
        "steps": [{"op": "aggregate", "by": ["subject"], "time": "time", "every": "1d", "period": "1d", "closed": "left", "label": "left", "metrics": {"mean_mass": {"column": "mass_g", "method": "mean", "nulls": "ignore", "min_count": 1}}, "null_keys": "error", "key": ["subject", "time"], "units": {"mean_mass": "g"}}],
    }
    daily = _execute(result, daily_recipe)
    assert daily.data["mean_mass"].to_list() == [10.0, 20.0, 8.0, 12.0]
    return {"longitudinal": result, "daily": daily}


def sampling_and_partitions():
    """Select whole subjects and partition subjects without splitting their visits."""
    source = pl.DataFrame({"visit": list(range(8)), "subject": ["a", "a", "b", "b", "c", "c", "d", "d"], "region": ["north"] * 4 + ["south"] * 4})
    sampling = {
        "version": 1, "name": "Stratified subject subsample", "key": ["visit"],
        "steps": [{"op": "sample", "unit": "group", "groups": ["subject"], "strata": ["region"], "seed": 11, "replacement": False, "n": 1, "reason": "Predeclared one-subject-per-region pilot subset"}],
    }
    sampled = _execute(source, sampling)
    assert sampled.data.height == 4
    assert sampled.data.group_by("subject").len()["len"].to_list() == [2, 2]
    assert sampled.data["region"].to_list() == ["north", "north", "south", "south"]
    assert len(sampled.receipt["steps"][0]["excluded_keys"]) == 4
    partitioning = {
        "version": 1, "name": "Subject-disjoint review batches", "key": ["visit"],
        "steps": [{"op": "partition", "output": "review_batch", "labels": ["first", "second"], "groups": ["subject"], "strata": ["region"], "method": "random", "fractions": [0.5, 0.5], "seed": 11}],
    }
    partitioned = _execute(source, partitioning)
    assert partitioned.data.height == 8
    assert partitioned.data.group_by("subject").agg(pl.col("review_batch").n_unique())["review_batch"].to_list() == [1] * 4
    assert partitioned.data.filter(pl.col("review_batch") == "first").height == 4
    replacement_source = pl.DataFrame({"visit": [1, 2, 3, 4], "subject": ["a", "a", "b", "b"], "weight": [0.0, 0.0, 1.0, 1.0]})
    resampling = {
        "version": 1, "name": "Whole-subject repeated draws", "key": ["visit"],
        "steps": [{"op": "sample", "unit": "group", "groups": ["subject"], "strata": [], "seed": 11, "replacement": True, "weights": "weight", "n": 3, "draw_id": "draw", "key": ["draw", "visit"], "allow_expand": True, "reason": "Predeclared weighted subject resampling"}],
    }
    resampled = _execute(replacement_source, resampling)
    assert resampled.data["visit"].to_list() == [3, 4, 3, 4, 3, 4]
    assert resampled.data["draw"].to_list() == [0, 0, 1, 1, 2, 2]
    return {"sampled": sampled, "partitioned": partitioned, "resampled": resampled}


def run():
    results = {"fields": routine_fields(), "batches": batches(), "duplicates": duplicate_resolution(), "nested": nested_fields()}
    results.update(reshape_and_summarize())
    results.update(temporal_alignment())
    results.update(sampling_and_partitions())
    return results


if __name__ == "__main__":
    results = run()
    print(f"Verified {len(results)} scientific preparation results; exact replay receipts match.")
