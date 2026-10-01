"""Fit explicit preparation statistics once; replay them on an independent batch.

Reference data and its recipe remain reviewable. Receipts contain the actual
replacements/means/scales; replay embeds that JSON state and never refits it.
This example builds no predictive model and selects no method automatically.
"""
from copy import deepcopy
import json

import polars as pl
import wrangle as wr


def run():
    reference = pl.DataFrame({
        "specimen": ["001", "002", "003"],
        "mass_g": [1.0, 3.0, None],
        "group": ["treated", "control", None],
        "lab": ["north", None, "north"],
    })
    incoming = pl.DataFrame({
        "specimen": ["004", "005"],
        "mass_g": [5.0, None],
        "group": [None, "treated"],
        "lab": [None, "south"],
    })
    reference_before, incoming_before = reference.clone(), incoming.clone()
    recipe = {
        "version": 1, "name": "Reference cohort preparation", "key": ["specimen"],
        "units": {"mass_g": "g"}, "descriptions": {"mass_g": "Measured specimen mass"},
        "steps": [
            {"op": "fill", "columns": {"lab": "north"}},
            {"op": "impute", "columns": ["mass_g"], "method": "mean"},
            {"op": "impute", "columns": ["group"], "method": "mode"},
            {"op": "encode", "column": "group", "mode": "one_hot", "categories": ["control", "treated"], "names": ["control", "treated"], "drop": False, "nulls": "error"},
            {"op": "standardize", "columns": ["mass_g"], "ddof": 1},
        ],
        "checks": {"required": ["mass_g", "group", "lab"], "protocol": {"key": True, "units": ["mass_g"]}},
    }
    fitted = wr.prepare(reference, recipe)
    assert fitted.data["mass_g"].to_list() == [-1.0, 1.0, 0.0]
    assert fitted.receipt["steps"][1]["parameters"]["parameters"]["mass_g"] == {
        "dtype": "Float64", "count": 2, "method": "mean", "replacement": 2.0, "unit": "g",
    }
    assert fitted.receipt["steps"][4]["parameters"]["parameters"]["mass_g"] == {
        "dtype": "Float64", "count": 3, "mean": 2.0, "std": 1.0, "ddof": 1, "unit": "g",
    }
    frozen_recipe = deepcopy(recipe)
    frozen_recipe["name"] = "Independent batch using reference cohort parameters"
    frozen_recipe["steps"] = [
        {"op": step["op"], **step["parameters"]} for step in fitted.receipt["steps"]
    ]
    frozen_recipe = json.loads(json.dumps(frozen_recipe, allow_nan=False))
    applied = wr.prepare(incoming, frozen_recipe)
    replay = wr.prepare(incoming, frozen_recipe)
    assert applied.data["mass_g"].to_list() == [3.0, 0.0]
    assert applied.data["group"].to_list() == ["control", "treated"]
    assert applied.data["control"].to_list() == [1, 0]
    assert applied.data["treated"].to_list() == [0, 1]
    assert applied.data["lab"].to_list() == ["north", "south"]
    assert applied.data.schema == fitted.data.schema
    for index in (1, 2, 4):
        assert applied.receipt["steps"][index]["parameters"] == fitted.receipt["steps"][index]["parameters"]
    assert applied.receipt["units"] == {"mass_g": "1"}
    assert applied.receipt["variables"]["mass_g"]["description"] is None
    assert applied.receipt == replay.receipt
    assert reference.equals(reference_before) and incoming.equals(incoming_before)
    return {"reference": fitted, "incoming": applied, "frozen_recipe": frozen_recipe}


if __name__ == "__main__":
    results = run()
    print(results["incoming"].data)
    print("Verified: reference replacements and scales reused; batches unchanged.")
