"""Verified instrument batch + specimen metadata workflow; no external files needed."""
from pathlib import Path
import tempfile

import polars as pl
import wrangle as wr


def run(output=None):
    measurements = pl.DataFrame({
        "sample_id": ["001", "002", "003"],
        "mass_mg": ["1000", "2000", "3000"],
        "qc": ["pass", "fail", "pass"],
    })
    metadata = pl.DataFrame({
        "sample_id": ["003", "001", "002"],
        "group": ["treated", "control", "control"],
    })
    recipe = {
        "version": 1,
        "name": "Specimen mass with instrument quality control",
        "input": "measurements",
        "key": "sample_id",
        "units": {"mass_mg": "mg"},
        "descriptions": {"mass_mg": "Specimen mass"},
        "steps": [
            {"op": "cast", "columns": {"mass_mg": "Float64"}},
            {"op": "join", "source": "metadata", "on": "sample_id", "descriptions": {"group": "Study group"}},
            {"op": "filter", "where": {"eq": [{"col": "qc"}, "pass"]}, "reason": "Predefined instrument QC acceptance rule"},
            {"op": "convert_unit", "column": "mass_mg", "from_unit": "mg", "to_unit": "g", "factor": 0.001},
            {"op": "rename", "columns": {"mass_mg": "mass_g"}},
        ],
        "checks": {
            "required": ["group", "mass_g"],
            "ranges": {"mass_g": {"min": 0}},
            "row_count": {"exact": 2},
        },
    }
    sources = {"measurements": measurements, "metadata": metadata}
    profile = wr.inspect(measurements)
    result = wr.prepare(sources, recipe, output=output)
    replay = wr.prepare(sources, recipe)
    assert profile["rows"] == 3
    assert result.data["sample_id"].to_list() == ["001", "003"]
    assert result.data["mass_g"].to_list() == [1.0, 3.0]
    assert result.data["group"].to_list() == ["control", "treated"]
    assert result.receipt["steps"][2]["excluded_keys"] == [{"sample_id": "002"}]
    assert result.receipt["units"] == {"mass_g": "g"}
    assert result.receipt["variables"]["group"]["description"] == "Study group"
    assert result.receipt == replay.receipt
    assert measurements.height == 3
    return result


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="wrangle-example-") as directory:
        result = run(Path(directory) / "batch")
        print(result.data)
        print("Verified: 2 specimens, 1 declared exclusion, identical replay.")
