"""Verified disk preparation: output stays lazy, exclusions stay complete."""
from pathlib import Path
import tempfile

import polars as pl
import wrangle as wr


def run():
    with tempfile.TemporaryDirectory(prefix="wrangle-disk-workflow-") as temporary:
        root = Path(temporary)
        source = root / "measurements.csv"
        source.write_text("id,mass_mg\n001,1000\n002,2000\n003,-999\n", encoding="utf-8")
        recipe = {
            "key": ["id"],
            "units": {"mass_mg": "mg"},
            "steps": [
                {"op": "cast", "columns": {"mass_mg": "Float64"}},
                {"op": "normalize_missing", "columns": {"mass_mg": [-999]}},
                {"op": "filter", "where": {"is_not_null": {"col": "mass_mg"}}, "reason": "Protocol excludes unobserved mass."},
                {"op": "convert_unit", "column": "mass_mg", "from_unit": "mg", "to_unit": "g", "factor": 0.001},
                {"op": "rename", "columns": {"mass_mg": "mass_g"}},
            ],
            "checks": {"required": ["mass_g"], "ranges": {"mass_g": {"min": 0}}, "row_count": {"exact": 2}},
        }
        result = wr.prepare(source, recipe, execution="disk", output=root / "prepared")
        assert isinstance(result.data, pl.LazyFrame)
        assert result.data.collect().to_dict(as_series=False) == {"id": ["001", "002"], "mass_g": [1.0, 2.0]}
        assert result.receipt["units"] == {"mass_g": "g"}
        exclusion = result.receipt["steps"][2]["excluded_keys"]
        assert exclusion["rows"] == 1
        assert pl.scan_parquet(root / "prepared" / exclusion["path"]).collect().to_dict(as_series=False) == {"id": ["003"]}
        reused = wr.prepare(root / "prepared", {"checks": {"row_count": {"exact": 2}}}, execution="disk", output=root / "reused")
        assert reused.receipt["output"] == result.receipt["output"]
        assert reused.receipt["sources"]["data"]["parent"]["units"] == {"mass_g": "g"}
        return result.receipt["output"]


if __name__ == "__main__":
    run()
