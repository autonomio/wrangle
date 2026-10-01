"""Executable observation workflow: summaries, bounded groups, and schema drift.

Requires only installed Wrangle and Polars. Generated local source files remain
unchanged. Binary JSON examples are hexadecimal displays; source dtypes remain
Binary. Nonfinite floats display as JSON null, with separate observed counts.
"""
from pathlib import Path
import json
import tempfile

import polars as pl
import wrangle as wr


def run():
    current = pl.DataFrame({
        "sample_id": ["001", "002", "003", "004"],
        "group": ["treated", "control", "treated", "control"],
        "mass_mg": [1.0, None, 3.0, 5.0],
        "instrument_bytes": [b"\x00\xff", None, b"\x01", b"\x02"],
    })
    baseline = pl.DataFrame({"sample_id": ["001"], "mass_mg": [1], "retired_flag": [True]})
    original = current.clone()
    with tempfile.TemporaryDirectory(prefix="wrangle-inspection-") as directory:
        path = Path(directory) / "current batch.parquet"
        reference = Path(directory) / "prior batch.parquet"
        current.write_parquet(path)
        baseline.write_parquet(reference)
        unchanged = {file: file.read_bytes() for file in (path, reference)}
        result = wr.inspect(path, columns=["sample_id", "mass_mg", "instrument_bytes"], summary=True, groups=["group"], max_groups=1, sample_rows=2, baseline=reference)
        assert {file: file.read_bytes() for file in unchanged} == unchanged
    assert current.equals(original)
    assert result["rows"] == 4
    assert result["columns"]["instrument_bytes"] == "Binary"
    assert result["examples"] == [
        {"sample_id": "001", "mass_mg": 1.0, "instrument_bytes": "00ff"},
        {"sample_id": "002", "mass_mg": None, "instrument_bytes": None},
    ]
    mass = next(field for field in result["fields"] if field["name"] == "mass_mg")
    assert mass["nulls"] == 1
    assert mass["summary"] == {"status": "resolved", "count": 3, "min": 1.0, "max": 5.0, "mean": 3.0, "median": 3.0, "std": 2.0}
    assert result["summary_policy"]["ddof"] == 1
    assert result["groups"]["count"] == 2
    assert result["groups"]["truncated"] is True
    assert len(result["groups"]["items"]) == 1
    assert result["groups"]["items"][0]["values"] == {"group": "treated"}
    assert result["groups"]["items"][0]["rows"] == 2
    assert result["schema_changes"] == {
        "added": [{"name": "group", "dtype": "String"}, {"name": "instrument_bytes", "dtype": "Binary"}],
        "removed": [{"name": "retired_flag", "dtype": "Boolean"}],
        "type_changed": [{"name": "mass_mg", "before": "Int64", "after": "Float64"}],
    }
    assert result["example_encoding"]["binary_columns"] == ["instrument_bytes"]
    return result


if __name__ == "__main__":
    observed = run()
    print(json.dumps({"rows": observed["rows"], "groups": observed["groups"]["count"], "groups_truncated": observed["groups"]["truncated"], "mean_mass_mg": 3.0, "sources_unchanged": True}, sort_keys=True))
