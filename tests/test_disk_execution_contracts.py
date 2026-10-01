"""Disk checks keep scientific contracts and complete evidence without row collects."""
from __future__ import annotations

import polars as pl
import pytest

from wrangle._contracts import check_data
from wrangle._core import WrangleError
from wrangle._execution import (
    observation_effects,
    observation_transition,
    semantic_effects,
    validate_unit_basis,
    validate_unit_references,
)
from wrangle._storage import DiskTable, execution_context


@pytest.fixture(autouse=True)
def streaming_checks():
    with execution_context("disk"):
        yield


def disk(data, path):
    data.write_parquet(path)
    return DiskTable(pl.scan_parquet(path))


class Evidence:
    """Retain the unmaterialized evidence plan to inspect the recorded contract."""

    def __init__(self):
        self.records = []

    def record(self, plan, kind):
        assert isinstance(plan, pl.LazyFrame)
        self.records.append((plan, kind))
        return {"path": f"evidence/{kind}.parquet", "kind": kind}


def test_disk_research_checks_match_memory(tmp_path):
    data = pl.DataFrame({"id": [1, 2], "subject": ["a", "a"], "value": [1.0, 3.0]})
    parent = pl.DataFrame({"subject": ["a"]})
    rules = {
        "required": ["value"],
        "unique": [["subject", "id"]],
        "schema": {"id": "Int64", "subject": "String", "value": "Float64"},
        "ranges": {"value": {"min": 1, "max": 3}},
        "allowed": {"subject": ["a"]},
        "patterns": {"subject": "^a$"},
        "missing": {"value": {"max": 0}},
        "ordering": {"column": "id", "groups": ["subject"], "ties": "error"},
        "foreign_keys": [{"source": "parent", "columns": ["subject"], "reference": ["subject"]}],
        "group_counts": [{"by": ["subject"], "exact": 2}],
        "row_count": {"exact": 2},
        "protocol": {"key": True, "units": ["value"], "descriptions": ["value"]},
    }
    common = {"units": {"value": "mg"}, "descriptions": {"value": "measurement"}}
    expected = check_data(data, rules, ["id"], sources={"parent": parent}, **common)
    actual = check_data(disk(data, tmp_path / "data.parquet"), rules, ["id"], sources={"parent": disk(parent, tmp_path / "parent.parquet")}, **common)
    assert actual == expected


def test_disk_duplicate_key_keeps_error_details(tmp_path):
    data = pl.DataFrame({"id": [1, 1], "value": [2.0, 3.0]})
    with pytest.raises(WrangleError) as observed:
        check_data(disk(data, tmp_path / "duplicate.parquet"), {}, ["id"])
    assert observed.value.code == "DUPLICATE_KEY"
    assert observed.value.details == {"columns": ["id"], "affected_rows": 2}


def test_disk_exclusions_record_complete_native_plan(tmp_path):
    before = disk(pl.DataFrame({"id": [1, 2, 3], "value": [2, 3, 4]}), tmp_path / "before.parquet")
    after = DiskTable(before.lazy().filter(pl.col("id") == 2))
    evidence = Evidence()
    excluded, counts = observation_effects(before, after, "filter", {"reason": "protocol exclusion"}, ["id"], evidence=evidence)
    assert counts == {"excluded_observations": 2, "expanded_parents": 0, "introduced_observations": 0}
    assert excluded == {"path": "evidence/excluded_keys.parquet", "kind": "excluded_keys"}
    plan, kind = evidence.records[0]
    assert kind == "excluded_keys"
    assert plan.collect().to_dict(as_series=False) == {"id": [1, 3]}


def test_disk_identity_mapping_records_complete_native_plan(tmp_path):
    before = disk(pl.DataFrame({"id": [1, 2], "value": [2, 3]}), tmp_path / "before.parquet")
    after = DiskTable(before.lazy().with_columns((pl.col("id") + 10).alias("new_id")))
    evidence = Evidence()
    relation = observation_transition(before, after, "derive", {"key": ["new_id"]}, ["id"], ["new_id"], evidence=evidence)
    assert relation["identity_map"] == {"path": "evidence/identity_map.parquet", "kind": "identity_map"}
    plan, kind = evidence.records[0]
    assert kind == "identity_map"
    assert plan.collect().to_dicts() == [{"input": {"id": 1}, "output": {"new_id": 11}}, {"input": {"id": 2}, "output": {"new_id": 12}}]


def test_disk_unpivot_unit_column_does_not_collect(monkeypatch, tmp_path):
    before = disk(pl.DataFrame({"id": [1, 2], "mass": [2, 3], "length": [4, 5]}), tmp_path / "before.parquet")
    after = DiskTable(before.lazy().unpivot(on=["mass", "length"], index=["id"], variable_name="variable", value_name="value"))
    def reject(plan):
        pytest.fail("Unit metadata propagation materialized table rows")
    monkeypatch.setattr("wrangle._execution.collect", reject)
    output, units, descriptions, variable_units = semantic_effects(before, after, "unpivot", {"on": ["mass", "length"], "variable": "variable", "value": "value", "unit_column": "unit"}, {"mass": "mg", "length": "cm"}, {})
    assert isinstance(output, DiskTable)
    assert units["value"] == "@unit"
    assert output.lazy().select("variable", "unit").collect().to_dicts() == [{"variable": "mass", "unit": "mg"}, {"variable": "mass", "unit": "mg"}, {"variable": "length", "unit": "cm"}, {"variable": "length", "unit": "cm"}]
    assert variable_units["mapping"] == {"mass": "mg", "length": "cm"}


@pytest.mark.parametrize("key", [[], ["id"]])
def test_disk_unit_label_changes_still_reject(tmp_path, key):
    before = disk(pl.DataFrame({"id": [1, 2], "unit": ["mg", "mg"], "value": [2.0, 3.0]}), tmp_path / "before.parquet")
    after = DiskTable(before.lazy().with_columns(pl.lit("g").alias("unit")))
    validate_unit_references(before, {"value": "@unit"})
    with pytest.raises(WrangleError) as observed:
        validate_unit_basis(before, after, "recode", {"value": "@unit"}, {"value": "@unit"}, key)
    assert observed.value.code == "UNIT_MISMATCH"
