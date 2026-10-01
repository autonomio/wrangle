"""Integration contracts for disk preparation, publication and engine parity."""
import json
import subprocess
import sys

import polars as pl
import pytest

import wrangle as wr
from wrangle._api import _digest
from wrangle._storage import DiskWorkspace, execution_context


def test_disk_preparation_never_collects_complete_observations(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    pl.DataFrame({"id": [str(i).zfill(6) for i in range(2000)], "x": list(range(2000))}).write_csv(source)
    original = pl.LazyFrame.collect
    observed = []

    def reduced_only(plan, *args, **kwargs):
        table = original(plan, *args, **kwargs)
        observed.append(table.height)
        assert table.height <= 2, "Disk preparation materialized observation rows."
        return table

    monkeypatch.setattr(pl.LazyFrame, "collect", reduced_only)
    result = wr.prepare(source, {
        "key": ["id"],
        "steps": [
            {"op": "cast", "columns": {"x": "Int64"}},
            {"op": "filter", "where": {"lt": [{"col": "x"}, 1000]}, "reason": "Declared cohort."},
            {"op": "derive", "columns": {"y": {"mul": [{"col": "x"}, 2]}}},
        ],
        "checks": {"required": ["x", "y"], "row_count": {"exact": 1000}, "assertions": [{"name": "double", "where": {"eq": [{"col": "y"}, {"mul": [{"col": "x"}, 2]}]}}]},
    }, execution="disk", output=tmp_path / "result")
    assert observed and result.receipt["output"]["rows"] == 1000
    reference = result.receipt["steps"][1]["excluded_keys"]
    assert reference["rows"] == 1000
    assert len(json.dumps(result.receipt)) < 12000
    assert isinstance(result.data, pl.LazyFrame)
    assert original(result.data).height == 1000
    assert original(pl.scan_parquet(tmp_path / "result" / reference["path"])).height == 1000
    assert not list(tmp_path.glob(".wrangle*"))
    assert not list(tmp_path.glob("*.wrangle-lock"))


def test_disk_and_memory_keep_checks_order_and_logical_snapshots(tmp_path):
    source = pl.DataFrame({"id": ["002", "001", "003"], "x": [2., None, 4.]})
    recipe = {"key": ["id"], "units": {"x": "m"}, "steps": [
        {"op": "fill", "columns": {"x": 0.}},
        {"op": "sort", "by": ["id"], "descending": False, "nulls": "last", "ties": "source"},
        {"op": "filter", "where": {"gt": [{"col": "x"}, 0]}, "reason": "Declared positive measurements."},
    ], "checks": {"required": ["x"], "ranges": {"x": {"min": 0}}, "ordering": {"column": "id"}}}
    memory = wr.prepare(source, recipe)
    disk = wr.prepare(source, recipe, execution="disk", output=tmp_path / "result")
    assert disk.data.collect().equals(memory.data)
    assert disk.receipt["output"] == memory.receipt["output"]
    assert disk.receipt["checks"] == memory.receipt["checks"]
    assert disk.receipt["units"] == memory.receipt["units"]
    for old, new in zip(memory.receipt["steps"], disk.receipt["steps"]):
        assert {name: value for name, value in old.items() if name != "excluded_keys"} == {name: value for name, value in new.items() if name != "excluded_keys"}
    assert memory.receipt["steps"][-1]["excluded_keys"] == [{"id": "001"}]
    assert disk.receipt["execution"] == {"storage": "disk", "engine": "streaming"}
    assert memory.receipt["execution"]["storage"] == "memory"


@pytest.mark.parametrize("data", [
    pl.DataFrame({"text": ["é\n\r\t\"", None, ""], "x": [1, None, -2]}),
    pl.DataFrame({"x": [float("nan"), -0., float("inf"), -float("inf"), None]}),
    pl.DataFrame({"x": [[1., None], None, []], "struct": [{"x": 1.}, None, {"x": None}]}),
    pl.DataFrame({"x": pl.Series(["a", None], dtype=pl.Enum(["a", "b"]))}),
    pl.DataFrame({"x": pl.Series([], dtype=pl.String)}),
])
def test_disk_digest_retains_v1_logical_hash(tmp_path, data):
    with execution_context("disk"), DiskWorkspace(tmp_path / "result") as workspace:
        assert workspace.snapshot(data).logical_digest == _digest(data)


@pytest.mark.parametrize("recipe,code", [
    ({"key": ["id"], "steps": [{"op": "filter", "where": False}]}, "UNDECLARED_LOSS"),
    ({"key": ["id"], "checks": {"row_count": {"exact": 3}}}, "ROW_COUNT_VIOLATION"),
    ({"key": ["id"], "steps": [{"op": "df_to_lower"}]}, "DISK_OPERATION_UNSUPPORTED"),
])
def test_failed_disk_checks_publish_nothing_and_remove_lock(tmp_path, recipe, code):
    with pytest.raises(wr.WrangleError) as failure:
        wr.prepare(pl.DataFrame({"id": [1, 2]}), recipe, execution="disk", output=tmp_path / "result")
    assert failure.value.code == code
    assert list(tmp_path.iterdir()) == []


def test_disk_identity_maps_are_complete_evidence_and_verified_on_reuse(tmp_path):
    result = wr.prepare(pl.DataFrame({"id": [" a", " b"]}), {"key": ["id"], "steps": [{"op": "clean_text", "columns": ["id"], "key": ["id"]}]}, execution="disk", output=tmp_path / "result")
    reference = result.receipt["steps"][0]["observation_transition"]["identity_map"]
    assert pl.read_parquet(tmp_path / "result" / reference["path"]).to_dicts() == [{"input": {"id": " a"}, "output": {"id": "a"}}, {"input": {"id": " b"}, "output": {"id": "b"}}]
    result.write(tmp_path / "copy")
    assert wr.inspect(tmp_path / "copy")["rows"] == 2
    assert wr.prepare(result, {}).data.height == 2
    (tmp_path / "result" / reference["path"]).write_bytes(b"changed")
    with pytest.raises(wr.WrangleError, match="evidence") as failure:
        wr.prepare(tmp_path / "result", {}, execution="disk", output=tmp_path / "reused")
    assert failure.value.code == "BUNDLE_MISMATCH"
    assert not (tmp_path / "reused").exists()


@pytest.mark.parametrize("execution,output,code", [("invalid", None, "INVALID_EXECUTION"), ([], None, "INVALID_EXECUTION"), ("disk", None, "OUTPUT_REQUIRED")])
def test_explicit_disk_invocation_boundaries(execution, output, code):
    with pytest.raises(wr.WrangleError) as failure:
        wr.prepare([], {}, execution=execution, output=output)
    assert failure.value.code == code


def test_disk_rejects_overwrite_and_preserves_result(tmp_path):
    target = tmp_path / "result"
    target.mkdir()
    (target / "research.txt").write_text("retained")
    with pytest.raises(wr.WrangleError) as failure:
        wr.prepare(pl.DataFrame({"id": [1]}), {}, execution="disk", output=target)
    assert failure.value.code == "OUTPUT_EXISTS"
    assert (target / "research.txt").read_text() == "retained"
    assert not list(tmp_path.glob("*.wrangle-lock"))


def test_disk_cli_matches_python_engine_receipt(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("id,x\n001,2\n002,4\n")
    recipe = {"key": ["id"], "steps": [{"op": "cast", "columns": {"x": "Int64"}}]}
    protocol = tmp_path / "recipe.yaml"
    from wrangle._protocol import dump_recipe
    protocol.write_text(dump_recipe(recipe))
    direct = wr.prepare(source, recipe, execution="disk", output=tmp_path / "python")
    cli = subprocess.run([sys.executable, "-m", "wrangle", "prepare", str(protocol), "--json", "--source", "data=" + str(source), "--execution", "disk", "--output", str(tmp_path / "cli")], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert not cli.stderr
    assert json.loads(cli.stdout) == direct.receipt
    missing = subprocess.run([sys.executable, "-m", "wrangle", "prepare", str(protocol), "--json", "--source", "data=" + str(source), "--execution", "disk"], capture_output=True, text=True)
    assert missing.returncode == 1
    assert json.loads(missing.stderr)["code"] == "OUTPUT_REQUIRED"


def test_disk_engine_matches_verified_canonical_research_workflows(tmp_path, monkeypatch):
    """Run actual shipped recipes through both engines, comparing scientific results."""
    from wrangle.docs import preparation_workflows, research_batch, expression_workflow, frozen_parameters
    original = wr.prepare
    covered = set()
    invocation = 0

    def compared(sources, recipe, **kwargs):
        nonlocal invocation
        memory = original(sources, recipe, **kwargs)
        invocation += 1
        disk = original(sources, recipe, execution="disk", output=tmp_path / str(invocation))
        assert disk.data.collect().equals(memory.data)
        assert disk.receipt["checks"] == memory.receipt["checks"]
        assert disk.receipt["key"] == memory.receipt["key"]
        assert disk.receipt["units"] == memory.receipt["units"]
        assert disk.receipt["output"] == memory.receipt["output"]
        covered.update(step["op"] for step in memory.receipt["steps"])
        return memory

    monkeypatch.setattr(wr, "prepare", compared)
    preparation_workflows.run()
    research_batch.run()
    expression_workflow.run()
    frozen_parameters.run()
    wr.prepare(pl.DataFrame({"id": [1], "x": [None], "other": [0]}), {"key": ["id"], "steps": [{"op": "select", "columns": ["id", "x"]}, {"op": "fill", "columns": {"x": "observed"}}]})
    from wrangle._api import _SIMPLE
    assert covered == {*_SIMPLE, "join"}
    assert covered >= {"normalize_missing", "clean_text", "parse_datetime", "recode", "encode", "sort", "deduplicate", "concat", "pivot", "unpivot", "explode", "unnest", "aggregate", "join_asof", "window", "sample", "partition", "cast", "filter", "derive", "impute", "standardize", "convert_unit", "rename", "join"}


@pytest.mark.parametrize("changed", ["evidence", "receipt"])
def test_disk_parent_must_remain_consistent_during_capture(tmp_path, monkeypatch, changed):
    parent = wr.prepare(pl.DataFrame({"id": [" a"]}), {"key": ["id"], "steps": [{"op": "clean_text", "columns": ["id"], "key": ["id"]}]}, execution="disk", output=tmp_path / "parent")
    evidence = tmp_path / "parent" / parent.receipt["evidence_files"][0]["path"]
    original = DiskWorkspace.snapshot

    def mutate_after_capture(workspace, plan):
        table = original(workspace, plan)
        if changed == "evidence":
            evidence.write_bytes(b"changed")
        else:
            parent.receipt["units"] = {"id": "invented"}
        return table

    monkeypatch.setattr(DiskWorkspace, "snapshot", mutate_after_capture)
    with pytest.raises(wr.WrangleError) as failure:
        wr.prepare(parent, {}, execution="disk", output=tmp_path / "child")
    assert failure.value.code == ("BUNDLE_MISMATCH" if changed == "evidence" else "RESULT_CHANGED")
    assert not (tmp_path / "child").exists()
    assert not list(tmp_path.glob("*.wrangle-lock"))


def test_disk_write_rejects_source_changes_during_publication(tmp_path, monkeypatch):
    parent = wr.prepare(pl.DataFrame({"id": [1]}), {}, execution="disk", output=tmp_path / "parent")
    original = __import__("shutil").copyfile
    source_path = tmp_path / "parent" / "data.parquet"

    def changed_during_copy(source, destination, **kwargs):
        result = original(source, destination, **kwargs)
        if source == source_path:
            source_path.write_bytes(b"changed")
        return result

    monkeypatch.setattr("wrangle._storage.shutil.copyfile", changed_during_copy)
    with pytest.raises(wr.WrangleError) as failure:
        parent.write(tmp_path / "copy")
    assert failure.value.code == "SOURCE_CHANGED"
    assert not (tmp_path / "copy").exists()
    assert not list(tmp_path.glob("*.wrangle-lock"))


def test_disk_publication_must_match_checked_logical_output(tmp_path):
    with execution_context("disk"), DiskWorkspace(tmp_path / "result") as workspace:
        table = workspace.snapshot(pl.DataFrame({"id": [1]}))
        receipt = {"recipe": {}, "output": {"sha256": "wrong", "rows": 1, "columns": {"id": "Int64"}}}
        with pytest.raises(wr.WrangleError) as failure:
            workspace.publish(table, receipt)
        assert failure.value.code == "BUNDLE_MISMATCH"
    assert list(tmp_path.iterdir()) == []
