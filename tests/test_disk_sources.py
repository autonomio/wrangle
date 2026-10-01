"""Disk source capture preserves strict parsing, identity, and parent evidence."""
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle
from wrangle._api import Prepared, _digest
from wrangle._sources import read_source


class _Table:
    def __init__(self, path):
        self.path = path
        self.schema = self.lazy.collect_schema()
        self.columns = list(self.schema)
        self.height = self.lazy.select(pl.len()).collect(engine="streaming").item()

    @property
    def lazy(self):
        return pl.scan_parquet(self.path, glob=False)

    @property
    def logical_digest(self):
        # Test fixture only. Production storage computes this without collecting.
        return _digest(self.lazy.collect())


class _Workspace:
    def __init__(self, root):
        self.root = root
        self.root.mkdir()
        self.count = 0
        self.plans = []

    def snapshot(self, plan):
        assert isinstance(plan, pl.LazyFrame)
        self.plans.append(plan.explain())
        path = self.root / f"table-{self.count}.parquet"
        self.count += 1
        plan.sink_parquet(path, maintain_order=True, engine="streaming")
        return _Table(path)


def _disk_digest(data):
    return data.logical_digest if isinstance(data, _Table) else _digest(data)


def _read(source, workspace):
    return read_source(source, prepared_type=Prepared, digest=_disk_digest, workspace=workspace)


@pytest.fixture
def workspace(tmp_path):
    return _Workspace(tmp_path / "disk snapshots")


@pytest.mark.parametrize("format", ["csv", "tsv", "parquet", "ipc"])
def test_disk_files_are_captured_before_scanning_and_do_not_rescan_live_input(tmp_path, workspace, format):
    path = tmp_path / f"measurements[1].{format}"
    original = pl.DataFrame({"id": ["001", "002"], "value": ["1.25", None]})
    if format in {"csv", "tsv"}:
        original.write_csv(path, separator="\t" if format == "tsv" else ",")
    elif format == "parquet":
        original.write_parquet(path)
    else:
        original.write_ipc(path)
    table, info = _read(path, workspace)
    path.unlink()
    assert_frame_equal(table.lazy.collect(), original)
    assert info["rows"] == 2
    assert info["format"] == format
    assert info["path"] == str(path.resolve())
    assert info["snapshot_sha256"] == _digest(original)
    assert len(info["sha256"]) == 64
    assert str(path) not in workspace.plans[0]
    assert str(workspace.root) in workspace.plans[0]


def test_disk_csv_retains_declared_dialect_schema_and_missing_codes(tmp_path, workspace):
    path = tmp_path / "instrument.csv"
    path.write_text("# device\nraw_id;raw_value\n001;NA\n002;1,25\n", encoding="utf-8")
    table, info = _read({"path": path, "options": {
        "separator": ";", "comment_prefix": "#", "new_columns": ["id", "value"],
        "null_values": {"value": "NA"}, "decimal_comma": True,
    }, "schema": {"value": "Float64"}}, workspace)
    assert_frame_equal(table.lazy.collect(), pl.DataFrame({"id": ["001", "002"], "value": [None, 1.25]}))
    assert info["schema"] == {"value": "Float64"}
    assert info["options"]["null_values"] == {"value": "NA"}


@pytest.mark.parametrize("header", ["id,id", "id,", ",value"])
def test_disk_csv_validates_original_header_before_renaming(tmp_path, workspace, header):
    path = tmp_path / "duplicate.csv"
    path.write_text(header + "\n001,2\n")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": path, "options": {"new_columns": ["id", "value"]}}, workspace)
    assert caught.value.code == "DUPLICATE_COLUMNS"


@pytest.mark.parametrize("format", ["ndjson", "jsonl", "xlsx"])
def test_disk_rejects_formats_without_identical_strict_parsing_contract(tmp_path, workspace, format):
    path = tmp_path / f"source.{format}"
    path.write_text('{"id":"001"}\n')
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path, workspace)
    assert caught.value.code == "DISK_SOURCE_UNSUPPORTED"
    assert workspace.count == 0


def test_disk_non_utf8_csv_is_an_explicit_boundary(tmp_path, workspace):
    path = tmp_path / "source.csv"
    path.write_bytes("id,label\n001,café\n".encode("cp1252"))
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": path, "options": {"encoding": "cp1252"}}, workspace)
    assert caught.value.code == "DISK_SOURCE_UNSUPPORTED"
    # Existing explicit decoding remains supported in memory execution.
    memory, _ = read_source({"path": path, "options": {"encoding": "cp1252"}}, prepared_type=Prepared, digest=_digest)
    assert memory["label"].item() == "café"


def test_disk_utf8_alias_is_resolved_to_strict_native_parser(tmp_path, workspace):
    path = tmp_path / "source.csv"
    path.write_text("id,label\n001,café\n", encoding="utf-8")
    table, info = _read({"path": path, "options": {"encoding": "UTF-8"}}, workspace)
    assert table.lazy.collect()["label"].item() == "café"
    assert info["options"]["encoding"] == "utf8"


def test_disk_csv_rejects_invalid_utf8_without_lossy_decoding(tmp_path, workspace):
    path = tmp_path / "source.csv"
    path.write_bytes(b"id,label\n001,\xff\n")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path, workspace)
    assert caught.value.code == "INVALID_SOURCE"


def test_disk_native_schema_is_asserted(tmp_path, workspace):
    path = tmp_path / "source.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(path)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": path, "schema": {"id": "String"}}, workspace)
    assert caught.value.code == "SCHEMA_MISMATCH"


def test_disk_source_changed_during_copy_is_rejected(tmp_path, workspace, monkeypatch):
    from wrangle import _sources
    path = tmp_path / "source.csv"
    path.write_text("id\n001\n")
    original = _sources.shutil.copyfile
    def copy_then_change(source, target):
        result = original(source, target)
        path.write_text("id\n999\n")
        return result
    monkeypatch.setattr(_sources.shutil, "copyfile", copy_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path, workspace)
    assert caught.value.code == "SOURCE_CHANGED"
    assert not list(workspace.root.iterdir())


def test_disk_source_changed_during_execution_is_rejected(tmp_path, workspace, monkeypatch):
    path = tmp_path / "source.csv"
    path.write_text("id\n001\n")
    original = workspace.snapshot
    def snapshot_then_change(plan):
        result = original(plan)
        path.write_text("id\n999\n")
        return result
    monkeypatch.setattr(workspace, "snapshot", snapshot_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path, workspace)
    assert caught.value.code == "SOURCE_CHANGED"


def test_disk_dataframe_and_lazyframe_sources_sink_native_plans(workspace):
    original = pl.DataFrame({"id": ["001", "002"], "value": [1.5, 2.5]})
    first, first_info = _read(original, workspace)
    second, second_info = _read(original.lazy(), workspace)
    assert_frame_equal(first.lazy.collect(), original)
    assert_frame_equal(second.lazy.collect(), original)
    assert first_info == second_info


def test_disk_memory_prepared_and_bundle_preserve_parent_evidence(tmp_path, workspace):
    original = pl.DataFrame({"id": ["001", "002"], "mass": [1.5, 2.5]})
    prepared = wrangle.prepare(original, {"key": "id", "units": {"mass": "g"}}, output=tmp_path / "bundle")
    captured, captured_info = _read(prepared, workspace)
    saved, saved_info = _read(tmp_path / "bundle", workspace)
    assert_frame_equal(captured.lazy.collect(), original)
    assert_frame_equal(saved.lazy.collect(), original)
    assert captured_info["parent"] == saved_info["parent"]
    assert saved_info["parent"]["units"] == {"mass": "g"}
    assert saved_info["snapshot_sha256"] == prepared.receipt["output"]["sha256"]


def test_disk_bundle_must_retain_stable_files_during_capture(tmp_path, workspace, monkeypatch):
    from wrangle import _sources
    wrangle.prepare(pl.DataFrame({"id": ["001"]}), {"key": "id"}, output=tmp_path / "bundle")
    original = _sources.shutil.copyfile
    def copy_then_change(source, target):
        result = original(source, target)
        if Path(source).name == "data.parquet":
            receipt = tmp_path / "bundle" / "receipt.json"
            receipt.write_text(receipt.read_text() + "\n")
        return result
    monkeypatch.setattr(_sources.shutil, "copyfile", copy_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "bundle", workspace)
    assert caught.value.code == "SOURCE_CHANGED"


def test_parent_snapshot_format_is_explicitly_validated(tmp_path, workspace):
    import json
    wrangle.prepare(pl.DataFrame({"id": ["001"]}), {"key": "id"}, output=tmp_path / "bundle")
    receipt_path = tmp_path / "bundle" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["hash_format"] = "unrecognized-v9"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "bundle", workspace)
    assert caught.value.code == "INVALID_BUNDLE"


def _published_disk_result(tmp_path, *, with_evidence=False):
    from wrangle._storage import DiskWorkspace, execution_context
    memory = wrangle.prepare(pl.DataFrame({"id": ["001", "002"], "mass": [1.5, 2.5]}), {"key": "id", "units": {"mass": "g"}})
    receipt = dict(memory.receipt)
    with execution_context("disk"), DiskWorkspace(tmp_path / "parent") as native:
        table = native.snapshot(memory.data.lazy())
        if with_evidence:
            native.record(memory.data.lazy().select("id"), "observations")
            receipt["evidence_files"] = native.evidence_files
        return native.publish(table, receipt)


def test_native_disk_result_reuses_as_snapshot_without_collecting(tmp_path, monkeypatch):
    from wrangle._storage import DiskWorkspace, execution_context
    prepared = _published_disk_result(tmp_path)
    native_collect = pl.LazyFrame.collect
    def reduction_only(plan, *args, **kwargs):
        result = native_collect(plan, *args, **kwargs)
        assert result.height <= 1, "Source loading must not collect the full table."
        return result
    monkeypatch.setattr(pl.LazyFrame, "collect", reduction_only)
    with execution_context("disk"), DiskWorkspace(tmp_path / "child") as native:
        table, info = read_source(prepared, prepared_type=Prepared, digest=_digest, workspace=native)
        assert info["snapshot_sha256"] == prepared.receipt["output"]["sha256"]
        assert table.height == 2
        assert info["parent"]["units"] == {"mass": "g"}


def test_native_disk_result_can_be_explicitly_loaded_in_memory(tmp_path):
    prepared = _published_disk_result(tmp_path)
    memory, info = read_source(prepared, prepared_type=Prepared, digest=_digest)
    assert_frame_equal(memory, pl.DataFrame({"id": ["001", "002"], "mass": [1.5, 2.5]}))
    assert info["parent"]["units"] == {"mass": "g"}


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_saved_evidence_files_are_verified_before_bundle_reuse(tmp_path, workspace, execution):
    prepared = _published_disk_result(tmp_path, with_evidence=True)
    entry = prepared.receipt["evidence_files"][0]
    (tmp_path / "parent" / entry["path"]).write_bytes(b"changed evidence")
    with pytest.raises(wrangle.WrangleError) as caught:
        read_source(tmp_path / "parent", prepared_type=Prepared, digest=_disk_digest if execution == "disk" else _digest, workspace=workspace if execution == "disk" else None)
    assert caught.value.code == "BUNDLE_MISMATCH"


def test_evidence_files_remain_stable_during_bundle_snapshot(tmp_path, workspace, monkeypatch):
    prepared = _published_disk_result(tmp_path, with_evidence=True)
    entry = prepared.receipt["evidence_files"][0]
    evidence = tmp_path / "parent" / entry["path"]
    original = workspace.snapshot
    def snapshot_then_change(plan):
        result = original(plan)
        evidence.write_bytes(b"changed evidence")
        return result
    monkeypatch.setattr(workspace, "snapshot", snapshot_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "parent", workspace)
    assert caught.value.code == "SOURCE_CHANGED"


def test_disk_ipc_directory_wildcards_cannot_select_neighbor_files(tmp_path, monkeypatch):
    from wrangle import _sources
    workspace = _Workspace(tmp_path / "disk[X]snapshots")
    neighbor = tmp_path / "diskXsnapshots"
    neighbor.mkdir()
    path = tmp_path / "source.ipc"
    pl.DataFrame({"id": ["001"]}).write_ipc(path)
    original = _sources.shutil.copyfile
    def copy_with_neighbor(source, target):
        result = original(source, target)
        pl.DataFrame({"id": ["999"]}).write_ipc(neighbor / Path(target).name)
        return result
    monkeypatch.setattr(_sources.shutil, "copyfile", copy_with_neighbor)
    table, info = _read(path, workspace)
    assert table.lazy.collect()["id"].to_list() == ["001"]
    assert info["rows"] == 1
