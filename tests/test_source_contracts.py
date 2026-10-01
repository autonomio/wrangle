"""File dialects and retained parent evidence have explicit, verifiable contracts."""
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
import sys

import polars as pl
from polars.testing import assert_frame_equal
import pytest

import wrangle
from wrangle._api import Prepared, _digest
from wrangle._sources import read_source


def _read(source):
    return read_source(source, prepared_type=Prepared, digest=_digest)


@pytest.fixture
def prepared(tmp_path):
    return wrangle.prepare(pl.DataFrame({"id": ["001", "002"], "mass": [1.5, 2.5]}), {
        "version": 1, "key": "id", "units": {"mass": "g"},
        "descriptions": {"id": "Instrument sample identifier", "mass": "Measured mass"},
        "checks": {"required": ["id", "mass"]},
    }, output=tmp_path / "batch")


def test_plain_csv_and_tsv_keep_identifiers_strings(tmp_path):
    for suffix, separator in (("csv", ","), ("tsv", "\t")):
        path = tmp_path / f"measurements.{suffix}"
        path.write_text(f"id{separator}value\n001{separator}1.25\n002{separator}\n", encoding="utf-8")
        data, info = _read(path)
        assert data.schema == {"id": pl.String, "value": pl.String}
        assert data["id"].to_list() == ["001", "002"]
        assert data["value"].to_list() == ["1.25", None]
        assert info["snapshot_sha256"] == _digest(data)
        assert info["format"] == suffix


def test_declared_dialect_encoding_schema_and_null_codes(tmp_path):
    path = tmp_path / "instrument export.raw"
    path.write_bytes("# instrument export\nid;label;value\n001;'café;left';1,25\n002;'control';NA\n".encode("cp1252"))
    declaration = {"path": str(path), "format": "csv", "options": {
        "separator": ";", "quote_char": "'", "encoding": "cp1252", "comment_prefix": "#", "decimal_comma": True,
        "null_values": {"value": "NA"},
    }, "schema": {"value": "Float64"}}
    data, info = _read(declaration)
    assert data["id"].to_list() == ["001", "002"]
    assert data["label"].to_list() == ["café;left", "control"]
    assert data["value"].to_list() == [1.25, None]
    assert data.schema["value"] == pl.Float64
    assert info["options"]["has_header"] is True
    assert info["schema"] == {"value": "Float64"}
    json.dumps(info, allow_nan=False)


def test_headerless_csv_keeps_first_record_and_declared_names(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text("001,2\n002,3\n", encoding="utf-8")
    data, _ = _read({"path": str(path), "options": {"has_header": False, "new_columns": ["id", "value"]}, "schema": {"value": "Int64"}})
    assert data.rows() == [("001", 2), ("002", 3)]


def test_declared_skip_rows_and_skip_after_header(tmp_path):
    path = tmp_path / "preamble.csv"
    path.write_text("Instrument report\nid,value\nunits,g\n001,2\n", encoding="utf-8")
    data, _ = _read({"path": str(path), "options": {"skip_rows": 1, "skip_rows_after_header": 1}})
    assert data.rows() == [("001", "2")]


@pytest.mark.parametrize("header", ["id,id", "id,", ",value"])
def test_original_headers_checked_before_requested_renaming(tmp_path, header):
    path = tmp_path / "duplicate.csv"
    path.write_text(header + "\n001,2\n", encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": str(path), "options": {"new_columns": ["id", "value"]}})
    assert caught.value.code == "DUPLICATE_COLUMNS"


@pytest.mark.parametrize("options", [
    {"ignore_errors": True}, {"truncate_ragged_lines": True}, {"encoding": "utf8-lossy"},
    {"encoding": "not-a-codec"}, {"separator": "::"}, {"has_header": "false"},
    {"skip_rows": True}, {"skip_rows_after_header": -1}, {"null_values": [None]},
    {"new_columns": ["id"]}, {"quote_char": 1},
])
def test_unsupported_or_ambiguous_csv_options_fail(tmp_path, options):
    path = tmp_path / "samples.csv"
    path.write_text("id,value\n001,2\n", encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": str(path), "options": options})
    assert caught.value.code == "INVALID_SOURCE_OPTIONS"


def test_absent_schema_and_null_columns_fail(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("id,value\n001,2\n", encoding="utf-8")
    for declaration in ({"schema": {"absent": "Int64"}}, {"options": {"null_values": {"absent": "NA"}}}):
        with pytest.raises(wrangle.WrangleError) as caught:
            _read({"path": str(path), **declaration})
        assert caught.value.code == "COLUMN_NOT_FOUND"


def test_decimal_schema_preserves_declared_precision(tmp_path):
    path = tmp_path / "precision.csv"
    path.write_text("id,value\n001,9007199254740993.125\n", encoding="utf-8")
    data, _ = _read({"path": str(path), "schema": {"value": {"Decimal": {"precision": 38, "scale": 3}}}})
    assert data["value"].item() == Decimal("9007199254740993.125")
    assert data.schema["value"] == pl.Decimal(38, 3)


def test_native_file_schema_is_asserted_and_input_is_unchanged(tmp_path):
    path = tmp_path / "native.parquet"
    pl.DataFrame({"id": [1, 2]}).write_parquet(path)
    original = path.read_bytes()
    data, _ = _read({"path": str(path), "schema": {"id": "Int64"}})
    assert data["id"].to_list() == [1, 2]
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": str(path), "schema": {"id": "Float64"}})
    assert caught.value.code == "SCHEMA_MISMATCH"
    assert path.read_bytes() == original


def test_ndjson_duplicate_fields_are_not_silently_replaced(tmp_path):
    path = tmp_path / "records.ndjson"
    path.write_text('{"id":"001","id":"002"}\n', encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path)
    assert caught.value.code == "DUPLICATE_COLUMNS"


def test_excel_requires_explicit_one_sheet_and_optional_dependency(tmp_path, monkeypatch):
    path = tmp_path / "samples.xlsx"
    path.write_bytes(b"not-read-without-optional-dependency")
    for options in ({}, {"sheet_id": 0}, {"sheet_name": ["one", "two"]}, {"sheet_name": "one", "sheet_id": 1}):
        with pytest.raises(wrangle.WrangleError) as caught:
            _read({"path": str(path), "options": options})
        assert caught.value.code == "INVALID_SOURCE_OPTIONS"
    monkeypatch.setitem(sys.modules, "fastexcel", None)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read({"path": str(path), "options": {"sheet_name": "measurements"}})
    assert caught.value.code == "DEPENDENCY_REQUIRED"
    assert caught.value.details == {"dependency": "fastexcel"}


def test_prepared_and_saved_bundle_retain_identical_parent_evidence(prepared, tmp_path):
    memory, memory_info = _read(prepared)
    saved, saved_info = _read(tmp_path / "batch")
    assert_frame_equal(memory, prepared.data)
    assert_frame_equal(saved, prepared.data)
    assert memory_info["parent"] == saved_info["parent"]
    assert saved_info["parent"]["key"] == ["id"]
    assert saved_info["parent"]["units"] == {"mass": "g"}
    assert saved_info["parent"]["descriptions"]["mass"] == "Measured mass"
    assert saved_info["parent"]["recipe_sha256"] == prepared.receipt["recipe_sha256"]
    assert len(saved_info["parent"]["receipt_sha256"]) == 64
    assert saved_info["snapshot_sha256"] == memory_info["snapshot_sha256"] == prepared.receipt["output"]["sha256"]
    assert set(saved_info["files"]) == {"data.parquet", "recipe.yaml", "receipt.json", "report.txt"}
    json.dumps(saved_info, allow_nan=False)


@pytest.mark.parametrize("change", ["data", "receipt", "nonfinite_receipt"])
def test_mutated_prepared_evidence_cannot_be_reused(prepared, change):
    if change == "data":
        prepared.data.replace_column(1, pl.Series("mass", [9.0, 10.0]))
    elif change == "receipt":
        prepared.receipt["units"]["mass"] = "kg"
    else:
        prepared.receipt["unexpected"] = float("nan")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(prepared)
    assert caught.value.code == "RESULT_CHANGED"


@pytest.mark.parametrize("change", ["data", "recipe", "recipe_hash", "rows", "schema", "variable_unit"])
def test_saved_bundle_corruption_fails(prepared, tmp_path, change):
    bundle = tmp_path / "batch"
    receipt_path = bundle / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if change == "data":
        pl.DataFrame({"id": ["001", "002"], "mass": [9.0, 10.0]}).write_parquet(bundle / "data.parquet")
    elif change == "recipe":
        (bundle / "recipe.yaml").write_text("{}")
    else:
        if change == "recipe_hash":
            receipt["recipe_sha256"] = "0" * 64
        elif change == "rows":
            receipt["output"]["rows"] = 3
        elif change == "schema":
            receipt["output"]["columns"]["mass"] = "String"
        else:
            receipt["variables"]["mass"]["unit"] = "kg"
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(bundle)
    assert caught.value.code == "BUNDLE_MISMATCH"


@pytest.mark.parametrize("text", ['{"recipe":{},"recipe":{}}', '{"recipe":{},"value":NaN}', '[1,2]'])
def test_bundle_metadata_requires_strict_object_json(prepared, tmp_path, text):
    (tmp_path / "batch" / "receipt.json").write_text(text, encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "batch")
    assert caught.value.code == "INVALID_BUNDLE"


def test_source_changed_during_parsing_is_rejected(tmp_path, monkeypatch):
    from wrangle import _sources
    path = tmp_path / "changing.csv"
    path.write_text("id,value\n001,2\n", encoding="utf-8")
    native = _sources._csv
    def read_then_change(*args):
        result = native(*args)
        path.write_text("id,value\n001,2\n002,3\n", encoding="utf-8")
        return result
    monkeypatch.setattr(_sources, "_csv", read_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(path)
    assert caught.value.code == "SOURCE_CHANGED"


def test_bundle_files_must_be_stable_during_reading(prepared, tmp_path, monkeypatch):
    native = pl.read_parquet
    def read_then_change(path, **kwargs):
        result = native(path, **kwargs)
        receipt = tmp_path / "batch" / "receipt.json"
        receipt.write_text(receipt.read_text() + "\n")
        return result
    monkeypatch.setattr(pl, "read_parquet", read_then_change)
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "batch")
    assert caught.value.code == "SOURCE_CHANGED"


def test_equal_source_chunk_layouts_keep_one_snapshot_and_receipt():
    one = pl.DataFrame({"id": ["001", "002"], "mass": [1.5, 2.5]})
    chunked = pl.concat([one.head(1), one.tail(1)], rechunk=False)
    assert chunked["id"].n_chunks() == 2
    data, first = _read(one)
    other, second = _read(chunked)
    assert_frame_equal(data, other)
    assert first == second
    recipe = {"key": "id", "units": {"mass": "g"}}
    assert wrangle.prepare(one, recipe).receipt == wrangle.prepare(chunked, recipe).receipt


def test_null_recipe_file_is_not_accepted_as_retained_intent(prepared, tmp_path):
    (tmp_path / "batch" / "recipe.yaml").write_text("null", encoding="utf-8")
    with pytest.raises(wrangle.WrangleError) as caught:
        _read(tmp_path / "batch")
    assert caught.value.code == "INVALID_BUNDLE"


@pytest.mark.parametrize("format", ["csv", "parquet"])
def test_source_paths_are_literal_files_not_implicit_glob_sets(tmp_path, format):
    literal = tmp_path / f"samples[1].{format}"
    other = tmp_path / f"samples1.{format}"
    wanted = pl.DataFrame({"id": ["001"]})
    extra = pl.DataFrame({"id": ["999"]})
    if format == "csv":
        wanted.write_csv(literal)
        extra.write_csv(other)
    else:
        wanted.write_parquet(literal)
        extra.write_parquet(other)
    data, info = _read(literal)
    assert_frame_equal(data, wanted)
    assert info["rows"] == 1
    assert info["path"] == str(literal.resolve())


@pytest.mark.parametrize("values,dtype", [
    ([[1.0], None, [2.0]], pl.List(pl.Float64)),
    ([{"x": 1.0}, None, {"x": 2.0}], pl.Struct({"x": pl.Float64})),
    ([Decimal("1.25"), None, Decimal("2.5")], pl.Decimal(38, 2)),
    ([date(2026, 1, 1), None, date(2026, 1, 2)], pl.Date),
    ([b"\x00\xff", None, b"\x01"], pl.Binary),
    (["a", None, "b"], pl.Categorical),
    (["a", None, "b"], pl.Enum(["a", "b"])),
    ([[1.0, None], None, [2.0, 3.0]], pl.Array(pl.Float64, 2)),
    ([datetime(2026, 1, 1), None, datetime(2026, 1, 2)], pl.Datetime("ns")),
    ([datetime(2026, 1, 1, tzinfo=timezone.utc), None, datetime(2026, 1, 2, tzinfo=timezone.utc)], pl.Datetime("ns", "UTC")),
    ([timedelta(microseconds=1234), None, timedelta(days=1)], pl.Duration("ns")),
    ([time(12, 34, 56, 1234), None, time(13)], pl.Time),
])
def test_nullable_native_types_survive_verified_parquet_bundle(values, dtype, tmp_path):
    data = pl.DataFrame({"id": ["001", "002", "003"], "value": pl.Series("value", values, dtype=dtype)})
    prepared = wrangle.prepare(data, {"key": "id"}, output=tmp_path / "typed batch")
    restored, info = _read(tmp_path / "typed batch")
    assert_frame_equal(restored, prepared.data)
    assert info["snapshot_sha256"] == prepared.receipt["output"]["sha256"]
    assert info["parent"] == _read(prepared)[1]["parent"]


def test_snapshot_identity_distinguishes_null_and_nonfinite_float_observations():
    snapshots = []
    nested_snapshots = []
    for value in (None, float("nan"), float("inf"), float("-inf")):
        data = pl.DataFrame({"value": pl.Series("value", [value], dtype=pl.Float64)})
        nested = pl.DataFrame({"value": pl.Series("value", [[value]], dtype=pl.List(pl.Float64))})
        snapshots.append(_read(data)[1]["snapshot_sha256"])
        nested_snapshots.append(_read(nested)[1]["snapshot_sha256"])
    assert len(set(snapshots)) == 4
    assert len(set(nested_snapshots)) == 4


@pytest.mark.parametrize("has_header", [True, False])
def test_csv_schema_and_column_null_codes_follow_explicit_output_names(tmp_path, has_header):
    path = tmp_path / "named samples.csv"
    path.write_text(("old_id,old_value\n" if has_header else "") + "001,NA\n002,2\n")
    data, info = _read({
        "path": path,
        "options": {"has_header": has_header, "new_columns": ["id", "value"], "null_values": {"value": "NA"}},
        "schema": {"value": "Float64"},
    })
    assert_frame_equal(data, pl.DataFrame({"id": ["001", "002"], "value": [None, 2.0]}))
    assert info["options"]["null_values"] == {"value": "NA"}
    assert info["schema"] == {"value": "Float64"}
