"""Ingestion retains measurements, identifiers, missingness, and record shape."""
from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wrangle._core import WrangleError, as_series, frame


@pytest.mark.parametrize("records", [[{"x": 1}, {"x": 0.5}], [{"x": 0.5}, {"x": 1}]])
def test_record_inference_preserves_fractional_measurements(records):
    table = frame(records).collect()
    assert table.schema == {"x": pl.Float64}
    assert table["x"].to_list() == [record["x"] for record in records]


def test_record_union_preserves_late_fields_missingness_and_order():
    records = [{"sample": "001", "signal": 1}, {}, {"sample": "002", "signal": 0.5, "late": "kept"}]
    table = frame(records).collect()
    assert table.columns == ["sample", "signal", "late"]
    assert table.to_dict(as_series=False) == {"sample": ["001", None, "002"], "signal": [1.0, None, 0.5], "late": [None, None, "kept"]}


def test_fields_appearing_after_native_inference_window_are_kept():
    records = [{"first": i} for i in range(150)] + [{"first": 150, "late": 7}]
    table = frame(records).collect()
    assert table.columns == ["first", "late"]
    assert table.height == 151
    assert table["late"].null_count() == 150
    assert table["late"][-1] == 7


@pytest.mark.parametrize("input_data", [[{"x": 2**53 + 1}, {"x": 0.5}], {"x": [2**53 + 1, 0.5]}, [[2**53 + 1], [0.5]]])
def test_mixed_float_and_large_integer_never_rounds_identifier(input_data):
    with pytest.raises(WrangleError) as failure:
        frame(input_data)
    assert failure.value.code == "LOSSY_CAST"
    assert failure.value.details["column"] in {"x", "column_0"}


def test_single_column_normalization_retains_lossy_cast_error():
    with pytest.raises(WrangleError) as failure:
        as_series([2**53 + 1, 0.5])
    assert failure.value.code == "LOSSY_CAST"


def test_integer_only_identifiers_remain_exact():
    large = 2**53 + 1
    table = frame([{"id": large}, {"id": large + 2}]).collect()
    assert table.schema == {"id": pl.Int64}
    assert table["id"].to_list() == [large, large + 2]


@pytest.mark.parametrize("rows", [[[1, "a"], [0.5, "b"]], ((1, "a"), (0.5, "b"))])
def test_matrix_orientation_preserves_rows(rows):
    table = frame(rows).collect()
    assert table.columns == ["column_0", "column_1"]
    assert table.rows() == [(1.0, "a"), (0.5, "b")]


@pytest.mark.parametrize("invalid", [[{"x": 1}, [2]], [None, {"x": 1}], [{}, {}], [[], []], [[1], [2, 3]], [[1], None], [{1: "name"}], [{"x": 1}, {"x": "text"}]])
def test_ambiguous_rows_and_incompatible_fields_fail_explicitly(invalid):
    with pytest.raises(WrangleError) as failure:
        frame(invalid)
    assert failure.value.code == "INVALID_INPUT"


def test_nested_lists_preserve_null_empty_and_fractional_values():
    records = [{"v": [1, 0.5]}, {"v": None}, {"v": []}, {"v": [2.25, None]}]
    table = frame(records).collect()
    assert table.schema == {"v": pl.List(pl.Float64)}
    assert table["v"].to_list() == [[1.0, 0.5], None, [], [2.25, None]]


def test_nested_structs_preserve_late_fields_parent_null_and_fractional_values():
    records = [{"meta": {"x": 1, "labels": ["a"]}}, {"meta": None}, {"meta": {"x": 0.5, "late": "present"}}, {"meta": {}}]
    table = frame(records).collect()
    assert table.schema == {"meta": pl.Struct({"x": pl.Float64, "labels": pl.List(pl.String), "late": pl.String})}
    assert table["meta"].to_list() == [{"x": 1.0, "labels": ["a"], "late": None}, None, {"x": 0.5, "labels": None, "late": "present"}, {"x": None, "labels": None, "late": None}]


@pytest.mark.parametrize("records", [[{"v": [2**53 + 1, 0.5]}], [{"v": {"x": 2**53 + 1}}, {"v": {"x": 0.5}}]])
def test_nested_lossy_mixed_numbers_are_rejected(records):
    with pytest.raises(WrangleError) as failure:
        frame(records)
    assert failure.value.code == "LOSSY_CAST"


@pytest.mark.parametrize("records", [[{"v": [1]}, {"v": "x"}], [{"v": {"x": 1}}, {"v": [1]}], [{"v": [1, "x"]}], [{"v": object()}]])
def test_unsupported_nested_values_fail_explicitly(records):
    with pytest.raises(WrangleError) as failure:
        frame(records)
    assert failure.value.code == "INVALID_INPUT"


def test_native_nested_input_keeps_explicit_dtype_and_snapshot():
    source = pl.DataFrame({"id": ["001", "002"], "v": pl.Series([[1.0, 2.0], None], dtype=pl.List(pl.Float32))})
    assert_frame_equal(frame(source).collect(), source)
    assert_frame_equal(frame(source.lazy()).collect(), source)


def test_temporal_record_fields_keep_native_types():
    records = [{"d": date(2020, 1, 1), "t": datetime(2020, 1, 1)}, {"d": None, "t": None}]
    table = frame(records).collect()
    assert table.schema == {"d": pl.Date, "t": pl.Datetime("us")}
    assert table.rows() == [(date(2020, 1, 1), datetime(2020, 1, 1)), (None, None)]
