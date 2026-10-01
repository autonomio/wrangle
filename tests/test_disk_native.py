"""Disk execution retains native recipe results and scientific failures."""
from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wrangle._core import WrangleError
from wrangle._recipe_expressions import compile_expression
from wrangle._recipe_fields import clean_text, normalize_missing, parse_datetime, recode
from wrangle._recipe_join import join
from wrangle._recipe_statistics import impute, resolve_parameters, standardize
from wrangle._recipe_tables import aggregate, sample, window
from wrangle._storage import collect, execution_context


def scan(tmp_path, name, data):
    path = tmp_path / f"{name}.parquet"
    data.write_parquet(path)
    return pl.scan_parquet(path)


def compiler(value, data):
    if isinstance(value, dict) and set(value) == {"col"}:
        return pl.col(value["col"])
    if not isinstance(value, dict):
        return pl.lit(value)
    expression = compile_expression(value, data, compiler)
    assert expression is not None
    return expression


def test_file_backed_field_preparation_matches_memory(tmp_path):
    source = scan(tmp_path, "fields", pl.DataFrame({
        "id": ["001", "002", "003"],
        "group": [" CONTROL ", " Treated ", "control"],
        "day": ["2026-10-01", "2026-10-02", None],
        "x": [-999.0, 3.0, 8.0],
    }))
    results = []
    for mode in ("memory", "disk"):
        with execution_context(mode):
            plan = normalize_missing(source, {"x": [-999.0]})
            plan = clean_text(plan, ["group"], case="lower")
            plan = recode(plan, "group", {"control": 0, "treated": 1}, dtype="Int8")
            plan = parse_datetime(plan, ["day"], "%Y-%m-%d", dtype="Date")
            expression = compiler({"clip": {"value": {"col": "x"}, "min": 0.0, "max": 5.0, "nulls": "keep"}}, plan)
            results.append(collect(plan.with_columns(expression.alias("x"))))
    assert_frame_equal(*results, check_exact=True)
    assert results[1].to_dict(as_series=False) == {
        "id": ["001", "002", "003"], "group": [0, 1, 0],
        "day": [date(2026, 10, 1), date(2026, 10, 2), None],
        "x": [None, 3.0, 5.0],
    }


def test_file_backed_join_grouping_and_windows_match_memory(tmp_path):
    left = scan(tmp_path, "visits", pl.DataFrame({
        "id": ["b", "a", "a"], "visit": [1, 2, 1], "x": [10.0, 4.0, 2.0],
    }))
    right = scan(tmp_path, "subjects", pl.DataFrame({"id": ["a", "b"], "arm": ["control", "treated"]}))
    results = []
    for mode in ("memory", "disk"):
        with execution_context(mode):
            combined = join(left, right, on="id", cardinality="m:1", unmatched="error", maintain_order="left")
            longitudinal = window(combined, ["id"], ["visit"], {"previous": {"column": "x", "method": "lag", "n": 1, "nulls": "keep"}}, ties="error")
            reduced = aggregate(combined, ["id"], {"n": {"method": "len"}, "mean_x": {"column": "x", "method": "mean", "nulls": "ignore", "min_count": 1}}, null_keys="error")
            results.append((collect(longitudinal), collect(reduced)))
    assert_frame_equal(results[0][0], results[1][0], check_exact=True)
    assert_frame_equal(results[0][1], results[1][1], check_exact=True)
    assert results[1][0]["previous"].to_list() == [None, 2.0, None]
    assert results[1][1].to_dict(as_series=False) == {"id": ["b", "a"], "n": [1, 2], "mean_x": [10.0, 3.0]}


def test_disk_frozen_parameters_match_memory_and_do_not_refit(tmp_path):
    reference = scan(tmp_path, "reference", pl.DataFrame({"x": [2.0, 4.0, None]}))
    batch = scan(tmp_path, "batch", pl.DataFrame({"x": [100.0, None]}))
    results = []
    for mode in ("memory", "disk"):
        with execution_context(mode):
            imputation = resolve_parameters("impute", reference, {"columns": ["x"], "method": "mean"})
            scale = resolve_parameters("standardize", reference, {"columns": ["x"], "ddof": 0})
            prepared = standardize(impute(batch, **imputation), **scale)
            results.append((imputation, scale, collect(prepared)))
    assert results[0][:2] == results[1][:2]
    assert_frame_equal(results[0][2], results[1][2], check_exact=True)
    assert results[1][2]["x"].to_list() == [97.0, 0.0]


def test_file_backed_seeded_group_sampling_matches_memory(tmp_path):
    source = scan(tmp_path, "groups", pl.DataFrame({
        "id": ["a", "a", "b", "b", "c", "c"],
        "visit": [1, 2, 1, 2, 1, 2], "weight": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
    }))
    results = []
    for mode in ("memory", "disk"):
        with execution_context(mode):
            results.append(collect(sample(source, unit="group", groups=["id"], strata=[], seed=42, replacement=True, n=4, weights="weight", draw_id="draw")))
    assert_frame_equal(*results, check_exact=True)
    assert results[1].height == 8
    assert results[1].group_by("draw").len().sort("draw")["len"].to_list() == [2, 2, 2, 2]


@pytest.mark.parametrize("operation,expected", [
    (lambda data: join(data, pl.DataFrame({"id": ["a", "a"], "arm": [0, 1]}).lazy(), on="id", cardinality="m:1", unmatched="keep"), "JOIN_CARDINALITY"),
    (lambda data: recode(data, "id", {"a": 0}, dtype="Int8"), "UNKNOWN_CATEGORY"),
    (lambda data: window(data, ["id"], ["visit"], {"lag": {"column": "x", "method": "lag", "n": 1, "nulls": "keep"}}, ties="error"), "AMBIGUOUS_ORDER"),
])
def test_disk_native_contract_failures_match_memory(tmp_path, operation, expected):
    source = scan(tmp_path, "invalid", pl.DataFrame({"id": ["a", "a", "b"], "visit": [1, 1, 2], "x": [2.0, 4.0, 8.0]}))
    failures = []
    for mode in ("memory", "disk"):
        with execution_context(mode), pytest.raises(WrangleError) as caught:
            collect(operation(source))
        failures.append(caught.value.to_dict())
    assert failures[0] == failures[1]
    assert failures[1]["code"] == expected
