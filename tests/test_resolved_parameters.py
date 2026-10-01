"""Resolved statistical state must preserve scientific meaning on another batch."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import runpy

import polars as pl
import pytest
import wrangle as wr
from polars.testing import assert_frame_equal
from wrangle._recipe_statistics import OPERATIONS, fill, impute, resolve_parameters, standardize


def fails(code, function):
    with pytest.raises(wr.WrangleError) as captured:
        function()
    assert captured.value.code == code
    return captured.value


def test_mean_imputation_reuses_reference_value_on_a_different_batch():
    reference = pl.DataFrame({"id": ["a", "b", "c"], "x": [1.0, 3.0, None]})
    parameters = resolve_parameters("impute", reference.lazy(), {"columns": ["x"], "method": "mean"})
    assert parameters["parameters"]["x"] == {"dtype": "Float64", "count": 2, "method": "mean", "replacement": 2.0}
    batch = pl.DataFrame({"id": ["d", "e"], "x": [100.0, None]})
    result = impute(batch.lazy(), **parameters).collect()
    assert result["x"].to_list() == [100.0, 2.0]
    assert batch["x"].to_list() == [100.0, None]
    assert reference["x"].to_list() == [1.0, 3.0, None]
    assert resolve_parameters("impute", batch.lazy(), parameters) == parameters
    assert json.loads(json.dumps(parameters, allow_nan=False)) == parameters


def test_standardization_records_actual_ddof_mean_and_std_then_reuses_them():
    reference = pl.DataFrame({"x": [2.0, 4.0, None]})
    args = resolve_parameters("standardize", reference.lazy(), {"columns": ["x"], "ddof": 0})
    assert args["parameters"]["x"] == {"dtype": "Float64", "count": 2, "mean": 3.0, "std": 1.0, "ddof": 0}
    assert standardize(pl.DataFrame({"x": [5.0, None]}).lazy(), **args).collect()["x"].to_list() == [2.0, None]
    assert standardize(reference.lazy(), **args).collect()["x"].to_list() == [-1.0, 1.0, None]
    fails("INVALID_PARAMETERS", lambda: standardize(reference.lazy(), ["x"], ddof=1, parameters=args["parameters"]))


def test_frozen_imputation_applies_to_an_all_missing_batch_without_fitting():
    args = resolve_parameters("impute", pl.DataFrame({"x": [2.0, 4.0]}).lazy(), {"columns": ["x"], "method": "median"})
    empty = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Float64)})
    assert impute(empty.lazy(), **args).collect()["x"].to_list() == [3.0, 3.0]
    fails("NO_OBSERVATIONS", lambda: impute(empty.lazy(), ["x"], method="median"))
    fails("NO_OBSERVATIONS", lambda: standardize(empty.lazy(), ["x"], ddof=1))


def test_uniform_reference_moments_are_frozen_seeded_and_column_independent():
    source = pl.DataFrame({"x": [0.0, 10.0, None, None], "y": [0.0, 10.0, None, None]})
    args = resolve_parameters("impute", source.lazy(), {"columns": ["x", "y"], "method": "uniform", "seed": 42})
    assert args["parameters"]["x"]["mean"] == 5.0
    assert args["parameters"]["x"]["std"] == pytest.approx(50**0.5)
    a = impute(source.lazy(), **args).collect()
    b = impute(source.lazy(), **args).collect()
    assert_frame_equal(a, b)
    assert a["x"].to_list()[:2] == [0.0, 10.0]
    assert a["x"].to_list()[2:] != a["y"].to_list()[2:]
    for name in ["x", "y"]:
        assert all(5 - 50**0.5 <= value <= 5 + 50**0.5 for value in a[name].to_list()[2:])
    fails("INVALID_SEED", lambda: impute(source.lazy(), ["x"], method="uniform"))
    fails("INVALID_SEED", lambda: impute(source.lazy(), ["x"], method="mean", seed=42))


@pytest.mark.parametrize("dtype,value", [
    (pl.Int128, 2**100 + 1),
    (pl.Decimal(38, 2), "9007199254740993.10"),
    (pl.Date, 18444),
    (pl.Datetime("ns", "Europe/Helsinki"), 1_600_000_000_123_456_789),
    (pl.Time, 12_345_678_901),
    (pl.Duration("ns"), -12_345_678_901),
    (pl.Categorical, "treated"),
    (pl.Enum(["control", "treated"]), "treated"),
    (pl.Boolean, True),
])
def test_mode_parameters_are_exact_json_and_reconstruct_native_types(dtype, value):
    source = pl.DataFrame({"x": [value, value, None]}).with_columns(pl.col("x").cast(dtype))
    args = resolve_parameters("impute", source.lazy(), {"columns": ["x"], "method": "mode"})
    assert args["parameters"]["x"]["replacement"] == value
    assert json.loads(json.dumps(args, allow_nan=False)) == args
    batch = pl.DataFrame({"x": pl.Series([None], dtype=dtype)})
    result = impute(batch.lazy(), **args).collect()
    assert result.schema == source.schema
    assert result["x"].equals(source["x"].head(1))


def test_mode_ties_use_declared_native_sort_order_without_batch_refit():
    source = pl.DataFrame({"x": ["b", "a", None]})
    args = resolve_parameters("impute", source.lazy(), {"columns": ["x"], "method": "mode"})
    assert args["parameters"]["x"]["replacement"] == "a"
    assert impute(pl.DataFrame({"x": ["z", None]}).lazy(), **args).collect()["x"].to_list() == ["z", "a"]


def test_frozen_field_and_dtype_contracts_prevent_silent_reinterpretation():
    source = pl.DataFrame({"x": [1.0, 3.0]})
    args = resolve_parameters("impute", source.lazy(), {"columns": ["x"], "method": "mean"})
    fails("DTYPE_MISMATCH", lambda: impute(pl.DataFrame({"x": [1, None]}).lazy(), **args))
    fails("INVALID_PARAMETERS", lambda: impute(source.lazy(), ["x"], method="mean", parameters={}))
    fails("INVALID_PARAMETERS", lambda: impute(source.lazy(), ["x"], method="median", parameters=args["parameters"]))
    state = deepcopy(args["parameters"])
    state["x"]["replacement"] = 2**53 + 1
    fails("LOSSY_FILL", lambda: impute(source.lazy(), ["x"], method="mean", parameters=state))
    state["x"]["replacement"] = float("inf")
    fails("INVALID_PARAMETERS", lambda: impute(source.lazy(), ["x"], method="mean", parameters=state))


@pytest.mark.parametrize("dtype,values", [
    (pl.Int64, [2**53, 2**53 + 1]),
    (pl.Decimal(38, 1), ["9007199254740993.0", "9007199254740994.0"]),
])
def test_float_statistics_reject_lost_integer_and_decimal_precision(dtype, values):
    source = pl.DataFrame({"x": values}).with_columns(pl.col("x").cast(dtype))
    for method in ["mean", "median", "uniform"]:
        fails("LOSSY_CAST", lambda: impute(source.lazy(), ["x"], method=method, seed=1 if method == "uniform" else None))
    fails("LOSSY_CAST", lambda: standardize(source.lazy(), ["x"], ddof=1))


def test_fill_is_selected_exact_and_preserves_declared_dtypes():
    source = pl.DataFrame({"id": ["001", "002"], "x": [1, None], "category": ["control", None], "date": [date(2026, 9, 30), None]})
    result = fill(source.lazy(), {"x": 2, "category": "treated", "date": "2026-10-01"}).collect()
    assert result.schema == source.schema
    assert result["id"].to_list() == ["001", "002"]
    assert result["x"].to_list() == [1, 2]
    assert result["category"].to_list() == ["control", "treated"]
    assert result["date"].to_list() == [date(2026, 9, 30), date(2026, 10, 1)]
    fails("LOSSY_FILL", lambda: fill(source.lazy(), {"x": 1.5}))
    fails("INVALID_FILL_VALUE", lambda: fill(source.lazy(), {"x": "1"}))
    fails("INVALID_FILL_VALUE", lambda: fill(source.lazy(), {"category": 1}))


def test_decimal_fill_preserves_exact_scale_and_rejects_fractional_loss():
    source = pl.DataFrame({"x": ["1.20", None]}).with_columns(pl.col("x").cast(pl.Decimal(38, 2)))
    result = fill(source.lazy(), {"x": "2.5"}).collect()
    assert result["x"].cast(pl.String).to_list() == ["1.20", "2.50"]
    assert result.schema == source.schema
    fails("LOSSY_FILL", lambda: fill(source.lazy(), {"x": "2.501"}))


def test_typed_physical_fill_retains_nanosecond_precision_and_zone():
    dtype = pl.Datetime("ns", "UTC")
    value = 1_600_000_000_123_456_789
    source = pl.DataFrame({"x": pl.Series([None], dtype=dtype)})
    literal = {"dtype": {"Datetime": {"time_unit": "ns", "time_zone": "UTC"}}, "value": value}
    result = fill(source.lazy(), {"x": literal}).collect()
    assert result["x"].cast(pl.Int64).item() == value
    assert result.schema["x"] == dtype
    fails("DTYPE_MISMATCH", lambda: fill(source.lazy(), {"x": {"dtype": "Date", "value": 1}}))
    assert fill(pl.DataFrame({"x": [None]}).lazy(), {"x": {"dtype": "UInt8", "value": 2}}).collect().schema["x"] == pl.UInt8


def test_constant_singleton_and_insufficient_standardization_are_explicit():
    source = pl.DataFrame({"x": [3.0, 3.0, None]})
    args = resolve_parameters("standardize", source.lazy(), {"columns": ["x"], "ddof": 1})
    assert args["parameters"]["x"]["std"] == 0.0
    assert standardize(source.lazy(), **args).collect()["x"].to_list() == [0.0, 0.0, None]
    fails("ZERO_VARIANCE", lambda: standardize(pl.DataFrame({"x": [4.0]}).lazy(), **args))
    assert standardize(pl.DataFrame({"x": [3.0, None]}).lazy(), ["x"], ddof=1).collect()["x"].to_list() == [0.0, None]
    fails("INSUFFICIENT_OBSERVATIONS", lambda: standardize(pl.DataFrame({"x": [1.0, 2.0]}).lazy(), ["x"], ddof=2))
    fails("INVALID_OPTION", lambda: standardize(source.lazy(), ["x"], ddof=True))


def test_native_statistics_do_not_overflow_or_zero_out_distinct_extreme_values():
    tiny = pl.DataFrame({"x": [1e-300, 3e-300]}).lazy()
    tiny_result = standardize(tiny, ["x"], ddof=0).collect()["x"].to_list()
    assert tiny_result == pytest.approx([-1.0, 1.0])
    huge = pl.DataFrame({"x": [-1e308, 1e308]}).lazy()
    huge_result = standardize(huge, ["x"], ddof=0).collect()["x"].to_list()
    assert huge_result == pytest.approx([-1.0, 1.0])


def test_legacy_entrypoints_accept_reusable_parameters_and_preserve_shapes():
    source = pl.DataFrame({"id": ["a", "b"], "x": [2.0, 4.0]})
    args = resolve_parameters("df_impute_nan", source.lazy(), {"cols": "x", "impute_mode": "mean"})
    assert args["cols"] == ["x"]
    batch = pl.DataFrame({"id": ["c", "d"], "x": [100.0, None]})
    assert wr.df_impute_nan(batch, **args)["x"].to_list() == [100.0, 3.0]
    scaled = resolve_parameters("df_rescale_meanzero", source.lazy(), {"retain": "id", "ddof": 0})
    assert wr.df_rescale_meanzero(batch.lazy(), **scaled).collect()["x"].to_list() == [97.0, None]
    assert isinstance(wr.df_rescale_meanzero(batch, **scaled), pl.DataFrame)


def test_parameters_and_plans_reject_non_json_state_and_callbacks_before_execution():
    source = pl.DataFrame({"x": [1.0, None]}).lazy()
    args = resolve_parameters("impute", source, {"columns": ["x"], "method": "mean"})
    state = deepcopy(args["parameters"])
    state["x"]["replacement"] = (1.0,)
    fails("INVALID_PARAMETERS", lambda: impute(source, ["x"], method="mean", parameters=state))
    calls = []
    unsafe = source.with_columns(pl.col("x").map_elements(lambda value: calls.append(value) or value, return_dtype=pl.Float64))
    fails("UNSUPPORTED_CALLBACK", lambda: resolve_parameters("impute", unsafe, {"columns": ["x"], "method": "mean"}))
    assert calls == []
    assert set(OPERATIONS) == {"fill", "impute", "standardize"}


def test_installed_frozen_workflow_reuses_receipt_state_and_preserves_sources():
    installed = Path(wr.__file__).parent / "docs" / "frozen_parameters.py"
    checkout = Path(__file__).resolve().parents[1] / "examples" / "frozen_parameters.py"
    assert installed.read_bytes() == checkout.read_bytes()
    result = runpy.run_path(str(installed))["run"]()
    assert result["incoming"].data["mass_g"].to_list() == [3.0, 0.0]
    assert result["reference"].data["mass_g"].to_list() == [-1.0, 1.0, 0.0]


def test_public_recipe_records_and_reuses_actual_legacy_replacements():
    fitted = wr.prepare(pl.DataFrame({"id": ["a", "b"], "x": [2.0, 4.0]}), {"key": "id", "steps": [{"op": "df_impute_nan", "cols": "x", "impute_mode": "mean"}]})
    kwargs = fitted.receipt["steps"][0]["parameters"]
    assert kwargs["parameters"]["x"]["replacement"] == 3.0
    applied = wr.prepare(pl.DataFrame({"id": ["c", "d"], "x": [100.0, None]}), {"key": "id", "steps": [{"op": "df_impute_nan", **kwargs}]})
    assert applied.data["x"].to_list() == [100.0, 3.0]
    assert applied.receipt["steps"][0]["parameters"] == kwargs


def test_public_recipe_rejects_unknown_units_for_legacy_scaled_measurements():
    source = pl.DataFrame({"id": ["a", "b"], "x": [2.0, 4.0]})
    protocol = {"key": "id", "units": {"x": "g"}, "steps": [{"op": "df_rescale_meanzero", "retain": "id"}]}
    fails("UNDECLARED_UNITS", lambda: wr.prepare(source, protocol))
    protocol["steps"][0]["units"] = {"x": "1"}
    assert wr.prepare(source, protocol).receipt["units"] == {"x": "1"}


@pytest.mark.parametrize("method", ["mean", "median", "uniform"])
def test_large_constant_imputation_does_not_overflow_native_sums(method):
    reference = pl.DataFrame({"x": [1e308, 1e308]}).lazy()
    args = resolve_parameters("impute", reference, {"columns": ["x"], "method": method, **({"seed": 11} if method == "uniform" else {})})
    result = impute(pl.DataFrame({"x": pl.Series([None], dtype=pl.Float64)}).lazy(), **args).collect()
    assert result["x"].item() == 1e308
    assert standardize(reference, ["x"], ddof=1).collect()["x"].to_list() == [0.0, 0.0]


def test_adjacent_huge_values_retain_representable_scale_and_small_scales_do_not_become_zero():
    low, high = 1e308, 1.0000000000000002e308
    args = resolve_parameters("standardize", pl.DataFrame({"x": [low, high]}).lazy(), {"columns": ["x"], "ddof": 0})
    assert args["parameters"]["x"]["std"] == (high - low) / 2
    tiny = pl.DataFrame({"x": [1e-320, 2e-320]}).lazy()
    assert standardize(tiny, ["x"], ddof=0).collect()["x"].to_list() == pytest.approx([-1.0, 1.0])
    fails("STATISTIC_UNDERFLOW", lambda: standardize(pl.DataFrame({"x": [5e-324, 1e-323]}).lazy(), ["x"], ddof=0))


def test_supplied_float_moments_cannot_round_exact_integer_metadata():
    source = pl.DataFrame({"x": [1.0, 2.0]}).lazy()
    args = resolve_parameters("standardize", source, {"columns": ["x"], "ddof": 0})
    args["parameters"]["x"]["mean"] = 2**53 + 1
    fails("LOSSY_CAST", lambda: standardize(source, **args))


def test_mode_ties_respect_category_and_declared_ordinal_meaning():
    categories = pl.DataFrame({"x": ["b", "a", None]}).with_columns(pl.col("x").cast(pl.Categorical))
    ordinal = categories.with_columns(pl.col("x").cast(pl.Enum(["b", "a"])))
    categorical_args = resolve_parameters("impute", categories.lazy(), {"columns": ["x"], "method": "mode"})
    ordinal_args = resolve_parameters("impute", ordinal.lazy(), {"columns": ["x"], "method": "mode"})
    assert categorical_args["parameters"]["x"]["replacement"] == "a"
    assert ordinal_args["parameters"]["x"]["replacement"] == "b"
    fails("NONFINITE_RESULT", lambda: impute(pl.DataFrame({"x": [float("inf"), None]}).lazy(), ["x"], method="mode"))


def test_unit_context_records_physical_origin_and_rejects_changed_batch_units():
    source = pl.DataFrame({"x": [1000.0, 2000.0, None]}).lazy()
    arguments = {"columns": ["x"], "method": "mean"}
    fitted = resolve_parameters("impute", source, arguments, units={"x": "mg"})
    assert fitted["parameters"]["x"]["unit"] == "mg"
    assert fitted["parameters"]["x"]["replacement"] == 1500.0
    batch = pl.DataFrame({"x": pl.Series([None], dtype=pl.Float64)}).lazy()
    fails("UNIT_MISMATCH", lambda: resolve_parameters("impute", batch, fitted, units={"x": "g"}))
    fails("UNIT_MISMATCH", lambda: resolve_parameters("impute", batch, fitted, units={}))
    assert resolve_parameters("impute", batch, fitted, units={"x": "mg"}) == fitted
    assert impute(batch, **fitted).collect()["x"].item() == 1500.0


def test_unknown_origin_state_cannot_acquire_known_physical_units():
    source = pl.DataFrame({"x": [1.0, 3.0]}).lazy()
    arguments = {"columns": ["x"], "ddof": 0}
    direct = resolve_parameters("standardize", source, arguments)
    assert "unit" not in direct["parameters"]["x"]
    unknown = resolve_parameters("standardize", source, arguments, units={})
    assert unknown["parameters"]["x"]["unit"] is None
    assert resolve_parameters("standardize", source, direct, units={}) == unknown
    for state in (direct, unknown):
        fails("UNIT_MISMATCH", lambda: resolve_parameters("standardize", source, state, units={"x": "g"}))


@pytest.mark.parametrize("method", ["mean", "median", "mode", "uniform", "standardize"])
def test_row_unit_context_resolves_one_fixed_unit_and_never_pools_different_units(method):
    source = pl.DataFrame({"x": [1.0, 3.0, None], "unit": ["g", "g", "g"]}).lazy()
    operation = "standardize" if method == "standardize" else "impute"
    arguments = {"columns": ["x"], "ddof": 1} if method == "standardize" else {"columns": ["x"], "method": method, **({"seed": 11} if method == "uniform" else {})}
    fitted = resolve_parameters(operation, source, arguments, units={"x": "@unit"})
    assert fitted["parameters"]["x"]["unit"] == "g"
    mixed = pl.DataFrame({"x": [1.0, 3.0, None], "unit": ["g", "g", "mg"]}).lazy()
    fails("UNIT_MISMATCH", lambda: resolve_parameters(operation, mixed, arguments, units={"x": "@unit"}))
    fails("UNIT_MISMATCH", lambda: resolve_parameters(operation, mixed, fitted, units={"x": "@unit"}))
    assert resolve_parameters(operation, source, fitted, units={"x": "@unit"}) == fitted


@pytest.mark.parametrize("units", [["g"], {"x": None}, {"x": ""}])
def test_unit_context_requires_explicit_nonempty_strings(units):
    source = pl.DataFrame({"x": [1.0, 2.0]}).lazy()
    fails("INVALID_PARAMETERS", lambda: resolve_parameters("impute", source, {"columns": ["x"], "method": "mean"}, units=units))


@pytest.mark.parametrize("label", [None, "", "@another", "mg"])
def test_missing_or_different_row_units_include_rows_to_be_imputed(label):
    source = pl.DataFrame({"x": [1.0, None], "unit": ["g", label]}).lazy()
    fails("UNIT_MISMATCH", lambda: resolve_parameters("impute", source, {"columns": ["x"], "method": "mean"}, units={"x": "@unit"}))


@pytest.mark.parametrize("unit", [0, "", "@unit", {"value": "g"}])
def test_kernels_retain_only_scalar_fixed_unit_metadata(unit):
    source = pl.DataFrame({"x": [1.0, 2.0]}).lazy()
    args = resolve_parameters("impute", source, {"columns": ["x"], "method": "mean"})
    args["parameters"]["x"]["unit"] = unit
    fails("INVALID_PARAMETERS", lambda: impute(source, **args))
