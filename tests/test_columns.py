"""Scientific invariants for native Polars column preparation."""
from datetime import datetime
import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import wrangle as wr
from wrangle._core import WrangleError


def collected(table):
    return table.collect() if isinstance(table, pl.LazyFrame) else table


@pytest.mark.parametrize("lazy", [False, True])
def test_fill_keeps_table_and_order(lazy):
    source = pl.DataFrame({"id": [30, 10, 20], "x": [None, float("nan"), 4.0], "label": ["a", None, "b"]})
    result = wr.col_fill_nan(source.lazy() if lazy else source, ["x", "label"], fill_with=0)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    assert collected(result).to_dict(as_series=False) == {"id": [30, 10, 20], "x": [0.0, 0.0, 4.0], "label": ["a", "0", "b"]}
    assert source["x"].null_count() == 1


def test_allsame_counts_missing_and_empty():
    assert wr.col_check_allsame(pl.DataFrame({"x": [3, 3]}), "x")
    assert wr.col_check_allsame(pl.DataFrame({"x": [None, float("nan")]}), "x")
    assert not wr.col_check_allsame(pl.DataFrame({"x": [3, None]}), "x")
    assert not wr.col_check_allsame(pl.DataFrame({"x": []}), "x")


@pytest.mark.parametrize("lazy", [False, True])
def test_category_percentage_and_stats_use_observed_denominator(lazy):
    source = pl.DataFrame({"group": ["b", "a", "a", "a", "b"], "y": [0.0, 1.0, None, 0.0, None]})
    data = source.lazy() if lazy else source
    rate = collected(wr.col_corr_category(data, "group", "y", warning_threshold=2))
    assert rate.to_dict(as_series=False) == {"group": ["b", "a"], "y": [0.0, 50.0], "n": [1, 2], "low_sample": [True, False]}
    stats = collected(wr.col_groupby_stats(data, "group", "y"))
    assert stats["n"].to_list() == [1, 2]
    assert stats["sum"].to_list() == [0.0, 1.0]
    assert stats["std"][0] is None
    assert stats["std"][1] == pytest.approx(math.sqrt(0.5))
    with pytest.raises(WrangleError, match="only 0, 1"):
        wr.col_corr_category(pl.DataFrame({"group": ["a"], "y": [2]}), "group", "y")


@pytest.mark.parametrize("lazy", [False, True])
def test_categorical_ols_is_group_mean_rank_and_retains_identity(lazy):
    source = pl.DataFrame({"id": [3, 1, 5, 2, 4], "group": ["high", "low", "tie", "low", None], "y": [8.0, 2.0, 4.0, 6.0, 99.0]})
    result, mapping = wr.col_corr_ols(source.lazy() if lazy else source, "group", "y", return_mapping=True)
    assert collected(result)["id"].to_list() == source["id"].to_list()
    assert collected(result)["group"].to_list() == [2, 1, 1, 1, None]
    assert mapping.to_dict(as_series=False) == {"category": ["high", "low", "tie"], "coefficient": [8.0, 4.0, 4.0], "code": [2, 1, 1]}


@pytest.mark.parametrize("lazy", [False, True])
def test_outliers_respect_threshold_and_iqr_fences(lazy):
    source = pl.DataFrame({"id": range(5), "x": [100.0, 101.0, 102.0, 103.0, 200.0]})
    data = source.lazy() if lazy else source
    assert collected(wr.col_drop_outliers(data, "x", "iqr", 1.5))["id"].to_list() == [0, 1, 2, 3]
    assert collected(wr.col_drop_outliers(data, "x", "zscore", 1))["id"].to_list() == [0, 1, 2, 3]
    assert collected(wr.col_drop_outliers(data, "x", "zscore", 3)).height == 5
    constant = pl.DataFrame({"x": [7.0, 7.0, None]})
    assert wr.col_drop_outliers(constant, "x").height == 2
    with pytest.raises(WrangleError):
        wr.col_drop_outliers(data, "x", threshold=0)


@pytest.mark.parametrize("lazy", [False, True])
def test_empirical_pdf_cdf_are_distinct(lazy):
    source = pl.DataFrame({"group": ["a"] * 5 + ["b"] * 2, "value": [1, 1, 2, 3, None, 5, 6]})
    data = source.lazy() if lazy else source
    pdf = collected(wr.col_groupby_pdf(data, "group", "value", ascending=True))
    cdf = collected(wr.col_groupby_cdf(data, "group", "value", ascending=True))
    assert pdf["PDF"].to_list() == [0.5, 0.25, 0.25, 0.5, 0.5]
    assert cdf["CDF"].to_list() == [0.5, 0.75, 1.0, 0.5, 1.0]
    assert pdf.group_by("group").agg(pl.col("PDF").sum())["PDF"].to_list() == [1.0, 1.0]
    descending = collected(wr.col_groupby_cdf(data, "group", "value"))
    assert descending["CDF"].to_list() == [1.0, 0.75, 0.5, 1.0, 0.5]


@pytest.mark.parametrize("mode,expected", [("mean", [1.0, 2.0, 3.0, 2.0]), ("median", [1.0, 2.0, 3.0, 2.0]), ("mode", [1.0, 1.0, 3.0, 1.0]), ("common", [1.0, 1.0, 3.0, 1.0])])
def test_imputation_modes_and_existing_values(mode, expected):
    assert wr.col_impute_nan([1.0, None, 3.0, float("nan")], mode).to_list() == expected


def test_seeded_imputation_has_declared_distribution_and_replays():
    source = pl.Series("x", [1.0, 3.0] + [None] * 20)
    a = wr.col_impute_nan(source, seed=42)
    b = wr.col_impute_nan(source, seed=42)
    c = wr.col_impute_nan(source, seed=43)
    assert_series_equal(a, b)
    assert not a.equals(c)
    assert a[:2].to_list() == [1.0, 3.0]
    assert all(2 - math.sqrt(2) <= x <= 2 + math.sqrt(2) for x in a[2:])
    assert wr.col_impute_nan([5, None]).to_list() == [5.0, 5.0]
    with pytest.raises(WrangleError, match="without observed"):
        wr.col_impute_nan(pl.Series("x", [None], dtype=pl.Float64))


@pytest.mark.parametrize("lazy", [False, True])
def test_move_preserves_requested_order(lazy):
    source = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = collected(wr.col_move_place(source.lazy() if lazy else source, ["c", "b"]))
    assert result.columns == ["c", "b", "a"]
    assert result.rows() == [(5, 3, 1), (6, 4, 2)]


@pytest.mark.parametrize("lazy", [False, True])
def test_equal_sampling_replays_preserves_identity_and_rejects_short_groups(lazy):
    source = pl.DataFrame({"id": list(range(30)), "group": ["a"] * 15 + ["b"] * 15})
    data = source.lazy() if lazy else source
    first = collected(wr.col_resample_equal(data, "group", 5, seed=42))
    replay = collected(wr.col_resample_equal(data, "group", 5, seed=42))
    assert_frame_equal(first, replay)
    assert first["id"].to_list() == sorted(first["id"].to_list())
    assert first["id"].n_unique() == 10
    assert first.group_by("group").len()["len"].to_list() == [5, 5]
    assert not first.equals(collected(wr.col_resample_equal(data, "group", 5, seed=43)))
    with pytest.raises(WrangleError, match="fewer rows"):
        wr.col_resample_equal(data, "group", 16)


@pytest.mark.parametrize("lazy", [False, True])
def test_time_intervals_use_timestamp_order_and_right_labels(lazy):
    source = pl.DataFrame({"when": [datetime(2020, 1, 1, 0, 40), datetime(2020, 1, 1, 1, 0), datetime(2020, 1, 1, 0, 10)], "x": [3.0, 9.0, 1.0]})
    data = source.lazy() if lazy else source
    first = collected(wr.col_resample_interval(data, "x", "when", "first", 60))
    mean = collected(wr.col_resample_interval(data, "x", "when", "mean", 60))
    assert first["x"].to_list() == [1.0, 9.0]
    assert mean["x"].to_list() == [2.0, 9.0]
    assert first["when"].to_list() == [datetime(2020, 1, 1, 1), datetime(2020, 1, 1, 2)]


def test_max_scaling_retains_missing_and_handles_zero():
    scaled = wr.col_rescale_max([10.0, 6.0, None, 2.0]).to_list()
    assert scaled[:2] == pytest.approx([1.0, 0.6])
    assert scaled[2] is None
    assert scaled[3] == pytest.approx(0.2)
    assert wr.col_rescale_max([0, 0, None]).to_list() == [0.0, 0.0, None]
    assert wr.col_rescale_max([10, 6, 2], scale=3, to_int=True).to_list() == [3, 2, 1]
    with pytest.raises(WrangleError, match="Maximum is zero"):
        wr.col_rescale_max([-1, 0])


@pytest.mark.parametrize("lazy", [False, True])
def test_biclass_and_threshold_binary_preserve_missing(lazy):
    source = pl.DataFrame({"label": ["yes", "no", None], "x": [1.0, 3.0, None]})
    data = source.lazy() if lazy else source
    assert collected(wr.col_to_biclass(data, "label", "yes"))["label"].to_list() == [1, 0, None]
    assert collected(wr.col_to_binary(data, "x"))["x"].to_list() == [False, True, None]
    assert collected(wr.col_to_binary(data, "x", 0.5))["x"].to_list() == [False, True, None]
    with pytest.raises(WrangleError, match="at most two"):
        wr.col_to_biclass(pl.DataFrame({"x": ["a", "b", "c"]}), "x", "a")


@pytest.mark.parametrize("lazy", [False, True])
def test_category_codes_and_quantile_codes_are_sorted(lazy):
    source = pl.DataFrame({"s": ["b", "a", None, "c"], "x": [9.0, 1.0, None, 5.0]})
    data = source.lazy() if lazy else source
    assert collected(wr.col_to_binary(data, "s", "cat_string"))["s"].to_list() == [1, 0, None, 2]
    assert collected(wr.col_to_binary(data, "x", "cat_numeric"))["x"].to_list() == [2, 0, None, 1]


def test_equal_width_buckets_include_min_and_max_and_handle_constant():
    result = wr.col_to_buckets(pl.DataFrame({"x": [0.0, 2.0, 4.0, None]}), "x", 2)
    assert result.to_list() == ["0.0 to 2.0", "0.0 to 2.0", "2.0 to 4.0", None]
    assert wr.col_to_buckets(pl.DataFrame({"x": [7.0, 7.0, None]}), "x", 2).to_list() == ["7.0 to 7.0", "7.0 to 7.0", None]


@pytest.mark.parametrize("lazy", [False, True])
def test_pivot_sums_explicit_duplicate_records(lazy):
    source = pl.DataFrame({"sample": ["s2", "s1", "s1", "s1"], "label": ["a", "a", "a", "b"], "signal": [5.0, 1.0, 2.0, None]})
    result = collected(wr.col_to_cols(source.lazy() if lazy else source, "label", "sample"))
    assert result.to_dict(as_series=False) == {"sample": ["s2", "s1"], "a": [5.0, 3.0], "b": [0.0, None]}


@pytest.mark.parametrize("lazy", [False, True])
def test_onehot_names_order_missing_and_collision(lazy):
    source = pl.DataFrame({"id": [2, 1, 3], "label": ["b", "a", None]})
    result = collected(wr.col_to_multilabel(source.lazy() if lazy else source, "label", extended_colname=True))
    assert result.to_dict(as_series=False) == {"id": [2, 1, 3], "label_a": [0, 1, None], "label_b": [1, 0, None]}
    with pytest.raises(WrangleError, match="already exist"):
        wr.col_to_multilabel(source, "label", colnames=["id", "b"])


@pytest.mark.parametrize("lazy", [False, True])
def test_split_checks_field_counts_and_preserves_rows(lazy):
    source = pl.DataFrame({"id": [3, 1, 2], "name": ["a|b", None, "c|d"]})
    result = collected(wr.col_to_split(source.lazy() if lazy else source, "name", sep="|"))
    assert result.to_dict(as_series=False) == {"id": [3, 1, 2], "name_a": ["a", None, "c"], "name_b": ["b", None, "d"]}
    with pytest.raises(WrangleError, match="different numbers"):
        wr.col_to_split(pl.DataFrame({"name": ["a|b", "c"]}), "name", sep="|")


@pytest.mark.parametrize("name,args", [("col_corr_ols", ("group", "x")), ("col_drop_outliers", ("x",)), ("col_move_place", ("x",)), ("col_to_biclass", ("group", "a")), ("col_to_binary", ("x",)), ("col_to_split", ("group",))])
def test_destructive_argument_cannot_mutate_source(name, args):
    source = pl.DataFrame({"group": ["a", "b"], "x": [1, 2]})
    with pytest.raises(WrangleError) as failure:
        getattr(wr, name)(source, *args, destructive=True)
    assert failure.value.code == "IMMUTABLE_INPUT"


@pytest.mark.parametrize("name", ["category", "coefficient", "code"])
def test_ols_mapping_handles_scientific_input_column_names(name):
    source = pl.DataFrame({name: ["a", "b"], "outcome": [1.0, 2.0], "category" if name != "category" else "other": ["unused", "untouched"]})
    result, mapping = wr.col_corr_ols(source, name, "outcome", return_mapping=True)
    assert result[name].to_list() == [1, 2]
    assert result.columns == source.columns
    assert mapping.columns == ["category", "coefficient", "code"]


def test_boolean_outcome_percentage_and_explicit_missing_fill():
    source = pl.DataFrame({"group": ["a", "a", "a"], "y": [True, False, None]})
    result = wr.col_corr_category(source, "group", "y")
    assert result["y"].to_list() == [50.0]
    assert result["n"].to_list() == [2]
    assert wr.col_fill_nan(source, "y", None)["y"].to_list() == [True, False, None]


def test_sampling_does_not_normalize_selected_source_values():
    source = pl.DataFrame({"id": [1, 2, 3, 4], "group": [None, float("nan"), 1.0, 1.0]})
    result = wr.col_resample_equal(source, "group", 1, seed=0)
    assert_frame_equal(result, source.filter(pl.col("id").is_in(result["id"].to_list())))


@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["a", "b"])])
@pytest.mark.parametrize("lazy", [False, True])
def test_fill_preserves_category_schema_and_domain(dtype, lazy):
    source = pl.DataFrame({"group": ["a", None, "b"]}).with_columns(pl.col("group").cast(dtype))
    result = collected(wr.col_fill_nan(source.lazy() if lazy else source, "group", "a"))
    assert result.schema == source.schema
    assert result["group"].to_list() == ["a", "a", "b"]
    assert source["group"].null_count() == 1


@pytest.mark.parametrize("dtype,fill", [(pl.Int64, 0.5), (pl.UInt8, 300), (pl.Float32, 0.1), (pl.Float64, "unknown"), (pl.Boolean, 2), (pl.Enum(["a", "b"]), "unknown")])
def test_fill_rejects_lossy_or_incompatible_schema_changes(dtype, fill):
    source = pl.DataFrame({"x": pl.Series([None], dtype=dtype)})
    with pytest.raises(WrangleError) as failure:
        wr.col_fill_nan(source, "x", fill)
    assert failure.value.code in {"INVALID_FILL_VALUE", "LOSSY_FILL"}


def test_binary_missing_nan_never_becomes_low_or_high_class():
    source = pl.DataFrame({"x": [1.0, None, float("nan"), 3.0]})
    for func in ["mean", "median", "mode", 2, 0.5, "cat_numeric", "cat_int", "cat_string"]:
        output = wr.col_to_binary(source, "x", func)["x"].to_list()
        assert output[1:3] == [None, None]
    assert source["x"].is_nan().sum() == 1


def test_all_missing_class_encoding_retains_nulls():
    source = pl.DataFrame({"x": pl.Series([None, float("nan")], dtype=pl.Float64)})
    for func in ["mean", "median", "mode", 2, 0.5, "cat_numeric", "cat_string"]:
        assert wr.col_to_binary(source, "x", func)["x"].to_list() == [None, None]


def test_ols_temporary_names_cannot_overwrite_research_fields():
    source = pl.DataFrame({"x": ["a", "b"], "y": [1, 2], "__wrangle_code": [11, 22], "__wrangle_code_": [33, 44]})
    output = wr.col_corr_ols(source, "x", "y")
    assert output["__wrangle_code"].to_list() == [11, 22]
    assert output["__wrangle_code_"].to_list() == [33, 44]
    assert output.columns == source.columns


@pytest.mark.parametrize("operation", [
    lambda data: wr.col_impute_nan(data["x"], "mean"),
    lambda data: wr.col_rescale_max(data["x"]),
    lambda data: wr.col_to_buckets(data, "x"),
    lambda data: wr.col_to_binary(data, "x", "mean"),
    lambda data: wr.col_to_binary(data, "x", "cat_numeric"),
    lambda data: wr.col_groupby_stats(data, "group", "x"),
    lambda data: wr.col_corr_ols(data, "group", "x"),
    lambda data: wr.col_drop_outliers(data, "x"),
])
def test_float_statistics_reject_integer_measurement_precision_loss(operation):
    source = pl.DataFrame({"group": ["a", "b", "a"], "x": [2**53, 2**53 + 1, None]})
    with pytest.raises(WrangleError) as failure:
        operation(source)
    assert failure.value.code == "LOSSY_CAST"


def test_exact_integer_threshold_and_mode_do_not_require_float_conversion():
    source = pl.DataFrame({"x": [2**53, 2**53 + 1, None]})
    assert wr.col_to_binary(source, "x", 2**53 + 1)["x"].to_list() == [False, True, None]
    assert wr.col_impute_nan(source["x"], "mode").to_list() == [2**53, 2**53 + 1, 2**53]


def test_native_integer_pivot_and_interval_sum_cannot_wrap_at_int64_limit():
    value = 2**63 - 1
    source = pl.DataFrame({"sample": ["s1", "s1"], "group": ["a", "a"], "x": [value, value], "when": [datetime(2020, 1, 1, 0, 10), datetime(2020, 1, 1, 0, 20)]})
    assert wr.col_to_cols(source.select("sample", "group", "x"), "group", "sample")["a"].item() == 2 * value
    assert wr.col_resample_interval(source, "x", "when", "sum", 60)["x"].item() == 2 * value


def test_max_scaling_avoids_overflowing_reciprocal_for_tiny_measurements():
    assert wr.col_rescale_max([1e-320, 5e-321]).to_list() == pytest.approx([1.0, 0.5])
