from datetime import datetime, date
import polars as pl
import pytest

from wrangle._core import WrangleError
from wrangle._recipe_tables import (OPERATIONS, sort, deduplicate, concat, pivot, unpivot,
                                   explode, unnest, aggregate, join_asof, window, sample, partition)


def lf(**columns):
    return pl.DataFrame(columns).lazy()


def metric(column, method, **options):
    return {"column": column, "method": method, "nulls": "ignore", "min_count": 1, **options}


def test_all_operations_are_native_table_contracts():
    assert len(OPERATIONS) == 12
    for function in OPERATIONS.values():
        assert function.__wrangle_contract__["returns"] == ("table",)
        assert function.__wrangle_contract__["recipe"] == "yes"


def test_sort_requires_declared_ties_and_null_placement():
    data = lf(id=[1, 2, 3], day=[2, None, 2])
    with pytest.raises(WrangleError, match="ties"):
        sort(data, "day", descending=False, nulls="last", ties="error")
    assert sort(data, "day", descending=False, nulls="last", ties="source").collect()["id"].to_list() == [1, 3, 2]


def test_deduplicate_conflicts_survivor_and_source_order():
    data = lf(subject=["b", "a", "a", "c"], visit=[1, 1, 2, 1], x=[4, 10, 11, 5])
    with pytest.raises(WrangleError, match="differing"):
        deduplicate(data, "subject", keep="last", conflicts="error", order_by="visit")
    result = deduplicate(data, "subject", keep="last", conflicts="allow", order_by="visit").collect()
    assert result["subject"].to_list() == ["b", "a", "c"]
    assert result["x"].to_list() == [4, 11, 5]
    assert deduplicate(data, "subject", keep="none", conflicts="allow").collect()["subject"].to_list() == ["b", "c"]


def test_concat_union_requires_identical_shared_types_and_provenance():
    data, source = lf(id=[1], x=[3.0]), lf(id=[2], y=[4.0])
    result = concat(data, [source], schema="union", provenance="batch", labels=["before", "after"]).collect()
    assert result.columns == ["id", "x", "y", "batch"]
    assert result["batch"].to_list() == ["before", "after"]
    assert result["x"].to_list() == [3.0, None]
    with pytest.raises(WrangleError, match="identical"):
        concat(data, [lf(id=[2.0], x=[4.0])], schema="strict", provenance="batch", labels=["a", "b"])


def test_pivot_distinguishes_absent_cells_and_missing_observations():
    data = lf(subject=["a", "a", "b"], assay=["x", "x", "y"], amount=[2.0, 3.0, None])
    with pytest.raises(WrangleError, match="multiple"):
        pivot(data, ["subject"], "assay", ["amount"], ["x", "y"], duplicate_cells="error", missing_cells="zero")
    result = pivot(data, ["subject"], "assay", ["amount"], ["x", "y"], duplicate_cells="sum", missing_cells="zero").collect()
    assert result.to_dict(as_series=False) == {"subject": ["a", "b"], "amount__x": [5.0, 0.0], "amount__y": [0.0, None]}
    with pytest.raises(WrangleError, match="outside"):
        pivot(data, ["subject"], "assay", ["amount"], ["x"], duplicate_cells="sum", missing_cells="null")


def test_unpivot_never_coerces_measurements_and_preserves_declared_order():
    data = lf(id=[1, 2], b=[2.0, None], a=[1.0, 3.0])
    result = unpivot(data, ["id"], ["b", "a"], variable="assay", value="amount", nulls="keep").collect()
    assert result["id"].to_list() == [1, 1, 2, 2]
    assert result["assay"].to_list() == ["b", "a", "b", "a"]
    assert result["amount"].to_list() == [2.0, 1.0, None, 3.0]
    with pytest.raises(WrangleError, match="identical dtypes"):
        unpivot(lf(id=[1], a=[2], b=[2.0]), ["id"], ["a", "b"], variable="assay", value="amount", nulls="keep")


def test_explode_aligned_lists_retains_element_identity():
    data = lf(id=[1, 2, 3], a=[[10, 11], [], None], b=[["x", "y"], [], None])
    result = explode(data, ["a", "b"], index="element", empty="null", nulls="null").collect()
    assert result["id"].to_list() == [1, 1, 2, 3]
    assert result["element"].to_list() == [0, 1, -1, -1]
    assert explode(data, ["a", "b"], index="element", empty="drop", nulls="drop").collect().height == 2
    with pytest.raises(WrangleError, match="equal lengths"):
        explode(lf(a=[[1]], b=[[2, 3]]), ["a", "b"], index="element", empty="error", nulls="error")


def test_unnest_fields_never_shadow_parent_columns():
    data = lf(id=[1, 2], assay=[{"x": 2, "unit": "mg"}, None])
    assert unnest(data, ["assay"], separator="__").collect().to_dict(as_series=False) == {"id": [1, 2], "assay__x": [2, None], "assay__unit": ["mg", None]}
    with pytest.raises(WrangleError, match="collide"):
        unnest(lf(assay=[{"x": 2}], assay__x=[9]), ["assay"], separator="__")


def test_aggregate_counts_missingness_ddof_and_exact_integer_sum():
    data = lf(subject=["a", "a", "b"], x=[2, 4, None])
    result = aggregate(data, ["subject"], {"n": {"method": "len"}, "n_x": {"column": "x", "method": "count", "nulls": "ignore"}, "sum_x": metric("x", "sum"), "sd": metric("x", "std", ddof=1)}, null_keys="error").collect()
    assert result.to_dict(as_series=False) == {"subject": ["a", "b"], "n": [2, 1], "n_x": [2, 0], "sum_x": [6, None], "sd": [2 ** 0.5, None]}
    assert result.schema["sum_x"] == pl.Int128
    with pytest.raises(WrangleError) as caught:
        aggregate(lf(g=[1, 1], x=[2**53, 2**53 + 1]), ["g"], {"mean": metric("x", "mean")}, null_keys="error")
    assert caught.value.code == "LOSSY_CAST"


def test_calendar_aggregate_never_crosses_subjects():
    data = lf(subject=["a", "b", "a"], time=[datetime(2024, 1, 10), datetime(2024, 1, 12), datetime(2024, 2, 1)], x=[2, 9, 3])
    result = aggregate(data, ["subject"], {"sum": metric("x", "sum")}, null_keys="error", time="time", every="1mo", period="1mo", closed="left", label="left").collect()
    assert result["sum"].to_list() == [2, 3, 9]


def test_asof_within_subject_tolerance_retains_left_order_and_tie_policy():
    left = lf(subject=["b", "a", "a"], time=[3, 3, 8])
    right = lf(subject=["a", "b", "a"], time=[2, 2, 2], exposure=[10, 20, 11])
    options = dict(on="time", by=["subject"], strategy="backward", tolerance=2, exact=True, unmatched="keep")
    with pytest.raises(WrangleError, match="ties"):
        join_asof(left, right, ties="error", **options)
    result = join_asof(left, right, ties="last", **options).collect()
    assert result["exposure"].to_list() == [20, 11, None]
    with pytest.raises(WrangleError, match="no match"):
        join_asof(left, right, ties="last", **{**options, "unmatched": "error"})


def test_window_subject_local_order_does_not_reorder_observations():
    data = lf(subject=["a", "b", "a", "b"], time=[2, 1, 1, 2], x=[20, 9, 10, 11])
    metrics = {"previous": {"column": "x", "method": "lag", "n": 1, "nulls": "keep"}, "mean": {"column": "x", "method": "rolling_mean", "size": 2, "min_count": 2, "nulls": "keep"}}
    result = window(data, ["subject"], ["time"], metrics, ties="error").collect()
    assert result["previous"].to_list() == [10, None, None, 9]
    assert result["mean"].to_list() == [15.0, None, None, 10.0]


def test_time_rolling_calendar_membership_and_minimum():
    data = lf(subject=["a"] * 3, time=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 5)], x=[2.0, 4.0, 8.0])
    result = window(data, ["subject"], ["time"], {"mean": {"column": "x", "method": "rolling_mean", "period": "2d", "closed": "right", "min_count": 2, "nulls": "keep"}}, ties="error").collect()
    assert result["mean"].to_list() == [None, 3.0, None]


def test_sampling_whole_subjects_is_replayable_and_stratified():
    data = lf(subject=["a", "a", "b", "b", "c", "c", "d", "d"], stratum=["A"] * 4 + ["B"] * 4, x=list(range(8)))
    options = dict(unit="group", groups=["subject"], strata=["stratum"], seed=9, replacement=False, n=1)
    one, two = sample(data, **options).collect(), sample(data, **options).collect()
    assert one.equals(two)
    assert one.height == 4
    assert one.group_by("subject").len()["len"].to_list() == [2, 2]
    with pytest.raises(WrangleError, match="fewer"):
        sample(data, **{**options, "n": 3})


def test_weighted_replacement_tracks_each_cluster_draw():
    data = lf(subject=["a", "a", "b", "b"], x=[1, 2, 3, 4], weight=[0.0, 0.0, 1.0, 1.0])
    options = dict(unit="group", groups=["subject"], strata=[], weights="weight", seed=21, replacement=True, n=3, draw_id="draw")
    result = sample(data, **options).collect()
    assert result.height == 6
    assert result["subject"].to_list() == ["b"] * 6
    assert result["draw"].to_list() == [0, 0, 1, 1, 2, 2]
    assert result.equals(sample(data, **options).collect())


def test_partition_annotates_without_subject_leakage_and_exact_counts():
    data = lf(subject=[f"s{i}" for i in range(10) for _ in range(2)], x=list(range(20)))
    options = dict(output="split", labels=["train", "test"], groups=["subject"], strata=[], method="random", fractions=[0.7, 0.3], seed=7)
    result = partition(data, **options).collect()
    assert result.height == 20
    assert result.filter(pl.col("split") == "train").height == 14
    assert result.group_by("subject").agg(pl.col("split").n_unique())["split"].to_list() == [1] * 10
    assert result.equals(partition(data, **options).collect())


def test_chronological_partition_requires_declared_group_boundary_rule():
    data = lf(subject=["a", "a", "b"], time=[date(2024, 1, 1), date(2024, 2, 1), date(2024, 2, 2)])
    options = dict(output="split", labels=["before", "after"], groups=["subject"], strata=[], method="chronological", time="time", boundaries=["2024-02-01"])
    with pytest.raises(WrangleError, match="time assignment"):
        partition(data, **options)
    result = partition(data, **options, group_time="max").collect()
    assert result["split"].to_list() == ["after", "after", "after"]


def test_int128_sum_guard_is_native_and_refuses_overflow():
    data = pl.DataFrame({"g": [1, 1], "x": pl.Series([2**126, 2**126], dtype=pl.Int128)}).lazy()
    with pytest.raises(WrangleError) as caught:
        aggregate(data, ["g"], {"sum": metric("x", "sum")}, null_keys="error")
    assert caught.value.code == "ARITHMETIC_OVERFLOW"
    data = pl.DataFrame({"g": [1], "x": pl.Series([-(2**127)], dtype=pl.Int128)}).lazy()
    with pytest.raises(WrangleError) as caught:
        aggregate(data, ["g"], {"sum": metric("x", "sum")}, null_keys="error")
    assert caught.value.code == "ARITHMETIC_OVERFLOW"


def test_nearest_asof_requires_equidistant_winner_and_preserves_missingness():
    data = lf(time=[3, 8, 20])
    source = lf(time=[2, 4, 8], x=[10, 12, None])
    options = dict(on="time", by=[], strategy="nearest", tolerance=2, exact=True, ties="error", unmatched="keep")
    with pytest.raises(WrangleError, match="nearest_ties"):
        join_asof(data, source, **options)
    with pytest.raises(WrangleError, match="equidistant"):
        join_asof(data, source, **options, nearest_ties="error")
    assert join_asof(data, source, **options, nearest_ties="backward").collect()["x"].to_list() == [10, None, None]
    assert join_asof(data, source, **options, nearest_ties="forward").collect()["x"].to_list() == [12, None, None]


def test_nearest_temporal_subject_ties_are_explicit():
    data = lf(subject=["a"], time=[datetime(2024, 1, 2)])
    source = lf(subject=["a", "a"], time=[datetime(2024, 1, 1), datetime(2024, 1, 3)], x=[1, 3])
    options = dict(on="time", by=["subject"], strategy="nearest", tolerance="2d", exact=True, ties="error", unmatched="keep", nearest_ties="backward")
    assert join_asof(data, source, **options).collect()["x"].to_list() == [1]


def test_sampling_zero_mass_must_fail_within_each_requested_stratum():
    data = lf(stratum=["a", "a", "b", "b"], x=[1, 2, 3, 4], weight=[1.0, 1.0, 0.0, 0.0])
    for replacement in [False, True]:
        with pytest.raises(WrangleError) as caught:
            sample(data, unit="row", groups=[], strata=["stratum"], weights="weight", seed=0, replacement=replacement, n=1, draw_id="draw" if replacement else None)
        assert caught.value.code == "INVALID_WEIGHT"


def test_sampling_fraction_denominator_includes_zero_weight_units():
    data = lf(x=[1, 2, 3, 4], weight=[0.0, 0.0, 1.0, 1.0])
    result = sample(data, unit="row", groups=[], strata=[], weights="weight", seed=0, replacement=False, fraction=0.5).collect()
    assert result["x"].to_list() == [3, 4]


def test_sampling_refuses_collapsed_probability_bins():
    data = lf(x=[1, 2], weight=[1.0, 1e-30])
    with pytest.raises(WrangleError) as caught:
        sample(data, unit="row", groups=[], strata=[], weights="weight", seed=0, replacement=True, n=2, draw_id="draw")
    assert caught.value.code == "LOSSY_WEIGHT"


def test_empty_sample_zero_is_empty_positive_n_is_error():
    data = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)}).lazy()
    for replacement in [False, True]:
        opts = dict(unit="row", groups=[], strata=[], seed=0, replacement=replacement, draw_id="draw" if replacement else None)
        assert sample(data, **opts, n=0).collect().height == 0
        with pytest.raises(WrangleError) as caught:
            sample(data, **opts, n=1)
        assert caught.value.code == "INSUFFICIENT_SAMPLE"


def test_partition_share_uses_exact_declared_decimal_floor():
    data = lf(id=list(range(100)))
    result = partition(data, output="part", labels=["first", "second"], groups=[], strata=[], method="random", fractions=[0.29, 0.71], seed=0).collect()
    assert result.filter(pl.col("part") == "first").height == 29


def test_sort_nan_obeys_the_declared_missing_placement():
    data = lf(id=[1, 2, 3], x=[2.0, float("nan"), 1.0])
    result = sort(data, ["x"], descending=False, nulls="first", ties="source").collect()
    assert result["id"].to_list() == [2, 3, 1]
    assert result["x"][0] != result["x"][0]  # Sorting preserves the observed value.
    with pytest.raises(WrangleError, match="ties"):
        sort(lf(x=[float("nan"), None]), ["x"], descending=False, nulls="last", ties="error")


def test_declared_options_must_apply_to_the_actual_reducer():
    data = lf(g=[1, 1], t=[1, 2], x=[2.0, 3.0])
    with pytest.raises(WrangleError, match="do not apply"):
        aggregate(data, ["g"], {"mean": metric("x", "mean", ddof=0)}, null_keys="error")
    with pytest.raises(WrangleError, match="do not apply"):
        window(data, ["g"], ["t"], {"lag": {"column": "x", "method": "lag", "n": 1, "nulls": "keep", "min_count": 2}}, ties="error")


def test_explode_missing_element_and_absent_list_have_distinct_identity():
    data = pl.DataFrame({"id": [1, 2], "x": pl.Series([[None], []], dtype=pl.List(pl.Int64))}).lazy()
    result = explode(data, ["x"], index="element", empty="null", nulls="null").collect()
    assert result["element"].to_list() == [0, -1]


def test_chronological_boundary_cannot_be_silently_truncated():
    with pytest.raises(WrangleError) as caught:
        partition(lf(time=[1, 2]), output="part", labels=["before", "after"], groups=[], strata=[], method="chronological", time="time", boundaries=[1.5])
    assert caught.value.code == "LOSSY_CAST"
    with pytest.raises(WrangleError, match="parsed"):
        partition(lf(time=[date(2024, 1, 1)]), output="part", labels=["before", "after"], groups=[], strata=[], method="chronological", time="time", boundaries=["invalid-date"])


def test_group_weights_and_stratum_are_constant_within_subject():
    data = lf(subject=["a", "a"], weight=[1.0, 2.0])
    with pytest.raises(WrangleError) as caught:
        sample(data, unit="group", groups=["subject"], strata=[], weights="weight", seed=0, replacement=False, n=1)
    assert caught.value.code == "GROUP_CONFLICT"


def test_weighted_replacement_matches_known_mass_without_unmatched_draws():
    data = lf(category=["a", "b", "c"], weight=[1.0, 2.0, 7.0])
    result = sample(data, unit="row", groups=[], strata=[], weights="weight", seed=12, replacement=True, n=10000, draw_id="draw").collect()
    counts = result.group_by("category").len().sort("category")["len"].to_list()
    assert result.height == 10000 and result["draw"].n_unique() == 10000
    assert all(abs(observed / 10000 - expected) < 0.03 for observed, expected in zip(counts, [0.1, 0.2, 0.7]))


def test_boundary_must_retain_integer_identity_through_float_time_cast():
    with pytest.raises(WrangleError) as caught:
        partition(lf(time=[float(2**53)]), output="part", labels=["before", "after"], groups=[], strata=[], method="chronological", time="time", boundaries=[2**53 + 1])
    assert caught.value.code == "LOSSY_CAST"


def test_temporal_boundary_must_retain_declared_precision():
    data = pl.DataFrame({"time": pl.Series([datetime(2024, 1, 1)], dtype=pl.Datetime("us"))}).lazy()
    with pytest.raises(WrangleError) as caught:
        partition(data, output="part", labels=["before", "after"], groups=[], strata=[], method="chronological", time="time", boundaries=["2024-01-01T00:00:00.0000001"])
    assert caught.value.code == "LOSSY_CAST"
