"""Scientific invariants for native arrays and utilities; no network or downloads."""
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import wrangle as wr
from wrangle import utils
from wrangle._core import WrangleError
from wrangle.array.array_reshape_lstm import normalise_windows
from wrangle.utils.wrangler_utils import string_contains_to_binary, filling_nans, imputing_nans


def test_shuffle_reproducible_aligned_and_immutable():
    x = pl.DataFrame({"id": range(31), "value": range(31)})
    y = pl.Series("target", range(31))
    original = x.clone()
    x_out, y_out = wr.array_random_shuffle(x, y, seed=42)
    assert x_out["id"].to_list() == y_out.to_list()
    assert_frame_equal(x_out, wr.array_random_shuffle(x, seed=42))
    assert_frame_equal(x, original)
    assert_series_equal(y, pl.Series("target", range(31)))
    assert x_out["id"].to_list() != x["id"].to_list()
    with pytest.raises(WrangleError, match="row count"):
        wr.array_random_shuffle(x, y.head(3))


def test_multi_input_shuffle_and_lazy_output():
    x = [pl.Series("a", range(9)), pl.DataFrame({"b": range(9)})]
    y = pl.Series("y", range(9))
    (first, second), labels = wr.array_random_shuffle(x, y, multi_input=True, seed=99)
    assert first.to_list() == second["b"].to_list() == labels.to_list()
    lazy = wr.array_random_shuffle(pl.DataFrame({"id": range(9)}).lazy(), seed=99)
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect()["id"].to_list() == first.to_list()


def test_folds_cover_every_observation_and_no_remainders_lost():
    x = pl.DataFrame({"id": range(23)})
    y = pl.Series("y", range(23))
    folds_x, folds_y = wr.array_to_kfold(x, y, folds=5, seed=17)
    assert [len(value) for value in folds_x] == [5, 5, 5, 4, 4]
    assert pl.concat(folds_x)["id"].sort().to_list() == list(range(23))
    for features, labels in zip(folds_x, folds_y):
        assert features["id"].to_list() == labels.to_list()
    with pytest.raises(WrangleError):
        wr.array_to_kfold(x, y, folds=24)


def test_split_and_batches_retain_alignment_and_all_rows():
    x, y = [list(range(11)), list(range(11))]
    x_train, y_train, x_test, y_test = wr.array_split(x, y, 0.2, shuffled=False)
    assert x_train.to_list() == y_train.to_list() == list(range(8))
    assert x_test.to_list() == y_test.to_list() == [8, 9, 10]
    batches = list(wr.array_to_generator(x, y, batch_size=4, repeat=False))
    assert [len(a) for a, _ in batches] == [4, 4, 3]
    assert pl.concat([a for a, _ in batches]).to_list() == x
    with pytest.raises(WrangleError):
        list(wr.array_to_generator([], [], batch_size=4, repeat=False))


def test_native_conv1d_nested_features():
    x = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = wr.array_reshape_conv1d(x)
    assert result.to_list() == [[[1], [3]], [[2], [4]]]
    assert isinstance(result.dtype, pl.List)
    assert_frame_equal(wr.array_reshape_conv1d(x.lazy()).collect(), result.to_frame())


def test_lstm_windows_include_final_target_and_keep_chronology():
    train_x, train_y, test_x, test_y = wr.array_reshape_lstm(list(range(10)), 3, False, validation_split=0.3, shuffled=False)
    assert len(train_x) + len(test_x) == 7
    assert train_x[0].to_list() == [[0], [1], [2]]
    assert train_y[0] == 3
    assert test_x[-1].to_list() == [[6], [7], [8]]
    assert test_y[-1] == 9
    with pytest.raises(WrangleError) as failure:
        wr.array_reshape_lstm(list(range(10)), 3, True)
    assert failure.value.code == "ZERO_BASELINE"
    values = normalise_windows([[2, 4, 6], [4, 8, 12]])
    assert values.to_list() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]


def test_label_encoding_declared_mapping_and_invalid_values():
    result = wr.array_to_multilabel([2, 0, 2], classes=4)
    assert result.columns == ["class_0", "class_1", "class_2", "class_3"]
    assert result.rows() == [(0, 0, 1, 0), (1, 0, 0, 0), (0, 0, 1, 0)]
    for invalid in [[-1, 0], [0, None], [0.1, 1.0]]:
        with pytest.raises(WrangleError):
            wr.array_to_multilabel(invalid)
    with pytest.raises(WrangleError):
        wr.array_to_multilabel([2], classes=2)


def test_task_semantics_require_explicit_declaration():
    with pytest.raises(WrangleError) as failure:
        wr.array_detect_task([0, 1, 2])
    assert failure.value.code == "TASK_DECLARATION_REQUIRED"
    assert wr.array_detect_task([0, 1], task="binary") == ("binary", 1, "single")
    assert wr.array_detect_task([0, 1, 2], task="categorical") == ("category", 3, "single")
    assert wr.array_detect_task([0.5, 1.5], task="continuous") == ("continuous", 1.0, "single")
    with pytest.raises(WrangleError):
        wr.array_detect_task([0, 2], task="binary")


def test_weighted_sampling_never_selects_zero_weight_rows():
    source = pl.DataFrame({"id": [0, 1, 2], "label": ["a", "b", "c"]})
    result = wr.array_random_weighted(source, [0, 1, 0], 100, seed=31)
    assert result["id"].to_list() == [1] * 100
    assert result["label"].to_list() == ["b"] * 100
    assert_frame_equal(result, wr.array_random_weighted(source, [0, 1, 0], 100, seed=31))
    for weights in [[-1, 1, 1], [0, 0, 0], [1, float("nan"), 1], [1], "normal"]:
        with pytest.raises(WrangleError):
            wr.array_random_weighted(source, weights, 4)


def test_dictionary_sampling_is_immutable_and_complexity_exact():
    source = {"a": [0, 1, 2, 3], "b": ["x", "y", "z"]}
    original = {key: list(value) for key, value in source.items()}
    sampled = wr.dic_resample_values(source, 2, seed=19)
    assert source == original
    assert all(len(value) == 2 for value in sampled.values())
    assert wr.dic_count_complexity(source) == 12
    assert wr.dic_count_complexity({}) == 1
    assert wr.dic_count_complexity({"a": []}) == 0
    with pytest.raises(WrangleError):
        wr.dic_resample_values(source, 4)


def test_dictionary_bucket_percentage_counts():
    data = {"batch": pl.DataFrame({"dose": [0.0, 1.0, 2.0, 3.0], "outcome": [0, 1, 1, 1]})}
    result = wr.dic_corr_perc(data, "outcome", cuts=2, warning_threshold=3)
    assert result["samples"].sum() == 4
    assert result["outcome"].to_list() == [50.0, 100.0]
    assert result["metric"].to_list() == ["dose", "dose"]
    assert result["metric_context"].to_list() == ["batch", "batch"]


def test_calendar_sequences_honor_start_day_and_leap_year():
    result = utils.create_time_sequence(3, 2024, 1, 31, "%Y-%m-%d", "month")
    assert result.to_list() == ["2024-01-31", "2024-02-29", "2024-03-31"]
    assert utils.create_time_sequence(2, 2024, 2, 29, "%Y-%m-%d", "year").to_list() == ["2024-02-29", "2025-02-28"]
    values = utils.create_datetime_col([1, 2, 3], "2024-01-31", "2024-03-31", "month")
    assert len(values) == 3
    with pytest.raises(WrangleError) as failure:
        utils.create_datetime_col([1, 2], "2024-01-31", "2024-03-31", "month")
    assert failure.value.code == "ROW_ALIGNMENT"


def test_datetime_schema_detection_and_sequence_preserve_row_identity():
    data = pl.DataFrame({"id": [3, 1, 2, 4], "when": [datetime(2024, 2, 1), datetime(2024, 1, 1), datetime(2024, 2, 1), None], "text": ["2024-01-01"] * 4})
    assert utils.datetime_detector(data) == ["when"]
    result = utils.datetime_handler(data, "sequence")
    assert result["id"].to_list() == [3, 1, 2, 4]
    assert result["when"].to_list() == [1, 0, 1, None]
    assert utils.datetime_handler(data, "retain").columns == ["when", "id", "text"]
    assert utils.datetime_handler(data, "drop").columns == ["id", "text"]


def test_csv_scanning_never_fabricates_rows_or_casts_without_request(tmp_path):
    path = tmp_path / "measurements.csv"
    path.write_text("id,measurement\n001,0.25\n002,0.5\n")
    plan = utils.read_large_csv(path, n=20)
    assert isinstance(plan, pl.LazyFrame)
    result = plan.collect()
    assert result.height == 2
    assert result["id"].to_list() == ["001", "002"]
    assert result["measurement"].to_list() == ["0.25", "0.5"]
    assert utils.read_large_csv(path, n=1, cols=["measurement"], dtype="float32").collect_schema()["measurement"] == pl.Float32
    with pytest.raises(WrangleError):
        utils.read_large_csv(path, 3, cols=["absent"])


def test_synthetic_fixtures_seeded_native_and_shapes():
    for mode in ["binary", "multi_class", "multi_label", "continuous"]:
        x, y = utils.create_synth_data(mode, n=51, features=3, classes=4, seed=7)
        same_x, same_y = utils.create_synth_data(mode, n=51, features=3, classes=4, seed=7)
        assert_frame_equal(x, same_x)
        assert_frame_equal(y, same_y)
        assert x.shape == (51, 3)
        assert y.shape == (51, 4 if mode == "multi_label" else 1)
        assert x.select(pl.all().is_between(0, 1, closed="left").all()).row(0) == (True, True, True)
    for name in ["create_synth_binary_model", "create_synth_multi_class_model", "create_synth_multi_label_model", "create_synth_regression_model"]:
        with pytest.raises(WrangleError) as failure:
            getattr(utils, name)()
        assert failure.value.code == "MODEL_ENGINE_REQUIRED"


def test_explicit_column_selection_and_type_preservation():
    data = pl.DataFrame({"specimen": ["s1", "s2"], "value": [1.1, 2.2], "label": [False, True]})
    selected = utils.multi_input_support([0, 2], data)
    assert selected.columns == ["specimen", "label"]
    x, y = utils.transform_data(data, X=["specimen", "value"], Y="label", flatten=True)
    assert x.schema == {"specimen": pl.String, "value": pl.Float64}
    assert_series_equal(y, data["label"])
    with pytest.raises(WrangleError):
        utils.multi_input_support([0, 0], data)


def test_string_utilities_keep_missingness_and_literal_meaning():
    data = pl.DataFrame({"text": ["a.b", "axb", None, ""]})
    assert utils.value_starts_with(data, "text").to_list() == ["a", "a", None, ""]
    assert string_contains_to_binary(data, "text", ["a.b"])["a.b"].to_list() == [1, 0, None, 0]
    assert utils.int_to_chars([1, 12, 3]).to_list() == [["0", "1"], ["1", "2"], ["0", "3"]]
    assert utils.is_number("1.5") is True
    for value in [None, "no", "nan", float("inf")]:
        assert utils.is_number(value) is False


def test_category_codes_are_declared_and_unknown_values_blocked():
    data = pl.DataFrame({"group": ["control", "treated", None], "free_text": ["a", "b", "c"]})
    result = utils.to_category_labels(data, 2, categories={"group": ["control", "treated"]})
    assert result["group"].to_list() == [0, 1, None]
    assert result["free_text"].to_list() == ["a", "b", "c"]
    with pytest.raises(WrangleError) as failure:
        utils.to_category_labels(data, 2, categories={"group": ["control"]})
    assert failure.value.code == "UNKNOWN_CATEGORY"


def test_groupby_native_aggregation_and_callback_rejection():
    data = pl.DataFrame({"group": ["a", "a", "b"], "value": [1.0, 3.0, 8.0]})
    result = utils.groupby_func(data.group_by("group", maintain_order=True), "mean")
    assert result["value"].to_list() == [2.0, 8.0]
    native = utils.groupby_func(data.lazy().group_by("group", maintain_order=True), pl.col("value").sum()).collect()
    assert native["value"].to_list() == [4.0, 8.0]
    with pytest.raises(WrangleError):
        utils.groupby_func(data.group_by("group"), lambda values: values.mean())


def test_fill_and_impute_wrappers_use_explicit_mode():
    data = pl.DataFrame({"x": [1.0, 10.0, None]})
    assert filling_nans(data, "x", 0.0)["x"].to_list() == [1.0, 10.0, 0.0]
    assert imputing_nans(data, "x", "mean")["x"].to_list() == [1.0, 10.0, 5.5]
    assert data["x"].to_list() == [1.0, 10.0, None]


def test_entropy_is_native_and_invalid_mass_never_becomes_a_statistic():
    data = pl.DataFrame({"group": ["a", "a", "b"], "mass": [1.0, 1.0, 3.0], "other": [2.0, 2.0, 1.0]})
    result = utils.groupby_func(data.lazy().group_by("group", maintain_order=True), "entropy").collect()
    assert result.columns == ["group", "mass", "other"]
    assert result["mass"].to_list() == pytest.approx([0.6931471805599453, 0.0])
    for masses in [[1.0, -1.0, 3.0], [0.0, 0.0, 3.0], [1.0, None, 3.0], [1.0, float("nan"), 3.0]]:
        with pytest.raises(WrangleError) as failure:
            utils.groupby_func(data.with_columns(pl.Series("mass", masses)).group_by("group"), "entropy")
        assert failure.value.code == "INVALID_ENTROPY"


def test_normalization_preserves_variable_window_lengths():
    assert normalise_windows([[2, 4, 6], [4, 8]]).to_list() == [[0.0, 1.0, 2.0], [0.0, 1.0]]
    for windows in [[[2, None]], [[2, float("inf")]], [[], [2, 4]]]:
        with pytest.raises(WrangleError):
            normalise_windows(windows)


def test_tensor_reshape_blocks_implicit_stringification_and_precision_loss():
    with pytest.raises(WrangleError) as failure:
        wr.array_reshape_conv1d(pl.DataFrame({"label": ["s1"], "measurement": [1.0]}))
    assert failure.value.code == "NON_NUMERIC_COLUMN"
    with pytest.raises(WrangleError) as failure:
        wr.array_reshape_conv1d(pl.DataFrame({"count": [2**53 + 1], "measurement": [1.0]}))
    assert failure.value.code == "LOSSY_CAST"
    assert wr.array_reshape_conv1d(pl.DataFrame({"count": [3], "measurement": [1.5]})).to_list() == [[[3.0], [1.5]]]


def test_completeness_policy_drops_only_declared_missing_observations():
    data = pl.DataFrame({"id": [1, 2, 3, 4], "complete": [1.0, 2.0, 3.0, 4.0], "measurement": [1.0, None, float("nan"), 4.0]})
    high = utils.nan_dropper(data, treshold=0.75)
    assert high.columns == ["id", "complete"]
    assert high["id"].to_list() == [1, 2, 3, 4]
    low = utils.nan_dropper(data, treshold=0.5)
    assert low.columns == data.columns
    assert low["id"].to_list() == [1, 4]
    assert data.height == 4


def test_network_probe_has_bounded_timeout_and_closes_connection(monkeypatch):
    import importlib
    module = importlib.import_module("wrangle.utils.network_check")
    calls = []

    class Connection:
        def __enter__(self):
            calls.append("opened")
            return self

        def __exit__(self, *args):
            calls.append("closed")

    def connect(address, timeout):
        calls.append((address, timeout))
        return Connection()

    monkeypatch.setattr(module.socket, "create_connection", connect)
    assert utils.network_check(host="research.example", port=443, timeout=1.5) is True
    assert calls == [(("research.example", 443), 1.5), "opened", "closed"]
    with pytest.raises(WrangleError):
        utils.network_check(timeout=0)
    assert len(calls) == 3


def test_groupby_rejects_hidden_udf_before_any_callback_executes():
    called = []
    def callback(value):
        called.append(value)
        return value
    data = pl.DataFrame({"group": ["a", "a"], "value": [1, 2]})
    expression = pl.col("value").map_elements(callback, return_dtype=pl.Int64).sum()
    for grouped in [data.group_by("group"), data.lazy().group_by("group")]:
        with pytest.raises(WrangleError) as failure:
            utils.groupby_func(grouped, expression)
        assert failure.value.code == "UNSUPPORTED_CALLBACK"
    hidden_plan = data.lazy().map_batches(callback, schema=data.schema)
    with pytest.raises(WrangleError) as failure:
        utils.groupby_func(hidden_plan.group_by("group"), "mean")
    assert failure.value.code == "UNSUPPORTED_CALLBACK"
    assert called == []


def test_boolean_outcomes_and_missing_float_outcomes_keep_correct_denominators():
    assert wr.array_detect_task([True, False], task="binary") == ("binary", 1, "single")
    flags = pl.DataFrame({"a": [True, False], "b": [False, False]})
    assert wr.array_detect_task(flags, task="multilabel") == ("category", 2, "multilabel")
    boolean = wr.dic_corr_perc({"batch": pl.DataFrame({"dose": [1, 1], "y": [True, False]})}, "y")
    assert boolean["y"].to_list() == [50.0]
    missing = wr.dic_corr_perc({"batch": pl.DataFrame({"dose": [1, 1], "y": [1.0, float("nan")]})}, "y")
    assert missing["samples"].to_list() == [1]
    assert missing["y"].to_list() == [100.0]


def test_category_indicators_preserve_order_and_existing_scratch_names():
    data = pl.DataFrame({"__wr_row": [9, 4, 7], "__wr_row_": [3, 2, 1], "text": ["ok", None, "failed"]})
    result = utils.to_category_labels(data.lazy(), 2, col_that_contains="text", col_contains_strings=["ok"]).collect()
    assert result["__wr_row"].to_list() == [9, 4, 7]
    assert result["__wr_row_"].to_list() == [3, 2, 1]
    assert result["ok"].to_list() == [1, None, 0]
    assert result.columns == ["__wr_row", "__wr_row_", "text", "ok"]
