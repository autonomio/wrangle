"""Protocol field preparation preserves meaning across rows and batches."""
from datetime import date, datetime, timezone

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from wrangle._core import WrangleError, require_native
from wrangle._recipe_fields import OPERATIONS, clean_text, encode, normalize_missing, parse_datetime, recode


def fails(code, call):
    with pytest.raises(WrangleError) as caught:
        call()
    assert caught.value.code == code


def test_declared_missing_codes_are_exact_and_selected():
    source = pl.DataFrame({"id": ["001", "002", "003", "004"], "mass": [-999.0, 0.0, float("nan"), None], "note": ["NA", " NA ", "na", None], "other": ["NA"] * 4})
    result = normalize_missing(source.lazy(), {"mass": [-999], "note": ["NA"]}).collect()
    assert result.to_dict(as_series=False) == {"id": ["001", "002", "003", "004"], "mass": [None, 0.0, None, None], "note": [None, " NA ", "na", None], "other": ["NA"] * 4}
    assert result.schema == source.schema
    assert source["mass"].is_nan().sum() == 1
    assert source["note"][0] == "NA"


def test_nan_normalization_is_declared_and_does_not_invent_string_codes():
    source = pl.DataFrame({"x": [float("nan"), None], "text": ["NaN", "null"]})
    preserved = normalize_missing(source.lazy(), {"x": []}, nan=False).collect()
    assert preserved["x"].is_nan().sum() == 1
    normalized = normalize_missing(source.lazy(), {"x": []}).collect()
    assert normalized["x"].null_count() == 2
    assert normalized["text"].to_list() == ["NaN", "null"]


@pytest.mark.parametrize("codes,code", [(["-999"], "INVALID_MAPPING"), ([True], "INVALID_MAPPING"), ([0.5], "LOSSY_CAST"), ([2**64], "LOSSY_CAST")])
def test_missing_codes_cannot_coerce_integer_measurements(codes, code):
    source = pl.DataFrame({"mass": [0, 1, None]})
    fails(code, lambda: normalize_missing(source.lazy(), {"mass": codes}))


def test_missing_normalization_preserves_enum_domain():
    source = pl.DataFrame({"label": ["NA", "a", None]}).with_columns(pl.col("label").cast(pl.Enum(["NA", "a"])))
    result = normalize_missing(source.lazy(), {"label": ["NA"]}).collect()
    assert result.schema == source.schema
    assert result["label"].to_list() == [None, "a", None]


def test_text_cleaning_is_selected_ordered_unicode_and_null_preserving():
    source = pl.DataFrame({"id": [" 001 ", " 002 ", "003"], "group": ["  Cafe\u0301 ", "Ａ", None], "n": [1, 2, 3]})
    result = clean_text(source.lazy(), ["group"], unicode="NFKC", case="lower").collect()
    assert result["group"].to_list() == ["café", "a", None]
    assert result["id"].to_list() == source["id"].to_list()
    assert result["n"].to_list() == source["n"].to_list()
    assert result.schema == source.schema
    assert_frame_equal(source, pl.DataFrame({"id": [" 001 ", " 002 ", "003"], "group": ["  Cafe\u0301 ", "Ａ", None], "n": [1, 2, 3]}))


def test_case_only_cleaning_can_preserve_whitespace():
    result = clean_text(pl.DataFrame({"x": [" a ", None]}).lazy(), ["x"], trim=False, case="upper").collect()
    assert result["x"].to_list() == [" A ", None]


@pytest.mark.parametrize("arguments", [{"columns": []}, {"columns": ["x", "x"]}, {"columns": ["absent"]}, {"columns": ["x"], "trim": 1}, {"columns": ["x"], "case": "fold"}, {"columns": ["x"], "unicode": []}])
def test_bad_cleaning_decisions_fail_before_mutation(arguments):
    source = pl.DataFrame({"x": ["a"]}).lazy()
    with pytest.raises(WrangleError):
        clean_text(source, **arguments)


def test_text_cleaning_does_not_implicitly_convert_measurements():
    fails("INVALID_DTYPE", lambda: clean_text(pl.DataFrame({"x": [1, 2]}).lazy(), ["x"]))


def test_datetime_format_and_date_type_are_explicit():
    source = pl.DataFrame({"id": ["001", "002", "003"], "collected": ["30/09/2026 13:14:15", None, "01/10/2026 01:02:03"]})
    result = parse_datetime(source.lazy(), ["collected"], "%d/%m/%Y %H:%M:%S").collect()
    assert result["collected"].to_list() == [datetime(2026, 9, 30, 13, 14, 15), None, datetime(2026, 10, 1, 1, 2, 3)]
    assert result["id"].to_list() == source["id"].to_list()
    assert result.schema["collected"] == pl.Datetime("us")
    dates = parse_datetime(pl.DataFrame({"d": ["30/09/2026", None]}).lazy(), ["d"], "%d/%m/%Y", dtype="Date").collect()
    assert dates["d"].to_list() == [date(2026, 9, 30), None]


def test_malformed_dates_fail_or_become_null_only_when_declared():
    source = pl.DataFrame({"d": ["2026-09-30", "2026-02-30", " 2026-09-30 ", None]}).lazy()
    fails("INVALID_DATETIME", lambda: parse_datetime(source, ["d"], "%Y-%m-%d", dtype="Date"))
    result = parse_datetime(source, ["d"], "%Y-%m-%d", dtype="Date", invalid="null").collect()
    assert result["d"].to_list() == [date(2026, 9, 30), None, None, None]


def test_fractional_time_precision_is_checked_before_truncation():
    source = pl.DataFrame({"d": ["2026-09-30 13:14:15.123456789"]}).lazy()
    fmt = "%Y-%m-%d %H:%M:%S%.f"
    fails("DATETIME_PRECISION_LOSS", lambda: parse_datetime(source, ["d"], fmt))
    nanos = parse_datetime(source, ["d"], fmt, time_unit="ns").collect()
    assert nanos["d"].dt.nanosecond().item() == 123456789
    micros = parse_datetime(source, ["d"], fmt, precision_loss="truncate").collect()
    assert micros["d"].dt.nanosecond().item() == 123456000
    fails("DATETIME_PRECISION_LOSS", lambda: parse_datetime(source, ["d"], fmt, time_unit="ms"))


def test_nanosecond_date_range_cannot_wrap_historical_years():
    source = pl.DataFrame({"d": ["1500-01-01 01:02:03.123456"]}).lazy()
    fmt = "%Y-%m-%d %H:%M:%S%.f"
    fails("DATETIME_RANGE", lambda: parse_datetime(source, ["d"], fmt, time_unit="ns"))
    result = parse_datetime(source, ["d"], fmt).collect()
    assert result["d"].to_list() == [datetime(1500, 1, 1, 1, 2, 3, 123456)]


def test_offset_timestamps_require_output_zone_and_preserve_instants():
    source = pl.DataFrame({"d": ["2026-09-30T13:00:00+03:00", "2026-09-30T10:00:00Z"]}).lazy()
    fails("UNDECLARED_TIME_ZONE", lambda: parse_datetime(source, ["d"], "%+"))
    result = parse_datetime(source, ["d"], "%+", time_zone="UTC").collect()
    assert result.schema["d"] == pl.Datetime("us", "UTC")
    assert result["d"].to_list() == [datetime(2026, 9, 30, 10, tzinfo=timezone.utc)] * 2


def test_dst_ambiguous_time_requires_declared_resolution():
    source = pl.DataFrame({"d": ["2021-11-07 01:30:00"]}).lazy()
    args = dict(columns=["d"], format="%Y-%m-%d %H:%M:%S", time_zone="America/New_York")
    fails("INVALID_DATETIME", lambda: parse_datetime(source, **args))
    early = parse_datetime(source, **args, ambiguous="earliest").collect()["d"].cast(pl.Int64).item()
    late = parse_datetime(source, **args, ambiguous="latest").collect()["d"].cast(pl.Int64).item()
    assert late - early == 3_600_000_000
    assert parse_datetime(source, **args, ambiguous="null").collect()["d"].null_count() == 1


def test_dst_nonexistent_time_is_not_silently_shifted():
    source = pl.DataFrame({"d": ["2021-03-14 02:30:00"]}).lazy()
    args = dict(columns=["d"], format="%Y-%m-%d %H:%M:%S", time_zone="America/New_York")
    fails("INVALID_DATETIME", lambda: parse_datetime(source, **args))
    assert parse_datetime(source, **args, nonexistent="null").collect()["d"].null_count() == 1


@pytest.mark.parametrize("arguments", [{"format": ""}, {"format": "%Y %Z"}, {"format": "%Y", "dtype": "Date", "time_zone": "UTC"}, {"format": "%Y", "time_unit": "seconds"}, {"format": "%Y", "time_zone": "Invalid/Zone"}])
def test_invalid_datetime_decisions_fail_explicitly(arguments):
    with pytest.raises(WrangleError):
        parse_datetime(pl.DataFrame({"d": ["2026"]}).lazy(), ["d"], **arguments)


def test_recode_declares_synonyms_and_output_domain_without_changing_rows():
    source = pl.DataFrame({"id": ["001", "002", "003"], "group": ["ctrl", "control", None]}).lazy()
    result = recode(source, "group", {"ctrl": "control", "control": "control"}, dtype="String", domain=["control", "treated"]).collect()
    assert result.to_dict(as_series=False) == {"id": ["001", "002", "003"], "group": ["control", "control", None]}


def test_unknown_recoding_requires_error_keep_or_null_policy():
    source = pl.DataFrame({"x": ["a", "unseen", None]}).lazy()
    fails("UNKNOWN_CATEGORY", lambda: recode(source, "x", {"a": "A"}, dtype="String"))
    assert recode(source, "x", {"a": "A"}, dtype="String", unknown="keep").collect()["x"].to_list() == ["A", "unseen", None]
    assert recode(source, "x", {"a": "A"}, dtype="String", unknown="null").collect()["x"].to_list() == ["A", None, None]
    fails("INVALID_POLICY", lambda: recode(source, "x", {"a": 1}, dtype="Int64", unknown="keep"))


def test_typed_numeric_recode_preserves_large_integer_codes():
    value = 2**53 + 1
    source = pl.DataFrame({"x": [value, 0, None]}).lazy()
    result = recode(source, "x", [[value, "large"], [0, "zero"]], dtype="String").collect()
    assert result["x"].to_list() == ["large", "zero", None]
    fails("INVALID_MAPPING", lambda: recode(source, "x", {str(value): "large"}, dtype="String"))
    fails("LOSSY_CAST", lambda: recode(source, "x", [[value, value]], dtype="Float64", unknown="null"))


def test_null_recoding_needs_an_explicit_null_mapping():
    source = pl.DataFrame({"x": ["a", None]}).lazy()
    fails("MISSING_CATEGORY", lambda: recode(source, "x", {"a": "A"}, dtype="String", nulls="error"))
    assert recode(source, "x", [["a", "A"], [None, "missing"]], dtype="String", nulls="map").collect()["x"].to_list() == ["A", "missing"]
    fails("INVALID_POLICY", lambda: recode(source, "x", [["a", "A"], [None, "missing"]], dtype="String"))
    fails("INVALID_POLICY", lambda: recode(source, "x", {"a": "A"}, dtype="String", nulls="map"))


def test_recoding_domains_and_duplicate_source_codes_are_checked():
    source = pl.DataFrame({"x": [1, 2]}).lazy()
    fails("DUPLICATE_MAPPING", lambda: recode(source, "x", [[1, "a"], [1, "b"]], dtype="String"))
    fails("CATEGORY_DOMAIN", lambda: recode(source, "x", [[1, "a"], [2, "b"]], dtype="String", domain=["a"]))
    fails("LOSSY_CAST", lambda: recode(source, "x", [[1, 0.5], [2, 1]], dtype="Int64"))
    fails("INVALID_MAPPING", lambda: recode(source, "x", [[1, 2], [2, True]], dtype="Boolean"))


def test_ordinal_encoding_meanings_are_fixed_across_batches():
    first = pl.DataFrame({"x": ["b", "c", None]}).lazy()
    second = pl.DataFrame({"x": ["a", "b", "c"]}).lazy()
    for source, expected in [(first, [1, 2, None]), (second, [0, 1, 2])]:
        assert encode(source, "x", categories=["a", "b", "c"]).collect()["x"].to_list() == expected
    assert encode(first, "x", mapping={"a": 10, "b": 20, "c": 30}).collect()["x"].to_list() == [20, 30, None]


def test_fixed_one_hot_schema_survives_absent_categories_and_missing_values():
    args = dict(column="group", categories=["control", "treated", "vehicle"], names=["is_control", "is_treated", "is_vehicle"], mode="one_hot")
    first = pl.DataFrame({"id": ["001", "002", "003"], "group": ["control", "treated", None]})
    second = pl.DataFrame({"id": ["004"], "group": ["vehicle"]})
    a, b = encode(first.lazy(), **args).collect(), encode(second.lazy(), **args).collect()
    assert a.schema == b.schema
    assert a.to_dict(as_series=False) == {"id": ["001", "002", "003"], "is_control": [1, 0, None], "is_treated": [0, 1, None], "is_vehicle": [0, 0, None]}
    assert b.to_dict(as_series=False) == {"id": ["004"], "is_control": [0], "is_treated": [0], "is_vehicle": [1]}
    assert first.columns == ["id", "group"]


def test_explicit_missing_category_is_not_an_unknown_category():
    source = pl.DataFrame({"x": ["a", None, "unseen"]}).lazy()
    result = encode(source, "x", categories=["a", None], names=["a", "missing"], mode="one_hot", nulls="category", unknown="null").collect()
    assert result.to_dict(as_series=False) == {"a": [1, 0, None], "missing": [0, 1, None]}
    ordinal = encode(source, "x", categories=["a", None], nulls="category", unknown="null").collect()
    assert ordinal["x"].to_list() == [0, 1, None]


def test_all_missing_batch_keeps_configured_one_hot_schema():
    source = pl.DataFrame({"x": [None, None]}).lazy()
    result = encode(source, "x", categories=["a", "b"], names=["a", "b"], mode="one_hot").collect()
    assert result.to_dict(as_series=False) == {"a": [None, None], "b": [None, None]}
    assert result.schema == {"a": pl.UInt8, "b": pl.UInt8}


def test_one_hot_can_keep_source_and_rejects_output_collisions():
    source = pl.DataFrame({"id": [1], "x": ["a"]}).lazy()
    result = encode(source, "x", categories=["a"], names=["is_a"], mode="one_hot", drop=False).collect()
    assert result.to_dict(as_series=False) == {"id": [1], "x": ["a"], "is_a": [1]}
    fails("COLUMN_COLLISION", lambda: encode(source, "x", categories=["a"], names=["id"], mode="one_hot"))


@pytest.mark.parametrize("arguments", [
    {"categories": ["a", "a"]}, {"mapping": {"a": 0, "b": 0}},
    {"categories": ["a", None]}, {"categories": ["a"], "nulls": "category"},
    {"mode": "one_hot", "categories": ["a"], "names": []},
    {"mapping": {"a": True}}, {"categories": ["a"], "unknown": "keep"},
])
def test_encoding_declaration_errors_fail_before_transforming(arguments):
    with pytest.raises(WrangleError):
        encode(pl.DataFrame({"x": ["a"]}).lazy(), "x", **arguments)


def test_field_plans_remain_native_and_reject_hidden_callbacks():
    calls = []
    unsafe = pl.DataFrame({"x": ["a"]}).lazy().with_columns(pl.col("x").map_elements(lambda value: calls.append(value) or value, return_dtype=pl.String))
    fails("UNSUPPORTED_CALLBACK", lambda: clean_text(unsafe, ["x"]))
    assert calls == []
    safe = clean_text(pl.DataFrame({"x": [" A "]}).lazy(), ["x"], case="lower")
    require_native(safe)
    assert safe.collect()["x"].to_list() == ["a"]
    assert set(OPERATIONS) == {"normalize_missing", "clean_text", "parse_datetime", "recode", "encode"}


@pytest.mark.parametrize("value,format", [
    ("2020-01-01 00:00:00.1234567899", "%Y-%m-%d %H:%M:%S%.f"),
    ("01.01.2020 00:00:00.1234567899", "%d.%m.%Y %H:%M:%S%.f"),
    ("2020-01-01T00:00:00.1234567899Z", "%+"),
])
def test_subnanosecond_digits_require_explicit_truncation(value, format):
    source = pl.DataFrame({"d": [value]}).lazy()
    args = {"time_zone": "UTC"} if format == "%+" else {}
    fails("DATETIME_PRECISION_LOSS", lambda: parse_datetime(source, ["d"], format, time_unit="ns", **args))
    result = parse_datetime(source, ["d"], format, time_unit="ns", precision_loss="truncate", **args).collect()
    assert result["d"].dt.nanosecond().item() == 123456789


def test_leap_second_does_not_silently_become_next_day():
    source = pl.DataFrame({"d": ["2020-01-01 23:59:60"]}).lazy()
    fmt = "%Y-%m-%d %H:%M:%S"
    fails("INVALID_DATETIME", lambda: parse_datetime(source, ["d"], fmt))
    assert parse_datetime(source, ["d"], fmt, invalid="null").collect()["d"].null_count() == 1


def test_date_only_text_can_be_explicitly_parsed_as_midnight_datetime():
    result = parse_datetime(pl.DataFrame({"d": ["2026-09-30"]}).lazy(), ["d"], "%Y-%m-%d").collect()
    assert result["d"].to_list() == [datetime(2026, 9, 30)]


def test_invalid_null_policy_precedes_fraction_precision_checks():
    source = pl.DataFrame({"d": ["2020-02-31 00:00:00.1234567899"]}).lazy()
    result = parse_datetime(source, ["d"], "%Y-%m-%d %H:%M:%S%.f", invalid="null").collect()
    assert result["d"].null_count() == 1


def test_invalid_null_leap_second_does_not_report_nanosecond_overflow():
    source = pl.DataFrame({"d": ["2020-01-01 23:59:60"]}).lazy()
    result = parse_datetime(source, ["d"], "%Y-%m-%d %H:%M:%S", time_unit="ns", invalid="null").collect()
    assert result["d"].null_count() == 1


def test_mapping_integer_outside_native_precision_returns_a_stable_error():
    source = pl.DataFrame({"x": [1]}).lazy()
    fails("INVALID_MAPPING", lambda: normalize_missing(source, {"x": [2**200]}))
    fails("INVALID_MAPPING", lambda: recode(source, "x", [[1, 2**200]], dtype="Int128"))


@pytest.mark.parametrize("value,format,options", [
    ("2026-09-30 09:00", "%Y-%m-%d %H:%M", {"ambiguous": "earliest"}),
    ("2026-09-30T09:00:00Z", "%+", {"time_zone": "UTC", "nonexistent": "null"}),
])
def test_inapplicable_dst_decisions_cannot_be_silently_ignored(value, format, options):
    fails("INVALID_POLICY", lambda: parse_datetime(pl.DataFrame({"d": [value]}).lazy(), ["d"], format, **options))
