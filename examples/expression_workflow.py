"""Verified JSON expressions: bounded text, conditions, precision and time.

Run this file with installed Wrangle and Polars. No callback or expression text
executes Python. Dates use declared calendar rules; timestamps use named zones.
A calendar day preserves local clock time; an elapsed day lasts 24 hours.
Every derived measurement declares its unit and scientific meaning explicitly.
"""
from datetime import datetime, timezone

import polars as pl
import wrangle as wr


def _shift(value, offset, mode="calendar", month_end="error"):
    return {"datetime_shift": {"value": value, "offset": offset, "mode": mode, "precision": "exact", "month_end": month_end}}


def _fails(source, expression, code):
    try:
        wr.prepare(source, {"steps": [{"op": "derive", "columns": {"derived": expression}}]})
    except wr.WrangleError as error:
        assert error.code == code, (error.code, error.details)
    else:
        raise AssertionError(f"Expected {code}")


def run():
    source = pl.DataFrame({
        "specimen": ["001", "002"], "label": [" A:1 ", " B:2 "],
        "mass_mg": [1.25, None],
        "collected": [datetime(2024, 3, 30, 10, tzinfo=timezone.utc), datetime(2024, 3, 31, 10, tzinfo=timezone.utc)],
    })
    label = {"lower": {"trim": {"col": "label"}}}
    local = {"convert_time_zone": {"value": {"col": "collected"}, "time_zone": "Europe/Helsinki"}}
    classification = {"when": {"if": {"gt": [{"col": "mass_mg"}, 2.0]}, "then": "above", "else": "at_or_below", "nulls": "null"}}
    recipe = {
        "version": 1, "name": "Declared expression protocol", "key": ["specimen"],
        "units": {"mass_mg": "mg"},
        "steps": [{"op": "derive", "columns": {
            "normalized_label": label,
            "label_parts": {"split": {"value": label, "separator": ":", "max_splits": 1, "nulls": "error"}},
            "assessment": classification,
            "bounded_mass_mg": {"clip": {"value": {"col": "mass_mg"}, "min": 0.0, "max": 2.0, "nulls": "keep"}},
            "rounded_mass_mg": {"round": {"value": {"col": "mass_mg"}, "decimals": 1, "mode": "half_to_even"}},
            "local_time": local,
            "next_calendar_day": _shift(local, "1d"),
            "next_elapsed_day": _shift(local, "1d", mode="elapsed"),
            "weekday": {"date_part": {"value": local, "part": "weekday"}},
        }, "units": {"bounded_mass_mg": "mg", "rounded_mass_mg": "mg", "weekday": "1"}, "descriptions": {
            "assessment": "Whether observed mass exceeds 2 mg; missing mass remains unknown",
            "bounded_mass_mg": "Mass clipped to the predeclared 0..2 mg analysis interval",
            "rounded_mass_mg": "Mass rounded to one decimal place with ties to even",
            "next_calendar_day": "Same local clock time on the following Helsinki calendar date",
            "next_elapsed_day": "Instant exactly 24 hours after collection",
            "weekday": "ISO weekday in Helsinki, Monday=1 through Sunday=7",
        }}],
        "checks": {"row_count": {"exact": 2}, "ranges": {"bounded_mass_mg": {"min": 0, "max": 2}}},
    }
    result = wr.prepare(source, recipe)
    replay = wr.prepare(source, recipe)
    assert result.data.equals(replay.data) and result.receipt == replay.receipt
    assert result.data["normalized_label"].to_list() == ["a:1", "b:2"]
    assert result.data["label_parts"].to_list() == [["a", "1"], ["b", "2"]]
    assert result.data["assessment"].to_list() == ["at_or_below", None]
    assert result.data["rounded_mass_mg"].to_list() == [1.2, None]
    assert result.data["bounded_mass_mg"].to_list() == [1.25, None]
    assert result.data["next_calendar_day"][0].hour == 12
    assert result.data["next_elapsed_day"][0].hour == 13
    assert result.data["weekday"].to_list() == [6, 7]
    assert result.receipt["units"]["rounded_mass_mg"] == "mg"
    assert source.columns == ["specimen", "label", "mass_mg", "collected"]

    _fails(source, {"when": {**classification["when"], "nulls": "error"}}, "UNRESOLVED_EXPRESSION")
    _fails(pl.DataFrame({"collected": [datetime(2024, 1, 1)]}), local, "INVALID_EXPRESSION")
    _fails(pl.DataFrame({"x": [2**53 + 1]}), {"cast": {"value": {"col": "x"}, "dtype": "Float64", "invalid": "error", "precision": "exact"}}, "LOSSY_CAST")
    _fails(source, _shift(local, "1ns", mode="elapsed"), "LOSSY_CAST")
    _fails(pl.DataFrame({"collected": [datetime(2024, 1, 31)]}), _shift({"col": "collected"}, "1mo"), "INVALID_DATETIME_SHIFT")
    return result


if __name__ == "__main__":
    run()
    print("Verified expression protocol, exact replay and five scientific failure contracts.")
