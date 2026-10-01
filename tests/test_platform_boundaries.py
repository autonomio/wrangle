"""Portable time handling and UTF-8 CLI output preserve scientific data."""
import json
import io
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

from wrangle._cli import main


def test_named_time_zone_round_trip_without_system_iana_database():
    code = """import json
from datetime import datetime, timezone
from zoneinfo import TZPATH, ZoneInfo
import polars as pl
import wrangle
assert not TZPATH
assert ZoneInfo('UTC').key == 'UTC'
assert ZoneInfo('Europe/Helsinki').key == 'Europe/Helsinki'
source = pl.DataFrame({'id': ['001'], 'time': [datetime(2024, 3, 31, 0, 30, tzinfo=timezone.utc)]})
result = wrangle.prepare(source, {'key': ['id'], 'steps': [{'op': 'derive', 'columns': {
    'local_time': {'convert_time_zone': {'value': {'col': 'time'}, 'time_zone': 'Europe/Helsinki'}}
}}]})
assert result.data['time'].cast(pl.Int64).to_list() == result.data['local_time'].cast(pl.Int64).to_list()
print(json.dumps(result.data['local_time'][0].isoformat()))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONTZPATH": ""},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    assert json.loads(result.stdout) == "2024-03-31T02:30:00+02:00"


@pytest.mark.parametrize("entrypoint", ["module", "console"])
@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("exists", [False, True])
def test_cli_pipes_use_utf8_under_legacy_locale(tmp_path, entrypoint, as_json, exists):
    source = tmp_path / "測定 Δsamples.csv"
    if exists:
        source.write_text("specimen,Δmass,condition\n001,1.25,測定\n", encoding="utf-8")
    if entrypoint == "module":
        command = [sys.executable, "-m", "wrangle"]
    else:
        executable = shutil.which("wrangle", path=sysconfig.get_path("scripts"))
        assert executable is not None, "Install Wrangle to test its real console entrypoint."
        command = [executable]
    arguments = ["inspect", str(source), *(["--json"] if as_json else [])]
    result = subprocess.run(
        [*command, *arguments], cwd=tmp_path,
        env={**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"},
        capture_output=True, check=False,
    )
    assert result.returncode == (0 if exists else 1), result.stderr
    output = result.stdout if exists else result.stderr
    assert not (result.stderr if exists else result.stdout)
    text = output.decode("utf-8", errors="strict")
    assert "Δ" in text and "測定" in text
    if as_json:
        document = json.loads(text)
        if exists:
            assert document["columns"]["Δmass"] == "String"
            assert document["examples"][0]["condition"] == "測定"
        else:
            assert document["code"] == "SOURCE_NOT_FOUND"
            assert document["details"]["path"] == str(source.resolve())


def test_import_preserves_embedding_application_stdio_encoding():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; before = (sys.stdout.encoding, sys.stderr.encoding); import wrangle._cli; assert before == ('cp1252', 'cp1252'); assert (sys.stdout.encoding, sys.stderr.encoding) == before"],
        env={**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp1252"},
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout and not result.stderr


def test_cli_leaves_embedding_application_custom_streams_untouched(tmp_path, monkeypatch):
    source = tmp_path / "測定.csv"
    source.write_text("Δmass\n1.25\n", encoding="utf-8")
    output, errors = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    assert main(["inspect", str(source)]) == 0
    assert sys.stdout is output and sys.stderr is errors
    assert "Δmass" in output.getvalue() and not errors.getvalue()
