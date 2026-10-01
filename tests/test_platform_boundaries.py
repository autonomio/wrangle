"""Installed dependencies support scientific time handling without an OS database."""
import json
import os
import subprocess
import sys


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
