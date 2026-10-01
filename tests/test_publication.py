"""Existing scientific results survive destination races in both engines."""
from pathlib import Path
from types import SimpleNamespace
import ctypes
import errno

import polars as pl
import pytest

import wrangle as wr
from wrangle import _publication


@pytest.mark.parametrize("execution", ["memory", "disk"])
def test_destination_appearing_at_publication_is_never_replaced(tmp_path, monkeypatch, execution):
    destination = tmp_path / "result"
    original = _publication.publish_directory
    created = {}

    def race(source, target):
        target.mkdir()
        created["inode"] = target.stat().st_ino
        return original(source, target)

    monkeypatch.setattr("wrangle._storage.publish_directory" if execution == "disk" else "wrangle._api.publish_directory", race)
    with pytest.raises(wr.WrangleError) as failure:
        wr.prepare(pl.DataFrame({"id": [1]}), {}, execution=execution, output=destination)
    assert failure.value.code == "OUTPUT_EXISTS"
    assert destination.stat().st_ino == created["inode"]
    assert list(destination.iterdir()) == []
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("platform,name,flags", [("darwin", "renamex_np", 4), ("linux", "renameat2", 1)])
def test_exclusive_publication_uses_native_controls(monkeypatch, platform, name, flags):
    calls = []
    def native(*arguments):
        calls.append(arguments)
        return 0
    monkeypatch.setattr(_publication.sys, "platform", platform)
    monkeypatch.setattr(_publication.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(**{name: native}))
    _publication.publish_directory(Path("source"), Path("target"))
    assert calls[0][-1] == flags
    assert b"source" in calls[0] and b"target" in calls[0]
    assert native.restype == ctypes.c_int


@pytest.mark.parametrize("code,expected", [(errno.EEXIST, "OUTPUT_EXISTS"), (errno.ENOSYS, "ATOMIC_PUBLICATION_UNSUPPORTED"), (errno.EINVAL, "ATOMIC_PUBLICATION_UNSUPPORTED"), (errno.ENOTSUP, "ATOMIC_PUBLICATION_UNSUPPORTED")])
def test_publication_fails_closed_without_exclusive_filesystem_control(monkeypatch, code, expected):
    def rejected(*arguments):
        return -1
    monkeypatch.setattr(_publication.sys, "platform", "linux")
    monkeypatch.setattr(_publication.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(renameat2=rejected))
    monkeypatch.setattr(_publication.ctypes, "get_errno", lambda: code)
    with pytest.raises(wr.WrangleError) as failure:
        _publication.publish_directory(Path("source"), Path("target"))
    assert failure.value.code == expected


def test_publication_preserves_real_io_errors(monkeypatch):
    def denied(*arguments):
        return -1
    monkeypatch.setattr(_publication.sys, "platform", "linux")
    monkeypatch.setattr(_publication.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(renameat2=denied))
    monkeypatch.setattr(_publication.ctypes, "get_errno", lambda: errno.EACCES)
    with pytest.raises(PermissionError):
        _publication.publish_directory(Path("source"), Path("target"))
