"""Disk checkpoints and streaming native execution; no bounded-RAM promise."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import polars as pl

from ._core import WrangleError
from ._publication import publish_directory

_EXECUTION = ContextVar("wrangle_execution", default="memory")


@contextmanager
def execution_context(mode):
    token = _EXECUTION.set(mode)
    try:
        yield
    finally:
        _EXECUTION.reset(token)


def collect(plan):
    """Collect native reductions using this call's engine, without global config."""
    return plan.collect(engine="streaming" if _EXECUTION.get() == "disk" else "auto")


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def logical_digest(table):
    """Hash the existing logical-native-json-v1 format without a full table/JSON."""
    from ._api import _canonical_value, _json
    schema = table.schema
    descriptor = _json({"rows": table.height, "columns": [(name, str(dtype)) for name, dtype in schema.items()]}).encode()
    checksum = hashlib.sha256()
    checksum.update(len(descriptor).to_bytes(8, "big"))
    checksum.update(descriptor)
    checksum.update(b"[")
    with tempfile.TemporaryDirectory(prefix="wrangle-hash-", dir=table.path.parent if table.path else None) as directory:
        path = Path(directory) / "values.ndjson"
        canonical = table.lazy().select([_canonical_value(pl.col(name), dtype).alias(name) for name, dtype in schema.items()])
        try:
            canonical.sink_ndjson(path, maintain_order=True, engine="streaming")
            with path.open("rb") as stream:
                first = True
                for line in stream:
                    if not first:
                        checksum.update(b",")
                    checksum.update(line.rstrip(b"\n"))
                    first = False
        except (pl.exceptions.PolarsError, TypeError) as error:
            raise WrangleError("UNSUPPORTED_DTYPE", "Research data must use serializable native Polars dtypes.") from error
    checksum.update(b"]")
    return checksum.hexdigest()


class DiskTable:
    """Private table boundary: native plan plus cached metadata, never eager rows."""

    def __init__(self, plan, *, path=None):
        self._plan = plan.lazy() if isinstance(plan, pl.DataFrame) else plan
        self.path = Path(path) if path is not None else None
        self.schema = self._plan.collect_schema()
        self.columns = self.schema.names()
        self._height = None
        self._logical_digest = None

    @property
    def height(self):
        if self._height is None:
            self._height = collect(self._plan.select(pl.len())).item()
        return self._height

    @property
    def width(self):
        return len(self.columns)

    @property
    def logical_digest(self):
        if self._logical_digest is None:
            self._logical_digest = logical_digest(self)
        return self._logical_digest

    def lazy(self):
        return self._plan.clone()

    def select(self, *args, **kwargs):
        return DiskTable(self._plan.select(*args, **kwargs))

    def rename(self, *args, **kwargs):
        return DiskTable(self._plan.rename(*args, **kwargs))

    def unique(self, *args, **kwargs):
        return DiskTable(self._plan.unique(*args, **kwargs))

    def join(self, other, *args, **kwargs):
        return DiskTable(self._plan.join(other.lazy(), *args, **kwargs))

    def equals(self, other):
        if self.schema != other.schema or self.height != other.height:
            return False
        if not self.columns:
            return True
        left = self.lazy().select(pl.struct(self.columns).alias("left")).with_row_index("row")
        right = other.lazy().select(pl.struct(other.columns).alias("right")).with_row_index("row")
        compared = left.join(right, on="row", validate="1:1", maintain_order="left")
        return not collect(compared.select((~pl.col("left").eq_missing(pl.col("right"))).any())).item()


def verify_evidence(directory, receipt):
    """Verify all declared evidence files before trusting a saved result."""
    root = Path(directory).resolve()
    entries = receipt.get("evidence_files", [])
    if not isinstance(entries, list):
        raise WrangleError("INVALID_BUNDLE", "Evidence files must be declared as a list.")
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise WrangleError("INVALID_BUNDLE", "Evidence must declare relative file paths and checksums.")
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "evidence" or not path.is_relative_to(root) or entry["path"] in names:
            raise WrangleError("INVALID_BUNDLE", "Evidence paths must be distinct files inside the bundle's evidence directory.")
        names.add(entry["path"])
        if not path.is_file() or file_digest(path) != entry.get("sha256"):
            raise WrangleError("BUNDLE_MISMATCH", "The saved observation evidence changed.", {"path": entry["path"]})
    return entries


def verify_report(directory, receipt):
    """Verify the human report whenever its checksum is present in the receipt."""
    expected = receipt.get("report_sha256")
    if expected is None:
        return False  # Historical bundles have no human report manifest.
    path = Path(directory) / "report.txt"
    if not isinstance(expected, str) or not path.is_file() or file_digest(path) != expected:
        raise WrangleError("BUNDLE_MISMATCH", "The saved preparation report changed; restore it from the checked receipt.", {"path": "report.txt"})
    return True


class DiskWorkspace:
    """Own temporary snapshots, evidence and one atomic new-directory publication."""

    def __init__(self, destination):
        self.destination = Path(destination).expanduser().resolve()
        self.root = None
        self.lock = self.destination.parent / (self.destination.name + ".wrangle-lock")
        self._locked = False
        self._sequence = 0
        self.evidence_files = []

    def __enter__(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise WrangleError("OUTPUT_BUSY", "Another preparation is publishing this destination.", {"path": str(self.destination)}) from error
        os.close(descriptor)
        self._locked = True
        try:
            if self.destination.exists():
                raise WrangleError("OUTPUT_EXISTS", "Choose a new output directory; existing results are never overwritten.", {"path": str(self.destination)})
            self.root = Path(tempfile.mkdtemp(prefix=".wrangle-disk-", dir=self.destination.parent))
            (self.root / "checkpoints").mkdir()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
        if self._locked:
            self.lock.unlink(missing_ok=True)
            self._locked = False

    def snapshot(self, plan):
        path = self.root / "checkpoints" / f"{self._sequence:06d}.parquet"
        self._sequence += 1
        plan = plan.lazy() if isinstance(plan, (DiskTable, pl.DataFrame)) else plan
        schema = plan.collect_schema()
        plan.sink_parquet(path, maintain_order=True, engine="streaming", row_group_size=65536)
        result = DiskTable(pl.scan_parquet(path, glob=False), path=path)
        if result.schema != schema:
            raise WrangleError("UNSUPPORTED_DTYPE", "Disk checkpoints must retain exact native dtypes.", {"expected": str(schema), "actual": str(result.schema)})
        return result

    def record(self, plan, kind):
        plan = plan.lazy() if isinstance(plan, (DiskTable, pl.DataFrame)) else plan
        directory = self.root / "evidence"
        directory.mkdir(exist_ok=True)
        path = directory / f"{len(self.evidence_files):06d}-{kind}.parquet"
        plan.sink_parquet(path, maintain_order=True, engine="streaming", row_group_size=65536)
        table = DiskTable(pl.scan_parquet(path, glob=False), path=path)
        entry = {"path": str(path.relative_to(self.root)), "format": "parquet", "rows": table.height, "columns": {name: str(dtype) for name, dtype in table.schema.items()}, "sha256": file_digest(path), "snapshot_sha256": table.logical_digest}
        self.evidence_files.append(entry)
        return dict(entry)

    def publish(self, table, receipt, *, evidence_source=None):
        from ._api import Prepared, _json
        final = self.root / "published"
        final.mkdir()
        if table.path is None:
            table = self.snapshot(table.lazy())
        before = file_digest(table.path)
        shutil.copyfile(table.path, final / "data.parquet")
        if file_digest(table.path) != before or file_digest(final / "data.parquet") != before:
            raise WrangleError("SOURCE_CHANGED", "The verified table changed while copying its publication.")
        entries = receipt.get("evidence_files", [])
        if entries:
            source = Path(evidence_source) if evidence_source is not None else self.root
            verify_evidence(source, receipt)
            for entry in entries:
                target = final / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / entry["path"], target)
            verify_evidence(final, receipt)
        (final / "receipt.json").write_text(_json(receipt) + "\n", encoding="utf-8")
        from ._protocol import dump_recipe
        from ._presentation import render_prepare
        (final / "recipe.yaml").write_text(dump_recipe(receipt["recipe"]), encoding="utf-8")
        (final / "report.txt").write_text(render_prepare(receipt) + "\n", encoding="utf-8")
        verify_report(final, receipt)
        if self.destination.exists():
            raise WrangleError("OUTPUT_EXISTS", "The output destination appeared during writing.", {"path": str(self.destination)})
        # Construct/verify before publication; a failure must never publish a bundle.
        result = Prepared(pl.scan_parquet(final / "data.parquet", glob=False), receipt, _directory=final)
        snapshot = DiskTable(result.data, path=final / "data.parquet")
        expected = receipt.get("output", {})
        if result._data_digest != expected.get("sha256") or snapshot.height != expected.get("rows") or {name: str(dtype) for name, dtype in snapshot.schema.items()} != expected.get("columns"):
            raise WrangleError("BUNDLE_MISMATCH", "Published data must match the checked output snapshot.")
        publish_directory(final, self.destination)
        object.__setattr__(result, "_directory", self.destination)
        object.__setattr__(result, "data", pl.scan_parquet(self.destination / "data.parquet", glob=False))
        return result
