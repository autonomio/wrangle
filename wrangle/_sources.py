"""Local source parsing and verified preparation lineage; no scientific meaning is inferred."""
from __future__ import annotations

import codecs
from collections.abc import Mapping
from glob import escape as _glob_escape
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import polars as pl

from ._core import WrangleError, dtype_spec, frame

__all__ = ["read_source"]

_CSV_OPTIONS = {"separator", "quote_char", "has_header", "encoding", "comment_prefix", "skip_rows", "skip_rows_after_header", "null_values", "eol_char", "decimal_comma", "new_columns"}
_EXCEL_OPTIONS = {"sheet_name", "sheet_id", "has_header"}


def _canonical(value, code="INVALID_BUNDLE"):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise WrangleError(code, "Metadata must contain only finite JSON values.") from error


def _file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _json_text(text, code, path):
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise WrangleError("DUPLICATE_COLUMNS" if code == "INVALID_SOURCE" else code, "JSON field names must be unique.", {"path": str(path), "field": name})
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, UnicodeError) as error:
        if isinstance(error, WrangleError):
            raise
        raise WrangleError(code, "Use strict, finite UTF-8 JSON.", {"path": str(path)}) from error


def _names(names, *, path):
    if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
        raise WrangleError("DUPLICATE_COLUMNS", "Source columns must have unique, nonempty names.", {"path": str(path)})


def _schema(value):
    if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "schema maps nonempty column names to JSON dtype declarations.")
    _canonical(value, "INVALID_SOURCE_OPTIONS")
    return {name: dtype_spec(dtype) for name, dtype in value.items()}


def _csv_options(options, format):
    if set(options) - _CSV_OPTIONS:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "Use documented CSV parsing options.", {"unknown_options": sorted(set(options) - _CSV_OPTIONS)})
    result = {"separator": "\t" if format == "tsv" else ",", "has_header": True, "encoding": "utf8", **options}
    for name in ("has_header", "decimal_comma"):
        if name in result and type(result[name]) is not bool:
            raise WrangleError("INVALID_SOURCE_OPTIONS", f"{name} must be true or false.")
    for name in ("skip_rows", "skip_rows_after_header"):
        if name in result and (type(result[name]) is not int or result[name] < 0):
            raise WrangleError("INVALID_SOURCE_OPTIONS", f"{name} must be a nonnegative integer.")
    for name in ("separator", "eol_char"):
        if name in result and (not isinstance(result[name], str) or len(result[name].encode("utf-8")) != 1):
            raise WrangleError("INVALID_SOURCE_OPTIONS", f"{name} must be one UTF-8 byte.")
    if "quote_char" in result and result["quote_char"] is not None and (not isinstance(result["quote_char"], str) or len(result["quote_char"].encode("utf-8")) != 1):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "quote_char must be one UTF-8 byte or null.")
    if "comment_prefix" in result and result["comment_prefix"] is not None and (not isinstance(result["comment_prefix"], str) or not result["comment_prefix"]):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "comment_prefix must be a nonempty string or null.")
    encoding = result["encoding"]
    if not isinstance(encoding, str) or "lossy" in encoding.lower():
        raise WrangleError("INVALID_SOURCE_OPTIONS", "Declare a strict text encoding; lossy decoding is forbidden.")
    try:
        codecs.lookup(encoding)
    except LookupError as error:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "The declared text encoding is unknown.", {"encoding": encoding}) from error
    nulls = result.get("null_values")
    if nulls is not None and not (isinstance(nulls, str) or isinstance(nulls, list) and all(isinstance(value, str) for value in nulls) or isinstance(nulls, dict) and all(isinstance(name, str) and isinstance(value, str) for name, value in nulls.items())):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "null_values must be a string, string list, or column-to-string mapping.")
    if "new_columns" in result:
        if not isinstance(result["new_columns"], list) or not result["new_columns"]:
            raise WrangleError("INVALID_SOURCE_OPTIONS", "new_columns must declare every output column name.")
        _names(result["new_columns"], path="new_columns")
    return result


def _csv(path, format, options, schema, *, lazy=False):
    arguments = _csv_options(options, format)
    if lazy:
        if codecs.lookup(arguments["encoding"]).name != "utf-8":
            raise WrangleError("DISK_SOURCE_UNSUPPORTED", "Disk execution requires strict UTF-8 CSV or TSV; transcode explicitly or use memory execution.", {"format": format, "encoding": arguments["encoding"]})
        arguments["encoding"] = "utf8"
    header_arguments = {name: value for name, value in arguments.items() if name not in {"has_header", "new_columns", "null_values", "skip_rows_after_header", "decimal_comma"}}
    first = pl.read_csv(path, glob=False, has_header=False, n_rows=1, infer_schema=False, **header_arguments)
    if arguments["has_header"]:
        _names(first.row(0), path=path)
    names = arguments.get("new_columns")
    if names is not None and len(names) != first.width:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "new_columns must name every source column.", {"columns": first.width})
    raw_names = list(first.row(0)) if arguments["has_header"] else first.columns
    output_names = names if names is not None else raw_names
    unknown = set(schema) - set(output_names)
    null_columns = set(arguments.get("null_values", {})) - set(output_names) if isinstance(arguments.get("null_values"), dict) else set()
    if unknown or null_columns:
        raise WrangleError("COLUMN_NOT_FOUND", "Source declarations reference absent columns.", {"columns": sorted(unknown | null_columns)})
    original_names = dict(zip(output_names, raw_names))
    parsing = dict(arguments)
    if not lazy and isinstance(parsing.get("null_values"), dict):
        parsing["null_values"] = {original_names[name]: value for name, value in parsing["null_values"].items()}
    # The eager parser applies overrides before new_columns; the scanner applies
    # the declared output names first. Both paths retain the same public schema.
    overrides = schema if lazy else {original_names[name]: dtype for name, dtype in schema.items()}
    reader = pl.scan_csv if lazy else pl.read_csv
    data = reader(path, glob=False, infer_schema=False, try_parse_dates=False, schema_overrides=overrides or None, **parsing)
    parsed_schema = data.collect_schema() if lazy else data.schema
    if any(parsed_schema[name] != dtype for name, dtype in schema.items()):
        raise WrangleError("SCHEMA_MISMATCH", "CSV parsing did not produce the explicitly declared dtypes.")
    return data, arguments


def _excel(path, options, schema):
    if set(options) - _EXCEL_OPTIONS:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "Excel accepts sheet_name or sheet_id and has_header only.")
    chosen = [name for name in ("sheet_name", "sheet_id") if name in options]
    if len(chosen) != 1:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "Declare exactly one Excel sheet_name or positive sheet_id.")
    if chosen[0] == "sheet_name" and (not isinstance(options["sheet_name"], str) or not options["sheet_name"]):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "sheet_name must be a nonempty string.")
    if chosen[0] == "sheet_id" and (type(options["sheet_id"]) is not int or options["sheet_id"] <= 0):
        raise WrangleError("INVALID_SOURCE_OPTIONS", "sheet_id must select one sheet using a positive integer.")
    if "has_header" in options and type(options["has_header"]) is not bool:
        raise WrangleError("INVALID_SOURCE_OPTIONS", "has_header must be true or false.")
    try:
        # Polars<2 emits this upstream calamine adapter deprecation even though
        # read_excel still returns a DataFrame. Keep CLI evidence free of that
        # exact compatibility warning; parsing errors/data warnings still surface.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"from_arrow\(<ArrowStreamExportable>\) will return a Series instead of a DataFrame in 2\.0\.", category=FutureWarning)
            data = pl.read_excel(path, engine="calamine", drop_empty_rows=False, drop_empty_cols=False, schema_overrides=schema or None, **options)
    except ImportError as error:
        raise WrangleError("DEPENDENCY_REQUIRED", "Excel requires the optional wrangle[excel] dependency.", {"dependency": "fastexcel"}) from error
    if not isinstance(data, pl.DataFrame):
        raise WrangleError("INVALID_SOURCE", "The Excel declaration must select one sheet.")
    unknown = set(schema) - set(data.columns)
    if unknown:
        raise WrangleError("COLUMN_NOT_FOUND", "Source schema references absent Excel columns.", {"columns": sorted(unknown)})
    _names(data.columns, path=path)
    return data


def _parent(data, receipt, digest, recipe=None):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("recipe"), dict):
        raise WrangleError("INVALID_BUNDLE", "A preparation receipt must retain its recipe.")
    if receipt.get("hash_format") != "logical-native-json-v1":
        raise WrangleError("INVALID_BUNDLE", "Use the supported logical-native-json-v1 snapshot format.", {"hash_format": receipt.get("hash_format")})
    if recipe is not None and _canonical(receipt["recipe"]) != _canonical(recipe):
        raise WrangleError("BUNDLE_MISMATCH", "The saved recipe disagrees with the receipt.")
    recipe_hash = hashlib.sha256(_canonical(receipt["recipe"]).encode()).hexdigest()
    if receipt.get("recipe_sha256") != recipe_hash:
        raise WrangleError("BUNDLE_MISMATCH", "The retained recipe checksum does not match.")
    columns = {name: str(dtype) for name, dtype in data.schema.items()}
    output = receipt.get("output")
    if not isinstance(output, dict) or type(output.get("rows")) is not int or output["rows"] != data.height or output.get("columns") != columns or output.get("sha256") != digest(data):
        raise WrangleError("BUNDLE_MISMATCH", "Prepared data disagree with the receipt rows, schema, or snapshot checksum.")
    keys, units, variables = receipt.get("key"), receipt.get("units"), receipt.get("variables")
    if not isinstance(keys, list) or any(not isinstance(name, str) or name not in columns for name in keys) or len(keys) != len(set(keys)):
        raise WrangleError("INVALID_BUNDLE", "Parent keys must reference distinct output columns.")
    if not isinstance(units, dict) or any(name not in columns or not isinstance(unit, str) or not unit for name, unit in units.items()):
        raise WrangleError("INVALID_BUNDLE", "Parent units must reference output columns and nonempty units.")
    if not isinstance(variables, dict) or set(variables) != set(columns):
        raise WrangleError("INVALID_BUNDLE", "Parent variable metadata must describe every output column.")
    descriptions = {}
    for name, dtype in columns.items():
        variable = variables[name]
        if not isinstance(variable, dict) or variable.get("dtype") != dtype or variable.get("unit") != units.get(name):
            raise WrangleError("BUNDLE_MISMATCH", "Parent variable dtypes and units disagree with output metadata.", {"column": name})
        description = variable.get("description")
        if description is not None:
            if not isinstance(description, str) or not description:
                raise WrangleError("INVALID_BUNDLE", "Parent meanings must be nonempty strings.", {"column": name})
            descriptions[name] = description
    return {"recipe_sha256": recipe_hash, "receipt_sha256": hashlib.sha256(_canonical(receipt).encode()).hexdigest(), "key": keys, "units": units, "descriptions": descriptions}


def _copy_source(path, workspace):
    """Copy a stable local byte snapshot before a LazyFrame may read it."""
    before = _file_digest(path)
    descriptor, name = tempfile.mkstemp(prefix="raw-", suffix=path.suffix, dir=workspace.root)
    target = Path(name)
    try:
        os.close(descriptor)
        shutil.copyfile(path, target)
        if _file_digest(path) != before or _file_digest(target) != before:
            raise WrangleError("SOURCE_CHANGED", "The source changed while capturing a disk snapshot.", {"path": str(path)})
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target, before


def _bundle(path, digest, workspace=None):
    recipe_files = [name for name in ("recipe.yaml", "recipe.yml", "recipe.json") if (path / name).is_file()]
    if len(recipe_files) != 1:
        raise WrangleError("INVALID_BUNDLE", "A saved preparation must contain exactly one recipe.yaml (or one historical recipe.json).", {"path": str(path), "recipes": recipe_files})
    recipe_name = recipe_files[0]
    files = {name: path / name for name in ("data.parquet", recipe_name, "receipt.json")}
    if any(not file.is_file() for file in files.values()):
        raise WrangleError("INVALID_BUNDLE", "A saved preparation requires data.parquet, recipe.yaml, and receipt.json.", {"path": str(path)})
    before = {name: _file_digest(file) for name, file in files.items()}
    captured = files
    if workspace is not None:
        captured = {}
        for name, file in files.items():
            captured[name], checksum = _copy_source(file, workspace)
            if checksum != before[name]:
                raise WrangleError("SOURCE_CHANGED", "The saved preparation changed during snapshot capture.", {"path": str(path)})
    receipt = _json_text(captured["receipt.json"].read_text(encoding="utf-8"), "INVALID_BUNDLE", files["receipt.json"])
    if recipe_name == "recipe.json":
        # JSON is accepted only for historical saved evidence, never authored recipes.
        recipe = _json_text(captured[recipe_name].read_text(encoding="utf-8"), "INVALID_BUNDLE", files[recipe_name])
    else:
        from ._protocol import load_recipe
        try:
            recipe = load_recipe(captured[recipe_name])
        except WrangleError as error:
            raise WrangleError("INVALID_BUNDLE", "The saved YAML recipe is invalid.", {**error.details, "path": str(files[recipe_name]), "recipe_code": error.code}) from error
    if not isinstance(recipe, dict) or not isinstance(receipt, dict):
        raise WrangleError("INVALID_BUNDLE", "A saved recipe and receipt must be mappings.")
    from ._storage import verify_evidence
    entries = verify_evidence(path, receipt)
    from ._storage import verify_report
    if verify_report(path, receipt):
        files["report.txt"] = path / "report.txt"
        before["report.txt"] = receipt["report_sha256"]
    for entry in entries:
        files[entry["path"]] = path / entry["path"]
        before[entry["path"]] = entry["sha256"]
    data = workspace.snapshot(pl.scan_parquet(captured["data.parquet"], glob=False)) if workspace is not None else pl.read_parquet(files["data.parquet"], glob=False).rechunk()
    parent = _parent(data, receipt, digest, recipe)
    after = {name: _file_digest(file) for name, file in files.items()}
    if before != after:
        raise WrangleError("SOURCE_CHANGED", "The saved preparation changed during reading.", {"path": str(path)})
    return data, {"path": str(path), "sha256": digest(data), "format": "bundle", "files": before, "parent": parent}


def read_source(source, *, prepared_type, digest, workspace=None):
    """Return a stable native table and JSON source identity, including verified parent semantics."""
    if isinstance(source, prepared_type):
        receipt_text = _canonical(source.receipt, "RESULT_CHANGED")
        if hashlib.sha256(receipt_text.encode()).hexdigest() != source._receipt_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared receipt changed; prepare again before reuse.")
        retained = source._source_table() if hasattr(source, "_source_table") else source.data
        if workspace is not None:
            if getattr(retained, "path", None) is not None:
                captured, checksum = _copy_source(retained.path, workspace)
                if checksum != source._file_digest:
                    raise WrangleError("RESULT_CHANGED", "The prepared data changed during capture; prepare again before reuse.")
                plan = pl.scan_parquet(captured, glob=False)
            else:
                plan = retained.lazy() if callable(retained.lazy) else retained.lazy
            data = workspace.snapshot(plan)
            source._source_table()  # Verify parent metadata/evidence also after disk capture.
        elif isinstance(retained, pl.DataFrame):
            data = retained.rechunk()
        else:
            plan = retained.lazy() if callable(retained.lazy) else retained.lazy
            data = plan.collect().rechunk()
            source._source_table()  # Verify saved bytes/evidence also after scanning.
        if digest(data) != source._data_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared data changed; prepare again before reuse.")
        receipt = json.loads(receipt_text)
        info = {"sha256": digest(data), "format": "prepared", "parent": _parent(data, receipt, digest)}
    else:
        declaration = dict(source) if isinstance(source, Mapping) and "path" in source else None
        if declaration is not None:
            if set(declaration) - {"path", "format", "options", "schema"} or not isinstance(declaration.get("path"), (str, Path)):
                raise WrangleError("INVALID_SOURCE_OPTIONS", "A file source declares path, optional format, options, and schema only.")
            options = declaration.get("options", {})
            if not isinstance(options, dict):
                raise WrangleError("INVALID_SOURCE_OPTIONS", "options must be an object of documented file parsing choices.")
            _canonical(options, "INVALID_SOURCE_OPTIONS")
            schema = _schema(declaration.get("schema", {}))
            path = Path(declaration["path"]).expanduser().resolve()
        elif isinstance(source, (str, Path)):
            path, options, schema = Path(source).expanduser().resolve(), {}, {}
        else:
            try:
                plan = frame(source)
                data = workspace.snapshot(plan) if workspace is not None else plan.collect().rechunk()
            except pl.exceptions.PolarsError as error:
                raise WrangleError("INVALID_SOURCE", "The source plan cannot be evaluated.", {"error": str(error)}) from error
            info = {"sha256": digest(data), "format": "polars"}
            data_hash = digest(data)
            info.update({"rows": data.height, "columns": {name: str(dtype) for name, dtype in data.schema.items()}, "snapshot_sha256": data_hash})
            return data, info
        try:
            if path.is_dir():
                if declaration is not None and (options or schema or declaration.get("format") not in {None, "bundle"}):
                    raise WrangleError("INVALID_SOURCE_OPTIONS", "Verified bundles do not accept parsing overrides.")
                data, info = _bundle(path, digest, workspace=workspace)
            else:
                if not path.is_file():
                    raise WrangleError("SOURCE_NOT_FOUND", "Use an existing local data file or preparation bundle.", {"path": str(path)})
                format = declaration.get("format", path.suffix.lower()[1:]) if declaration else path.suffix.lower()[1:]
                if not isinstance(format, str):
                    raise WrangleError("INVALID_SOURCE_OPTIONS", "format must be a documented file format name.")
                format = format.lower()
                before = _file_digest(path)
                captured = path
                if workspace is not None:
                    if format in {"ndjson", "jsonl", "excel", "xlsx", "xls", "xlsb"}:
                        raise WrangleError("DISK_SOURCE_UNSUPPORTED", "Disk execution supports strict UTF-8 CSV/TSV, Parquet, and IPC sources; use memory execution for this format.", {"path": str(path), "format": format})
                    captured, checksum = _copy_source(path, workspace)
                    if checksum != before:
                        raise WrangleError("SOURCE_CHANGED", "The source changed during snapshot capture.", {"path": str(path)})
                if format in {"csv", "tsv"}:
                    data, resolved = _csv(captured, format, options, schema, lazy=True) if workspace is not None else _csv(path, format, options, schema)
                elif format in {"excel", "xlsx", "xls", "xlsb"}:
                    data, resolved = _excel(path, options, schema), options
                else:
                    if options:
                        raise WrangleError("INVALID_SOURCE_OPTIONS", "This native file format does not accept parsing options.")
                    if format == "parquet":
                        data = pl.scan_parquet(captured, glob=False) if workspace is not None else pl.read_parquet(path, glob=False)
                    elif format in {"ipc", "arrow"}:
                        # The minimum supported IPC scanner has no glob flag.
                        # Escape every path component so only the captured file
                        # can match, including literal wildcard directory names.
                        data = pl.scan_ipc(_glob_escape(str(captured))) if workspace is not None else pl.read_ipc(path)
                    elif format in {"ndjson", "jsonl"}:
                        with path.open(encoding="utf-8") as stream:
                            rows = [_json_text(line, "INVALID_SOURCE", path) for line in stream if line.strip()]
                        if any(not isinstance(row, dict) for row in rows):
                            raise WrangleError("INVALID_SOURCE", "Every NDJSON record must be an object.", {"path": str(path)})
                        data = frame(rows).collect()
                    else:
                        raise WrangleError("UNSUPPORTED_FORMAT", "Use CSV, TSV, Parquet, Arrow IPC, NDJSON, or an explicitly selected Excel sheet.", {"path": str(path), "format": format})
                    parsed_schema = data.collect_schema() if isinstance(data, pl.LazyFrame) else data.schema
                    if any(name not in parsed_schema or parsed_schema[name] != dtype for name, dtype in schema.items()):
                        raise WrangleError("SCHEMA_MISMATCH", "Native source dtypes disagree with the declared schema.")
                    resolved = options
                if workspace is not None:
                    data = workspace.snapshot(data)
                if _file_digest(path) != before:
                    raise WrangleError("SOURCE_CHANGED", "The source changed during reading; use a stable snapshot.", {"path": str(path)})
                info = {"path": str(path), "sha256": before, "format": format}
                if declaration is not None:
                    info.update({"options": resolved, "schema": declaration.get("schema", {})})
        except (pl.exceptions.PolarsError, UnicodeError, OSError, TypeError, ValueError) as error:
            if isinstance(error, WrangleError):
                raise
            raise WrangleError("INVALID_SOURCE", "The source cannot be parsed without errors.", {"path": str(path), "error": str(error)}) from error
    if workspace is None:
        data = data.rechunk()
    info.update({"rows": data.height, "columns": {name: str(dtype) for name, dtype in data.schema.items()}, "snapshot_sha256": digest(data)})
    return data, info
