"""The two-call preparation interface: inspect data, then execute a recipe."""
from __future__ import annotations
from ._core import operation

import hashlib
import inspect as signatures
import json
import math
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from ._core import WrangleError, dtype_spec, floating_count, require_columns
from ._publication import publish_directory
from ._storage import DiskTable, DiskWorkspace, collect, execution_context, file_digest, verify_evidence

VERSION = "1.0.0"
_RECIPE_KEYS = {"version", "name", "input", "key", "units", "steps", "checks", "descriptions", "source_options", "source_contracts", "pending_decisions", "research_decisions"}


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise WrangleError("INVALID_RECIPE", "Recipes must contain only finite JSON values.") from error


def _canonical_value(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Represent logical values without bytes hidden underneath null validity."""
    valid = expression.is_not_null().alias("valid")
    if isinstance(dtype, pl.Struct):
        fields = []
        for field in dtype.fields:
            child = pl.when(expression.is_null()).then(pl.lit(None, dtype=field.dtype)).otherwise(expression.struct.field(field.name))
            fields.append(_canonical_value(child, field.dtype).alias(field.name))
        value = pl.struct(fields) if fields else pl.lit(0, dtype=pl.UInt8)
    elif isinstance(dtype, (pl.List, pl.Array)):
        values = expression.arr.to_list() if isinstance(dtype, pl.Array) else expression
        values = values.fill_null(pl.lit([], dtype=pl.List(dtype.inner)))
        value = values.list.eval(_canonical_value(pl.element(), dtype.inner))
    elif dtype.is_float():
        value = expression.cast(pl.String).fill_null("")
        return pl.struct(valid, value.alias("value"))
    elif dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        value = expression.cast(pl.String).fill_null("")
    elif dtype == pl.Binary:
        value = expression.bin.encode("hex").fill_null("")
    elif dtype == pl.Boolean:
        value = expression.fill_null(False)
    elif dtype.is_numeric() or dtype.is_temporal():
        value = expression.fill_null(pl.lit(0).cast(dtype))
    elif dtype == pl.Null:
        value = pl.lit(0, dtype=pl.UInt8)
    else:
        raise WrangleError("UNSUPPORTED_DTYPE", "Use serializable native research dtypes.", {"dtype": str(dtype)})
    return pl.struct(valid, value.alias("value"))


def _digest(data: pl.DataFrame | DiskTable) -> str:
    if isinstance(data, DiskTable):
        return data.logical_digest
    descriptor = _json({"rows": data.height, "columns": [(name, str(dtype)) for name, dtype in data.schema.items()]}).encode()
    try:
        canonical = collect(data.lazy().select([_canonical_value(pl.col(name), dtype).alias(name) for name, dtype in data.schema.items()])).rechunk()
        payload = canonical.write_json().encode("utf-8")
    except (pl.exceptions.PolarsError, TypeError) as error:
        raise WrangleError("UNSUPPORTED_DTYPE", "Research data must use serializable native Polars dtypes.") from error
    checksum = hashlib.sha256()
    checksum.update(len(descriptor).to_bytes(8, "big"))
    checksum.update(descriptor)
    checksum.update(payload)
    return checksum.hexdigest()


def _source(source: Any, *, workspace=None):
    from ._sources import read_source
    return read_source(source, prepared_type=Prepared, digest=_digest, workspace=workspace)


class _Inspection(dict):
    """An ordinary observation mapping with readable notebook/terminal views."""

    def summary(self) -> str:
        from ._presentation import render_inspect
        return render_inspect(self)

    def _repr_html_(self) -> str:
        from html import escape
        return "<pre>" + escape(self.summary()) + "</pre>"


def inspect(source: Any, *, sample_rows: int = 5, columns: list[str] | None = None, summary: bool = False, groups: list[str] | None = None, max_groups: int = 20, baseline: Any = None) -> dict[str, Any]:
    """Observe a source, with optional bounded summaries and schema comparison.

    Numeric summaries describe non-null observations with ddof=1. Nonfinite or
    inexact floating conversions remain unresolved. No units, meaning, missing
    codes or schema corrections are inferred. CSV identifiers remain strings.
    """
    from ._profile import profile, validate_options
    validate_options(sample_rows=sample_rows, columns=columns, summary=summary, groups=groups, max_groups=max_groups)
    data, info = _source(source)
    baseline_columns = None
    if baseline is not None:
        reference, _ = _source(baseline)
        baseline_columns = {name: str(dtype) for name, dtype in reference.schema.items()}
    return _Inspection(profile(data, info, sample_rows=sample_rows, columns=columns, summary=summary, groups=groups, max_groups=max_groups, baseline_columns=baseline_columns))



def _column_list(value, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty) or any(not isinstance(name, str) or not name for name in value) or len(set(value)) != len(value):
        raise WrangleError("INVALID_RECIPE", "Columns must be a list of distinct nonempty names.")


def _column_mapping(value, *, values_are_strings=False):
    if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
        raise WrangleError("INVALID_RECIPE", "Columns must be an object mapping names to declared values.")
    if values_are_strings and any(not isinstance(item, str) or not item for item in value.values()):
        raise WrangleError("INVALID_RECIPE", "Column declarations must contain nonempty strings.")


def _validate_recipe_decisions(recipe):
    """Reject unanswered guided decisions before reading or transforming sources."""
    pending = recipe.get("pending_decisions", [])
    if not isinstance(pending, list):
        raise WrangleError("INVALID_RECIPE", "pending_decisions must be a list of unanswered questions.")
    identities = set()
    for question in pending:
        if not isinstance(question, dict) or any(not isinstance(question.get(name), str) or not question[name].strip() for name in ("id", "question")):
            raise WrangleError("INVALID_RECIPE", "Each pending decision requires a nonempty id and question.")
        if question["id"] in identities:
            raise WrangleError("INVALID_RECIPE", "Pending decision identifiers must be distinct.", {"id": question["id"]})
        identities.add(question["id"])
    if pending:
        raise WrangleError("UNRESOLVED_PROTOCOL", "Answer the required research questions before preparing data; rerun start with the completed answers.", {"decisions": pending})
    if "research_decisions" in recipe:
        from ._start import validate_decisions
        validate_decisions(recipe)


def _validate_rules(rules):
    from ._contracts import validate_rules
    validate_rules(rules)


def _finite_output(data):
    expressions = [(floating_count(pl.col(name), dtype, kind="nan") + floating_count(pl.col(name), dtype, kind="infinite")).sum().alias(name) for name, dtype in data.schema.items()]
    counts = collect(data.lazy().select(expressions))
    bad = {name: count for name, count in (counts.row(0, named=True) if counts.width else {}).items() if count}
    if bad:
        raise WrangleError("NONFINITE_RESULT", "Resolve floating NaN and infinity explicitly, including nested fields; nulls remain subject to required-value checks.", {"columns": bad})


def _dtype(value) -> pl.DataType:
    return dtype_spec(value)


def _expression_dtype(data: pl.LazyFrame, expression: pl.Expr) -> pl.DataType:
    return data.select(expression).collect_schema().dtypes()[0]


def _checked_integer(data: pl.LazyFrame, expression: pl.Expr) -> pl.Expr:
    narrowed = expression.cast(pl.Int64, strict=False)
    count = collect(data.select((expression.is_not_null() & narrowed.is_null()).sum())).item()
    if count:
        raise WrangleError("INTEGER_OVERFLOW", "Integer arithmetic must fit signed 64-bit values; no wrapped result is permitted.", {"affected_rows": count})
    return expression.cast(pl.Int64, strict=True)


def _float_operand(data: pl.LazyFrame, expression: pl.Expr) -> pl.Expr:
    dtype = _expression_dtype(data, expression)
    converted = expression.cast(pl.Float64, strict=True)
    if dtype.is_integer() or isinstance(dtype, pl.Decimal):
        restored = converted.cast(dtype, strict=False)
        count = collect(data.select((~expression.eq_missing(restored)).sum())).item()
        if count:
            raise WrangleError("LOSSY_CAST", "Floating arithmetic would lose integer precision; resolve the measurement dtype explicitly.", {"affected_rows": count})
    return converted


def _expression(value: Any, data: pl.LazyFrame) -> pl.Expr:
    if not isinstance(value, dict):
        if isinstance(value, (str, int, float, bool)) or value is None:
            try:
                return pl.lit(value)
            except (OverflowError, TypeError, pl.exceptions.PolarsError) as error:
                raise WrangleError("INVALID_EXPRESSION", "Use a scalar supported by native Polars dtypes.") from error
        raise WrangleError("INVALID_EXPRESSION", "Expression leaves must be a column reference or JSON scalar.")
    if len(value) != 1:
        raise WrangleError("INVALID_EXPRESSION", "Each expression node must have exactly one operator.")
    op, arguments = next(iter(value.items()))
    if op == "col":
        if not isinstance(arguments, str):
            raise WrangleError("INVALID_EXPRESSION", "col requires a column name.")
        require_columns(data, arguments)
        expression = pl.col(arguments)
        if data.collect_schema()[arguments].is_float():
            if collect(data.select(expression.is_infinite().sum())).item():
                raise WrangleError("NONFINITE_RESULT", "Resolve infinite source measurements before deriving or filtering them.", {"column": arguments})
            return expression.fill_nan(None)
        return expression
    if op == "lit":
        if isinstance(arguments, (dict, list)):
            raise WrangleError("INVALID_EXPRESSION", "lit requires a JSON scalar.")
        return _expression(arguments, data)
    unary = {"abs": lambda x: x.abs(), "sqrt": lambda x: x.sqrt(), "log1p": lambda x: x.log1p(), "not": lambda x: ~x, "is_null": lambda x: x.is_null(), "is_not_null": lambda x: x.is_not_null()}
    if op in unary:
        expression = _expression(arguments, data)
        dtype = _expression_dtype(data, expression)
        if op == "not" and dtype != pl.Boolean and dtype != pl.Null:
            raise WrangleError("INVALID_EXPRESSION", "not requires a Boolean expression.")
        if op in {"abs", "sqrt", "log1p"} and not dtype.is_numeric() and dtype != pl.Null:
            raise WrangleError("INVALID_EXPRESSION", "Arithmetic requires numeric expressions.")
        if op == "abs" and dtype.is_integer():
            return _checked_integer(data, _checked_integer(data, expression).cast(pl.Int128).abs())
        if op in {"sqrt", "log1p"} and dtype.is_numeric():
            expression = _float_operand(data, expression)
        return unary[op](expression)
    binary = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        "pow": lambda a, b: a.pow(b), "eq": lambda a, b: a == b,
        "ne": lambda a, b: a != b, "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b, "and": lambda a, b: a & b,
        "or": lambda a, b: a | b,
    }
    if op in binary and isinstance(arguments, list) and len(arguments) == 2:
        left, right = [_expression(arg, data) for arg in arguments]
        types = [_expression_dtype(data, operand) for operand in (left, right)]
        if op in {"and", "or"} and any(dtype not in {pl.Boolean, pl.Null} for dtype in types):
            raise WrangleError("INVALID_EXPRESSION", "and/or require Boolean expressions.")
        if op in {"add", "sub", "mul", "div", "pow"} and any(not dtype.is_numeric() and dtype != pl.Null for dtype in types):
            raise WrangleError("INVALID_EXPRESSION", "Arithmetic requires numeric expressions.")
        integer = all(dtype.is_integer() or dtype == pl.Null for dtype in types) and any(dtype.is_integer() for dtype in types)
        if op in {"add", "sub", "mul", "pow"} and integer:
            left, right = [_checked_integer(data, operand).cast(pl.Int128) for operand in (left, right)]
            if op == "pow":
                invalid = (right < 0) | (right > 63)
                if collect(data.select(invalid.fill_null(False).sum())).item():
                    raise WrangleError("INVALID_DOMAIN", "Integer powers require exponents from 0 to 63; declare floating arithmetic for other powers.")
                # Reject huge powers before native integer evaluation can wrap.
                too_large = left.abs().cast(pl.Float64).pow(right.cast(pl.Float64)) > float(2**63)
                if collect(data.select(too_large.fill_null(False).sum())).item():
                    raise WrangleError("INTEGER_OVERFLOW", "The integer power exceeds signed 64-bit arithmetic.")
            return _checked_integer(data, binary[op](left, right))
        numeric = all(dtype.is_numeric() or dtype == pl.Null for dtype in types)
        if numeric and (op in {"add", "sub", "mul", "div", "pow"} or any(dtype.is_float() for dtype in types)):
            left, right = [_float_operand(data, operand) for operand in (left, right)]
        elif numeric and integer and op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            left, right = [operand.cast(pl.Int128) for operand in (left, right)]
        return binary[op](left, right)
    from ._recipe_expressions import compile_expression
    extended = compile_expression(value, data, _expression)
    if extended is not None:
        return extended
    raise WrangleError("INVALID_EXPRESSION", "Use a documented expression operator and arity.", {"operator": op})


@operation(returns=('table',), recipe='yes')
def _cast(data: pl.LazyFrame, columns: dict[str, Any]) -> pl.LazyFrame:
    """Cast declared columns strictly; fractional numbers cannot become integers."""
    _column_mapping(columns)
    require_columns(data, list(columns))
    schema = data.collect_schema()
    expressions = []
    for name, dtype in columns.items():
        target = _dtype(dtype)
        if target.is_integer() and schema[name].is_float():
            count = collect(data.select(((pl.col(name) % 1) != 0).fill_null(False).sum())).item()
            if count:
                raise WrangleError("LOSSY_CAST", "Fractional values cannot be cast to integers.", {"column": name, "affected_rows": count})
        if schema[name].is_numeric() and target.is_numeric() and schema[name] != target:
            roundtrip = pl.col(name).cast(target, strict=True).cast(schema[name], strict=True)
            count = collect(data.select((~pl.col(name).eq_missing(roundtrip)).sum())).item()
            if count:
                raise WrangleError("LOSSY_CAST", "The cast would lose numeric precision.", {"column": name, "affected_rows": count})
        expressions.append(pl.col(name).cast(target, strict=True))
    return data.with_columns(expressions)


@operation(returns=('table',), recipe='yes')
def _rename(data: pl.LazyFrame, columns: dict[str, str]) -> pl.LazyFrame:
    """Rename columns explicitly, rejecting collisions."""
    _column_mapping(columns, values_are_strings=True)
    require_columns(data, list(columns))
    names = [columns.get(name, name) for name in data.collect_schema().names()]
    if len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise WrangleError("DUPLICATE_COLUMNS", "Renaming must produce distinct, nonempty names.")
    return data.rename(columns)


@operation(returns=('table',), recipe='yes')
def _select(data: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Select named columns in the declared order."""
    _column_list(columns)
    require_columns(data, columns)
    return data.select(columns)


@operation(returns=('table',), recipe='yes')
def _derive(data: pl.LazyFrame, columns: dict[str, Any], overwrite: bool = False) -> pl.LazyFrame:
    """Derive named columns from finite, declarative expression trees."""
    _column_mapping(columns)
    if not isinstance(overwrite, bool):
        raise WrangleError("INVALID_RECIPE", "overwrite must be true or false.")
    collisions = set(columns) & set(data.collect_schema().names())
    if collisions and not overwrite:
        raise WrangleError("COLUMN_EXISTS", "Set overwrite=true to deliberately replace a column.", {"columns": sorted(collisions)})
    return data.with_columns([_expression(expression, data).alias(name) for name, expression in columns.items()])


@operation(returns=('table',), recipe='yes')
def _filter(data: pl.LazyFrame, where: Any, nulls: str = "error") -> pl.LazyFrame:
    """Apply an explicit predicate; missing truth values require a declared policy."""
    if nulls not in {"error", "keep", "drop"}:
        raise WrangleError("INVALID_ARGUMENT", "Filter nulls must be error, keep, or drop.")
    predicate = _expression(where, data)
    if data.select(predicate).collect_schema().dtypes() != [pl.Boolean]:
        raise WrangleError("INVALID_EXPRESSION", "A filter must evaluate to booleans.")
    unknown = collect(data.select(predicate.is_null().sum())).item()
    if unknown and nulls == "error":
        raise WrangleError("UNRESOLVED_FILTER", "The filter is unknown for missing values; declare nulls=keep or drop.", {"affected_rows": unknown})
    return data.filter(predicate.fill_null(nulls == "keep"))


@operation(returns=('table',), recipe='yes')
def _convert_unit(data: pl.LazyFrame, column: str, from_unit: str, to_unit: str, factor: float, offset: float = 0) -> pl.LazyFrame:
    """Apply a declared linear unit conversion, recording both unit names."""
    require_columns(data, column)
    if not isinstance(from_unit, str) or not from_unit or not isinstance(to_unit, str) or not to_unit or not isinstance(factor, (int, float)) or isinstance(factor, bool) or not math.isfinite(factor) or factor == 0 or not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
        raise WrangleError("INVALID_ARGUMENT", "Declare nonempty units, a finite nonzero factor, and finite offset.")
    if not data.collect_schema()[column].is_numeric():
        raise WrangleError("INVALID_DTYPE", "Cast the measurement to a numeric dtype before converting its unit.", {"column": column})
    expression = {"add": [{"mul": [{"col": column}, factor]}, offset]}
    return data.with_columns(_expression(expression, data).alias(column))


from ._recipe_fields import OPERATIONS as _FIELD_OPERATIONS
from ._recipe_tables import OPERATIONS as _TABLE_OPERATIONS
from ._recipe_statistics import OPERATIONS as _STATISTIC_OPERATIONS
from ._recipe_join import OPERATIONS as _JOIN_OPERATIONS, join as _join

_SIMPLE = {"cast": _cast, "rename": _rename, "select": _select, "derive": _derive, "filter": _filter, "convert_unit": _convert_unit, **_FIELD_OPERATIONS, **_TABLE_OPERATIONS, **_STATISTIC_OPERATIONS, **_JOIN_OPERATIONS}


def _checks(data: pl.DataFrame, rules: dict[str, Any], key: list[str], *, sources=None, units=None, descriptions=None) -> list[dict[str, Any]]:
    from ._contracts import check_data
    try:
        return check_data(data, rules, key, sources=sources, units=units, descriptions=descriptions, expression=_expression)
    except WrangleError:
        raise
    except (pl.exceptions.PolarsError, TypeError, ValueError, KeyError) as error:
        raise WrangleError("INVALID_RECIPE", "The declared research check cannot be evaluated; correct its field types, bounds or expression.", {"error": str(error)}) from error


@dataclass(frozen=True)
class Prepared:
    """A verified Polars table and its JSON-compatible execution receipt."""

    data: pl.DataFrame | pl.LazyFrame
    receipt: dict[str, Any]
    _directory: Path | None = field(default=None, repr=False)
    _file_digest: str | None = field(init=False, default=None, repr=False)
    _receipt_digest: str = field(init=False, repr=False)
    _data_digest: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_receipt_digest", hashlib.sha256(_json(self.receipt).encode()).hexdigest())
        if self._directory is not None:
            table = DiskTable(self.data, path=self._directory / "data.parquet")
            object.__setattr__(self, "_data_digest", _digest(table))
            object.__setattr__(self, "_file_digest", file_digest(self._directory / "data.parquet"))
        else:
            object.__setattr__(self, "_data_digest", _digest(self.data))

    def _source_table(self):
        """Verify captured disk bytes/metadata before exposing a source snapshot."""
        from ._sources import _canonical
        if hashlib.sha256(_canonical(self.receipt, "RESULT_CHANGED").encode()).hexdigest() != self._receipt_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared receipt changed; prepare again before reuse.")
        if self._directory is None:
            return self.data.clone()
        if not (self._directory / "data.parquet").is_file() or file_digest(self._directory / "data.parquet") != self._file_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared data changed; prepare again before reuse.")
        # Saved metadata must describe the same captured protocol and evidence.
        from ._protocol import load_recipe
        from ._sources import _json_text
        try:
            saved_receipt = _json_text((self._directory / "receipt.json").read_text(encoding="utf-8"), "RESULT_CHANGED", self._directory / "receipt.json")
            saved_recipe = load_recipe(self._directory / "recipe.yaml")
            if _json(saved_receipt) != _json(self.receipt) or _json(saved_recipe) != _json(self.receipt["recipe"]):
                raise WrangleError("RESULT_CHANGED", "The saved protocol or receipt changed; prepare again before reuse.")
        except WrangleError as error:
            if error.code == "RESULT_CHANGED":
                raise
            raise WrangleError("RESULT_CHANGED", "The saved protocol or receipt changed; prepare again before reuse.", error.details) from error
        except (OSError, UnicodeError) as error:
            raise WrangleError("RESULT_CHANGED", "The saved protocol or receipt is missing or unreadable; restore the verified bundle.") from error
        verify_evidence(self._directory, self.receipt)
        from ._storage import verify_report
        verify_report(self._directory, self.receipt)
        return DiskTable(pl.scan_parquet(self._directory / "data.parquet", glob=False), path=self._directory / "data.parquet")

    def summary(self) -> str:
        """Review the checked preparation in plain language, without reading JSON."""
        from ._presentation import render_prepare
        table = self._source_table()
        if self._directory is None and _digest(table) != self._data_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared data changed; prepare again before reviewing its evidence.")
        return render_prepare(self.receipt)

    def _repr_html_(self) -> str:
        from html import escape
        return "<pre>" + escape(self.summary()) + "</pre>"

    def write(self, directory: str | Path) -> Path:
        """Publish the table, YAML protocol, readable report and receipt atomically."""
        if self._directory is not None:
            table = self._source_table()
            receipt = json.loads(_json(self.receipt))
            with execution_context("disk"), DiskWorkspace(directory) as workspace:
                workspace.publish(table, receipt, evidence_source=self._directory)
            return Path(directory).expanduser().resolve()
        data = self.data.clone()
        receipt = json.loads(_json(self.receipt))
        if _digest(data) != self._data_digest or hashlib.sha256(_json(receipt).encode()).hexdigest() != self._receipt_digest:
            raise WrangleError("RESULT_CHANGED", "The prepared data or receipt changed; prepare again before publishing.")
        destination = Path(directory).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock = destination.parent / (destination.name + ".wrangle-lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise WrangleError("OUTPUT_BUSY", "Another preparation is publishing this destination.", {"path": str(destination)}) from error
        os.close(descriptor)
        temporary = None
        try:
            if destination.exists():
                raise WrangleError("OUTPUT_EXISTS", "Choose a new output directory; existing results are never overwritten.", {"path": str(destination)})
            temporary = Path(tempfile.mkdtemp(prefix=".wrangle-", dir=destination.parent))
            data.write_parquet(temporary / "data.parquet")
            (temporary / "receipt.json").write_text(_json(receipt) + "\n", encoding="utf-8")
            from ._protocol import dump_recipe
            from ._presentation import render_prepare
            (temporary / "recipe.yaml").write_text(dump_recipe(receipt["recipe"]), encoding="utf-8")
            (temporary / "report.txt").write_bytes((render_prepare(receipt) + "\n").encode("utf-8"))
            from ._storage import verify_report
            verify_report(temporary, receipt)
            if destination.exists():
                raise WrangleError("OUTPUT_EXISTS", "The output destination appeared during writing.", {"path": str(destination)})
            publish_directory(temporary, destination)
            temporary = None
            return destination
        finally:
            if temporary is not None:
                shutil.rmtree(temporary)
            lock.unlink(missing_ok=True)


def _column_changed(before: pl.DataFrame, after: pl.DataFrame, name: str, key: list[str]) -> bool:
    if before.schema[name] != after.schema[name]:
        return True
    if not key:
        return not before.select(name).equals(after.select(name))
    old, new = "__wrangle_before", "__wrangle_after"
    while old in key:
        old += "_"
    while new in key:
        new += "_"
    compared = before.lazy().select(*key, pl.col(name).alias(old)).join(
        after.lazy().select(*key, pl.col(name).alias(new)), on=key, how="inner", validate="1:1", maintain_order="left",
    )
    return bool(collect(compared.select((~pl.col(old).eq_missing(pl.col(new))).any())).item())


def prepare(sources: Any, recipe: Mapping[str, Any] | str | Path, *, output: str | Path | None = None, execution: str = "memory") -> Prepared:
    """Prepare with the same research contracts in memory or on disk.

    execution="disk" requires output and returns a LazyFrame in result.data.
    Inputs/intermediates are disk snapshots; checks and evidence remain complete.
    Stateful native operations can still require substantial working memory.
    """
    if not isinstance(execution, str) or execution not in {"memory", "disk"}:
        raise WrangleError("INVALID_EXECUTION", "execution must be memory or disk.")
    if execution == "disk":
        if output is None:
            raise WrangleError("OUTPUT_REQUIRED", "Disk preparation requires a new output directory.")
        with execution_context("disk"), DiskWorkspace(output) as workspace:
            return _prepare(sources, recipe, _workspace=workspace)
    with execution_context("memory"):
        return _prepare(sources, recipe, output=output)


def _prepare(sources: Any, recipe: Mapping[str, Any] | str | Path, *, output: str | Path | None = None, _workspace=None) -> Prepared:
    """Execute declared scientific preparation with one engine for Python and CLI.

    Named sources may be paths, native tables, parsing declarations or verified
    Prepared results/bundles. step.key declares a checked output observation
    unit; otherwise existing identity is preserved. Existing outputs never change.
    """
    from ._execution import STEP_METADATA, GRAIN_OPERATIONS, key_columns, source_contract, compatible_sources, semantic_effects, observation_transition, observation_effects, validate_unit_references, validate_unit_basis
    from ._catalog import resolve, operation_contract
    from ._protocol import load_recipe, normalize_recipe
    if isinstance(recipe, (str, Path)):
        recipe = load_recipe(recipe)
    recipe = normalize_recipe(recipe)
    recipe = json.loads(_json(recipe))
    unknown = set(recipe) - _RECIPE_KEYS
    if unknown or type(recipe.get("version", 1)) is not int or recipe.get("version", 1) != 1:
        raise WrangleError("INVALID_RECIPE", "Use recipe version 1 and documented fields.", {"unknown_fields": sorted(unknown)})
    _validate_recipe_decisions(recipe)
    if "name" in recipe and (not isinstance(recipe["name"], str) or not recipe["name"].strip()):
        raise WrangleError("INVALID_RECIPE", "name must be a nonempty protocol name.")
    steps, rules = recipe.get("steps", []), recipe.get("checks", {})
    if not isinstance(steps, list):
        raise WrangleError("INVALID_RECIPE", "steps must be a list.")
    _validate_rules(rules)
    declaration = isinstance(sources, Mapping) and "path" in sources
    source_map = dict(sources) if isinstance(sources, Mapping) and not declaration else {"data": sources}
    if not source_map or any(not isinstance(name, str) or not name for name in source_map):
        raise WrangleError("INVALID_INPUT", "Sources must have distinct nonempty names.")
    options, contracts = recipe.get("source_options", {}), recipe.get("source_contracts", {})
    for mapping in (options, contracts):
        if not isinstance(mapping, dict) or set(mapping) - set(source_map):
            raise WrangleError("INVALID_RECIPE", "Source declarations must name existing sources.")
    source_data, source_info, semantics = {}, {}, {}
    for name, source in source_map.items():
        if name in options:
            configuration = options[name]
            if not isinstance(configuration, dict) or set(configuration) - {"format", "options", "schema"}:
                raise WrangleError("INVALID_RECIPE", "source_options declares format, options and schema.")
            if isinstance(source, (str, Path)):
                source = {"path": source, **configuration}
            elif isinstance(source, Mapping) and "path" in source:
                if set(source) & set(configuration):
                    raise WrangleError("INVALID_RECIPE", "Declare each source parsing option once.")
                source = {**source, **configuration}
            else:
                raise WrangleError("INVALID_RECIPE", "Parsing options apply only to declared file sources.")
        source_data[name], source_info[name] = _source(source, workspace=_workspace)
        semantics[name] = source_contract(source_data[name], contracts.get(name, {}), source_info[name].get("parent", {}))
        if contracts.get(name):
            source_info[name]["contract"] = semantics[name]
        _checks(source_data[name], {}, semantics[name]["key"])
    input_name = recipe.get("input", next(iter(source_map)) if len(source_map) == 1 else None)
    if input_name not in source_data:
        raise WrangleError("UNKNOWN_SOURCE", "Set recipe.input to the observation table's source name.", {"sources": list(source_data)})
    data = source_data[input_name]
    inherited = semantics[input_name]
    declared_initial = {name: recipe[name] for name in ("units", "descriptions") if name in recipe}
    initial = source_contract(data, declared_initial, inherited)
    key = key_columns(recipe.get("key", initial["key"]))
    if initial["key"] and "key" in recipe and key != initial["key"]:
        raise WrangleError("SOURCE_CONTRACT_MISMATCH", "Keep the verified input key and declare a step transition.")
    units, descriptions = initial["units"], initial["descriptions"]
    _checks(data, {}, key)
    validate_unit_references(data, units)
    receipts, variable_units = [], []
    preserve = {"fill", "impute", "standardize", "cast", "rename", "select", "filter", "join", "join_asof", "convert_unit", "df_impute_nan", "col_fill_nan", "df_fill_empty", "normalize_missing", "sort", "deduplicate", "concat", "pivot", "unpivot", "explode", "unnest", "aggregate", "window", "sample", "partition"}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("op"), str):
            raise WrangleError("INVALID_RECIPE", "Each step requires an operation name.", {"step": index})
        op = step["op"]
        if "reason" in step and (not isinstance(step["reason"], str) or not step["reason"].strip()):
            raise WrangleError("INVALID_RECIPE", "Exclusion reasons must be nonempty strings.", {"step": index})
        if "allow_expand" in step and type(step["allow_expand"]) is not bool:
            raise WrangleError("INVALID_RECIPE", "allow_expand must be true or false.", {"step": index})
        if "unit_column" in step and op != "unpivot":
            raise WrangleError("INVALID_RECIPE", "unit_column applies to unpivot only.", {"step": index})
        declared_units, declared_descriptions = step.get("units", {}), step.get("descriptions", {})
        _column_mapping(declared_units, values_are_strings=True)
        _column_mapping(declared_descriptions, values_are_strings=True)
        parameters = {name: value for name, value in step.items() if name not in STEP_METADATA}
        before, current_key, before_units = data, list(key), dict(units)
        source_names, right_semantics = [], None
        try:
            function = _join if op == "join" else _SIMPLE[op] if op in _SIMPLE else resolve(op)
            contract = operation_contract(function)
            if _workspace is not None and op not in {*_SIMPLE, "join"}:
                raise WrangleError("DISK_OPERATION_UNSUPPORTED", "Disk execution requires a canonical recipe operation; choose its documented equivalent.", {"operation": op})
            if contract["retired"]:
                raise WrangleError("MODEL_ENGINE_REQUIRED", "Model building and fitting are outside preparation scope.", {"operation": op})
            if contract["recipe"] == "never":
                raise WrangleError("NOT_A_TABLE", "This is a direct Python capability; choose a recipe table operation.", {"operation": op})
            if (contract["aggregates"] and key or op in GRAIN_OPERATIONS or op == "sample" and step.get("replacement")) and "key" not in step:
                raise WrangleError("OBSERVATION_UNIT_CHANGED", "Declare step.key for the output observation unit.", {"operation": op})
            if op in {"join", "join_asof"}:
                source_name = parameters.get("source")
                if not isinstance(source_name, str) or source_name not in source_data:
                    raise WrangleError("UNKNOWN_SOURCE", "The join source is absent.", {"source": source_name})
                source_names = [source_name]
                right_semantics = {**semantics[source_name], "columns": source_data[source_name].columns, "data": source_data[source_name]}
                parameters["source"] = source_data[source_name].lazy()
            if op == "concat":
                names = parameters.get("sources")
                if not isinstance(names, list) or any(not isinstance(name, str) or name not in source_data for name in names) or len(names) != len(set(names)) or input_name in names:
                    raise WrangleError("UNKNOWN_SOURCE", "concat.sources must name distinct additional sources.")
                source_names = list(names)
                units, descriptions = compatible_sources(op, before, units, descriptions, source_names, semantics, source_data)
                parameters["sources"] = [source_data[name].lazy() for name in source_names]
            if op == "join":
                from ._recipe_join import resolved_parameters
                target = parameters.pop("source")
                parameters = {**resolved_parameters(parameters), "source": target}
            if op in {"impute", "standardize", "df_impute_nan", "df_rescale_meanzero"}:
                from ._recipe_statistics import resolve_parameters
                parameters = resolve_parameters(op, before.lazy(), parameters, units=units)
            bound = signatures.signature(function).bind(before.lazy(), **parameters)
            bound.apply_defaults()
            result = function(*bound.args, **bound.kwargs)
            if not isinstance(result, (pl.DataFrame, pl.LazyFrame)):
                raise WrangleError("NOT_A_TABLE", "Recipe steps must return a native table.", {"operation": op})
            data = _workspace.snapshot(result) if _workspace is not None else (result.collect() if isinstance(result, pl.LazyFrame) else result).rechunk()
            if op == "rename":
                key = [step["columns"].get(name, name) for name in key]
            if "key" in step:
                key = key_columns(step["key"], empty=False)
            effect_step = {**step, **{name: value for name, value in bound.arguments.items() if name not in {"data", "source", "sources"}}}
            data, units, descriptions, variable = semantic_effects(before, data, op, effect_step, units, descriptions, right_semantics)
            if _workspace is not None and data.path is None:
                data = _workspace.snapshot(data.lazy())
            if variable:
                variable_units.append({"step": index, **variable})
            _finite_output(data)
            require_columns(data, list(declared_units))
            transformed = set()
            if op not in preserve:
                for name in set(units) | set(descriptions):
                    if name not in data.columns or name not in before.columns:
                        continue
                    changed = _column_changed(before, data, name, key if current_key == key else [])
                    if changed:
                        transformed.add(name)
                    if changed and name in units and name not in declared_units:
                        raise WrangleError("UNDECLARED_UNITS", "A transformed measurement needs explicit output units.", {"column": name, "input_unit": units[name]})
                    if changed and name in descriptions and name not in declared_descriptions:
                        descriptions.pop(name, None)
            conflicts = {name: {"observed": units[name], "declared": value} for name, value in declared_units.items() if name in units and units[name] != value and name not in transformed}
            if conflicts:
                raise WrangleError("UNIT_MISMATCH", "Declared output units must agree with preserved or computed units; convert measurements explicitly.", {"columns": conflicts})
            units.update(declared_units)
            require_columns(data, list(declared_descriptions))
            descriptions.update(declared_descriptions)
            units = {name: unit for name, unit in units.items() if name in data.columns}
            descriptions = {name: meaning for name, meaning in descriptions.items() if name in data.columns}
            validate_unit_references(data, units)
            validate_unit_basis(before, data, op, before_units, units, current_key)
            _checks(data, {}, key)
            transition = observation_transition(before, data, op, effect_step, current_key, key, aggregates=contract["aggregates"], sources=[input_name, *source_names], right=source_data[source_names[0]] if op == "join" else None, evidence=_workspace)
            excluded, observation_counts = observation_effects(before, data, op, effect_step, current_key, aggregates=contract["aggregates"], right=source_data[source_names[0]] if op == "join" else None, evidence=_workspace)
            first_parameter = next(iter(signatures.signature(function).parameters))
            resolved = {name: value for name, value in bound.arguments.items() if name not in {first_parameter, "source", "sources"}}
            if op in {"join", "join_asof"}:
                resolved["source"] = source_names[0]
            if op == "concat":
                resolved["sources"] = source_names
            receipt_step = {"step": index, "op": op, "parameters": resolved, "reason": step.get("reason"), "rows_before": before.height, "rows_after": data.height, "columns_added": [name for name in data.columns if name not in before.columns], "columns_removed": [name for name in before.columns if name not in data.columns], "input_sha256": _digest(before), "output_sha256": _digest(data), "excluded_keys": excluded, "observation_counts": observation_counts}
            if transition:
                receipt_step["observation_transition"] = transition
            if op == "normalize_missing":
                fields = list(step["columns"])
                if fields:
                    old_counts = before.lazy().select([(pl.col(name).is_null().sum().cast(pl.Int64) - (floating_count(pl.col(name), before.schema[name], kind="nan").sum().cast(pl.Int64) if isinstance(before.schema[name], (pl.Struct, pl.List, pl.Array)) and step.get("nan", True) else pl.lit(0))).alias(f"old_{i}") for i, name in enumerate(fields)])
                    new_counts = data.lazy().select([pl.col(name).is_null().sum().cast(pl.Int64).alias(f"new_{i}") for i, name in enumerate(fields)])
                    counts = collect(old_counts.join(new_counts, how="cross").select([(pl.col(f"new_{i}") - pl.col(f"old_{i}")).alias(name) for i, name in enumerate(fields)]))
                    receipt_step["replacements"] = counts.row(0, named=True)
                else:
                    receipt_step["replacements"] = {}
            receipts.append(receipt_step)
        except WrangleError as error:
            error.details = {**error.details, "step": index, "operation": op}
            raise
        except (pl.exceptions.PolarsError, TypeError, ValueError, KeyError) as error:
            raise WrangleError("STEP_FAILED", "The operation could not satisfy its declared parameters.", {"step": index, "operation": op, "error": str(error)}) from error
    _finite_output(data)
    checks = _checks(data, rules, key, sources=source_data, units=units, descriptions=descriptions)
    receipt = {"wrangle_version": VERSION, "polars_version": pl.__version__, "environment": {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(), "polars_threads": pl.thread_pool_size()}, "hash_format": "logical-native-json-v1", "recipe": recipe, "recipe_sha256": hashlib.sha256(_json(recipe).encode()).hexdigest(), "sources": source_info, "input": input_name, "key": key, "units": units, "units_unresolved": [name for name, dtype in data.schema.items() if dtype.is_numeric() and name not in units and name not in key], "descriptions_unresolved": [name for name in data.columns if name not in descriptions], "steps": receipts, "checks": checks, "variables": {name: {"dtype": str(dtype), "unit": units.get(name), "description": descriptions.get(name)} for name, dtype in data.schema.items()}, "output": {"rows": data.height, "columns": {name: str(dtype) for name, dtype in data.schema.items()}, "sha256": _digest(data)}}
    receipt["execution"] = {"storage": "disk" if _workspace is not None else "memory", "engine": "streaming" if _workspace is not None else "auto"}
    if _workspace is not None:
        receipt["evidence_files"] = _workspace.evidence_files
    if variable_units:
        receipt["variable_units"] = variable_units
    # Hash the same canonical value order that is retained and rendered later.
    receipt = json.loads(_json(receipt))
    from ._presentation import render_prepare
    receipt["report_sha256"] = hashlib.sha256((render_prepare(receipt) + "\n").encode("utf-8")).hexdigest()
    receipt = json.loads(_json(receipt))
    if _workspace is not None:
        return _workspace.publish(data, receipt)
    prepared = Prepared(data, receipt)
    if output is not None:
        prepared.write(output)
    return prepared
